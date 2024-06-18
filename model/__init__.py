import re
import os
import sys
import time
import inspect
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

from model import util as model_util
from arguments import ArgumentBuilder
from container import Container

import model.mytorch.util as pyt_util
import model.mytorch_geometric.util as pyg_util
from model.mytorch.nn import kLinear, gLinear, kgLinear, vwLinear, kvwLinear, chebvwLinear
from model.mytorch.nn import mLSTMCell, gGRUCell, gLSTMCell
from model.mytorch_geometric.nn import gcSoftmax
from model.mytorch_geometric.nn.conv.flow_conv import FlowConv as _FlowConv
from model.mytorch_geometric.nn.conv.message_passing import MessagePassing

from model.AGCRN_PyTorch.model.AGCRNCell import AGCRNCell
from model.AGCRN_PyTorch.model.AGCN import AVWGCN as agcnLinear

import util


class Model(torch.nn.Module):

    # Init str -> function maps
    act_fn_map = model_util.act_fn_map
    opt_fn_map = model_util.opt_fn_map
    sched_fn_map = model_util.sched_fn_map
    init_fn_map = model_util.init_fn_map
    loss_fn_map = model_util.loss_fn_map
    layer_fn_map = model_util.layer_fn_map
    gcnlayer_fn_map = model_util.gcnlayer_fn_map
    gcnlayer_supported_map = model_util.gcnlayer_supported_map
    gcnlayer_required_map = model_util.gcnlayer_required_map
    # Other variables
    train_losses, valid_losses, test_losses = [-1], [-1], [-1]
    warnings = 0
    debug = 0
    to_gpu_time, to_cpu_time = 0, 0
    all_to_gpu, mbatches_to_gpu = True, False

    def forward(self, inputs):
        fn_name = "%s.%s" % (self.__class__.__name__, inspect.currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (fn_name))
        return a

    def optimize(self, train, valid, test, var):
        time_elapsed, last_time = 0, time.time()
#        torch.use_deterministic_algorithms(mode=True, warn_only=True)
#        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.all_to_gpu = "cuda" in var.device and var.gpu_data_mapping == "all"
        self.mbatches_to_gpu = "cuda" in var.device and var.gpu_data_mapping == "minibatch"
        # Reformat data to accepted shape, convert to tensors, put data + model onto device, then unpack
        self, train = self.prepare(train, var.device)
        self, valid = self.prepare(valid, var.device)
        self, test = self.prepare(test, var.device)
        # Initialize loaders, loss, optimizer, parameters, etc
        train_iterable = self.LoaderDatasetClass()(train, var)
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(
                train_iterable,
                generator=torch.Generator().manual_seed(
                    time.time() if var.batch_shuffle_seed is None else var.batch_shuffle_seed
                )
            ),
            var.mbatch_size,
            True
        )
        train_loader = torch.utils.data.DataLoader(
            train_iterable,
            sampler=sampler,
            collate_fn=pyt_util.batch_sampler_collate,
        )
        if not valid is None:
            valid_iterable = self.LoaderDatasetClass()(valid, var)
            sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(valid_iterable),
                var.mbatch_size,
                False
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_iterable,
                sampler=sampler,
                collate_fn=pyt_util.batch_sampler_collate,
            )
        if not test is None:
            test_iterable = self.LoaderDatasetClass()(test, var)
            sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(test_iterable),
                var.mbatch_size,
                False
            )
            test_loader = torch.utils.data.DataLoader(
                test_iterable,
                sampler=sampler,
                collate_fn=pyt_util.batch_sampler_collate,
            )
        self.criterion = self.init_criterion(var)
        self.optimizer = self.init_optimizer(var)
        self.scheduler = self.init_scheduler(self.optimizer, var)
        max_resets, patience = 0, var.patience
        if isinstance(var.patience, (tuple, list)):
            max_resets, patience = var.patience
        self.stopper = model_util.EarlyStopper(patience)
        self.init_params(var.initializer, var.initialization_seed)
        if var.log_to_tensorboard:
            self.summary_writer = pyt_util.SummaryWriter(var.checkpoint_dir)
        if var.log_to_tensorboard and var.tensorboard.log_text:
            self.summary_writer.add_text("Model", str(self).replace("\n", "\n\n"))
        # Begin optimization
        print(110 * "+")
        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        min_valid_loss = sys.float_info.max
        halt_code = 0
        reset_count = 0
        epoch = 0
        while epoch <= var.n_epochs:
            # Training set pass
            self.train()
            epoch_loss, n_sample = 0, 0
            self.pre_epoch_update(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
            for mb_in in train_loader: # foreach training mini-batch
                mb_in = util.merge_dicts(mb_in, train, False)
                if self.mbatches_to_gpu:
                    start = time.time()
                    for key in mb_in.keys():
                        mb_in[key] = util.to_device(mb_in[key], util.get_device(var.device))
                    self.to_gpu_time += time.time() - start
                mb_out = self.forward(mb_in) # model forward
                if self.mbatches_to_gpu:
                    start = time.time()
                    for key in mb_in.keys():
                        mb_in[key] = util.to_device(mb_in[key], util.get_device("cpu"))
                    if mb_out["yhat"].device != mb_in["y"].device:
                        mb_out["yhat"] = util.to_device(mb_out["yhat"], mb_in["y"].device)
                    self.to_cpu_time += time.time() - start
                mb_loss = self.loss(mb_in, mb_out, var) # compute loss
                if epoch > 0: # take SGD step
                    self.pre_step(mb_loss, mb_in, mb_out, var)
                    self.step(mb_loss, mb_in, mb_out, var)
                    self.post_step(mb_loss, mb_in, mb_out, var)
                epoch_loss += self.loss_to_numeric(mb_loss, var)
                # check for timeout
                time_elapsed += time.time() - last_time
                if var.train_timeout > -1 and time_elapsed > var.train_timeout:
                    halt_code = 1
                    break
                last_time = time.time()
            epoch_loss /= len(train_loader)
            self.train_losses.append(epoch_loss)
            self.eval()
            self.post_epoch_update(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
            # Validation set pass
            if not (valid is None or halt_code):
                with torch.set_grad_enabled(False):
                    epoch_loss, n_sample = 0, 0
                    for mb_in in valid_loader:
                        mb_in = util.merge_dicts(mb_in, valid, False)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device(var.device))
                            self.to_gpu_time += time.time() - start
                        mb_out = self.forward(mb_in)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device("cpu"))
                            if mb_out["yhat"].device != mb_in["y"].device:
                                mb_out["yhat"] = util.to_device(mb_out["yhat"], mb_in["y"].device)
                            self.to_cpu_time += time.time() - start
                        mb_loss = self.loss(mb_in, mb_out, var)
                        epoch_loss += self.loss_to_numeric(mb_loss, var)
                    epoch_loss /= len(valid_loader)
                    self.valid_losses.append(epoch_loss)
                    # Check for improvement and update early stopping
                    if epoch > 0 and epoch_loss < min_valid_loss: 
                        self.checkpoint(os.sep.join([var.checkpoint_dir, "Best.pth"]))
                        min_valid_loss = epoch_loss
            # Testing set pass
            if not (test is None or halt_code):
                with torch.set_grad_enabled(False):
                    epoch_loss, n_sample = 0, 0
                    for mb_in in test_loader:
                        mb_in = util.merge_dicts(mb_in, test, False)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device(var.device))
                            self.to_cpu_time += time.time() - start
                        mb_out = self.forward(mb_in)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device("cpu"))
                            if mb_out["yhat"].device != mb_in["y"].device:
                                mb_out["yhat"] = util.to_device(mb_out["yhat"], mb_in["y"].device)
                            self.to_cpu_time += time.time() - start
                        mb_loss = self.loss(mb_in, mb_out, var)
                        epoch_loss += self.loss_to_numeric(mb_loss, var)
                    epoch_loss /= len(test_loader)
                    self.test_losses += [epoch_loss]
            if halt_code:
                break
            # update stopper and scheduler
            if epoch > 0:
                self.stopper.step(self.valid_losses[-1])
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.valid_losses[-1])
                elif isinstance(self.scheduler, model_util.TriggerScheduler):
                    self.scheduler.step(self.stopper)
                else:
                    self.scheduler.step()
            # print epoch results
            train_loss = self.train_losses[-1]
            valid_loss = (-1 if valid is None else self.valid_losses[-1])
            test_loss = (-1 if test is None else self.test_losses[-1])
            best_mark = ("*" if epoch > 0 and valid_loss == min_valid_loss else "")
            print(
                "Epoch %3.d [%9.1fs]: Train=[%.5f], Valid=[%.5f], Test=[%.5f] %s" % (
                    epoch, time_elapsed, train_loss, valid_loss, test_loss, best_mark
                )
            )
            # handle halt cases and logging
            if np.isnan(train_loss) or np.isnan(valid_loss) or np.isnan(test_loss):
                halt_code = 2
                break
            if var.log_to_tensorboard:
                self.log_epoch_info(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
            if self.stopper.stop():
                if reset_count >= max_resets:
                    break
                print("******************************RESET******************************")
                self.load(os.sep.join([var.checkpoint_dir, "Best.pth"]), var)
                self.stopper.reset()
                reset_count += 1
                epoch -= patience
            epoch += 1
#            print("Optimizer LRs:", [_["lr"] for _ in self.optimizer.param_groups])
        print(110 * "+")
        if var.log_to_tensorboard:
            self.summary_writer.close()
        # Save final model
        path = os.sep.join([var.checkpoint_dir, "Final.pth"])
        self.checkpoint(path)
        # Return data to original device (cpu), shape, and type (NumPy.ndarray)
        self, train = self.prepare(train, False, True)
        self, valid = self.prepare(valid, False, True)
        self, test = self.prepare(test, False, True)
        # Handle halt codes
        if halt_code & 1:
            warnings.warn("Model Optimization: Training runtime (%.1fs) exceeded timeout duration (%.1fs). Halting optimization." % (time_elapsed, var.train_timeout))
        if halt_code & 2:
            warnings.warn("Model Optimization: Found nan in training, validation, and/or testing loss at epoch %d. Optimization has failed and will now be halted." % (epoch))
        # Finish
        return halt_code

    def predict(self, data, var):
        self.all_to_gpu = "cuda" in var.device and var.get("gpu_data_mapping") == "all"
        self.mbatches_to_gpu = "cuda" in var.device and var.get("gpu_data_mapping") == "minibatch"
        self, data = self.prepare(data, var.device)
        # Initialize loader
        data_iterable = self.LoaderDatasetClass()(data, var)
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(data_iterable),
            var.mbatch_size,
            False
        )
        loader = torch.utils.data.DataLoader(
            data_iterable, sampler=sampler, collate_fn=pyt_util.batch_sampler_collate
        )
        # Commence prediction
        self.eval()
        outputs = {}
        with torch.set_grad_enabled(False):
            if isinstance(data_iterable, model_util.PartitionLoader) and data_iterable.__sampled__ == []: # semi-supervised
                mb_in = util.merge_dicts(next(iter(loader)), data, False)
                mb_out = self.forward(mb_in)
                yhat = mb_out["yhat"][mb_in["indices"]]
                outputs = util.merge_dicts(outputs, mb_out)
            else:
                yhats = []
                for mb_in in loader:
                    mb_in = util.merge_dicts(mb_in, data, False)
                    mb_out = self.forward(mb_in)
                    yhats.append(mb_out["yhat"])
                    outputs = util.merge_dicts(outputs, mb_out)
                if isinstance(yhats[0], np.ndarray):
                    yhat = np.concatenate(yhats)
                elif isinstance(yhats[0], torch.Tensor):
                    yhat = torch.cat(yhats)
                else:
                    raise ValueError("Cannot concatenate predicted elements of type %s" % (type(yhats[0])))
            outputs["yhat"] = yhat
        self, data = self.prepare(data, False, True)
        self, outputs = self.prepare(outputs, False, True)
        return outputs

    def init_criterion(self, var):
        kwargs = util.copy_dict(var.loss_kwargs)
        if kwargs.pop("mask_loss", False):
            kwargs["reduction"] = "none"
        return self.loss_fn_map[var.loss](**kwargs)

    def init_optimizer(self, var):
        name_param_map = dict(self.named_parameters())
        param_groups = []
        for name, param in self.named_parameters():
            param_group = {"params": param}
            for regex_name, lr in var.param_lr_map.items():
                if re.match(regex_name, name):
                    param_group["lr"] = lr
                    print(name, lr)
            param_groups.append(param_group)
        opt = self.opt_fn_map[var.optimizer](
            param_groups,
            lr=var.lr,
            weight_decay=var.l2_reg
        )
        return opt

    def init_scheduler(self, optimizer, var):
        if var.lr_scheduler is None:
            class Scheduler:
                def __init__(self): pass
                def step(self): pass
            return Scheduler()
        return self.sched_fn_map[var.lr_scheduler](optimizer, last_epoch=-1, **var.lr_scheduler_kwargs)

    def init_params(self, init, seed=None):
        return model_util.init_params(self, init, seed)

    def init_params_from_model(self, model):
        name_param_map = dict(self.named_parameters())
        for name, param in model.named_parameters():
            if name in name_param_map:
                self_param = name_param_map[name]
                if self_param.shape != param.shape:
                    raise ValueError("Shape mismatch for coincident parameter %s from initializer model %s. Expected parameter %s.shape=%s but received %s.shape=%s" % (
                        name, model.name(), name, self_param.shape, name, param.shape
                        )
                    )
                self_param.data.copy_(param.data)

    def reset_parameters(self):
        model_util.reset_parameters(self)

    def pre_epoch_update(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        pass

    def post_epoch_update(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        pass

    def compute_gradients(self, loss, mb_in, mb_out, var):
        loss.backward()

    def pre_step(self, loss, mb_in, mb_out, var):
        pass

    # Takes a step of gradient descent
    def step(self, loss, mb_in, mb_out, var):
        self.optimizer.zero_grad()
        self.compute_gradients(loss, mb_in, mb_out, var)
        if not var.grad_clip is None:
            if var.grad_clip == "norm":
                torch.nn.utils.clip_grad_norm_(self.parameters(), **var.grad_clip_kwargs)
            elif var.grad_clip == "value":
                torch.nn.utils.clip_grad_value_(self.parameters(), **var.grad_clip_kwargs)
            else:
                raise ValueError(var.grad_clip)
        self.optimizer.step()

    def post_step(self, loss, mb_in, mb_out, var):
        pass

    # Computes the loss
    def loss(self, mb_in, mb_out, var):
        mask = mb_in.get("loss_mask", None)
        mask_loss = var.loss_kwargs.get("mask_loss", False)
        #
        L = self.criterion(mb_out["yhat"], mb_out.get("y", mb_in["y"]))
        if mask_loss:
            if mask is None:
                raise ValueError("Trying to mask loss with mask=%s." % (mask))
            L = (L * mask.float()).sum() / mask.sum()
        return L

    def loss_to_numeric(self, loss, var):
        return loss.item()

    def log_epoch_info(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        # Add loss scalars to tensorboard
        self.log_epoch_losses(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
        # Add weight histograms to tensorboard
        self.log_epoch_params(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
        # Add computational graph to tensorboard
        self.log_compgraph(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)

    def log_epoch_losses(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        if not var.tensorboard.log_losses:
            return
        self.summary_writer.add_scalar("Training Loss", self.train_losses[epoch], epoch)
        if len(self.valid_losses) >= epoch:
            self.summary_writer.add_scalar("Validation Loss", self.valid_losses[epoch], epoch)
        if len(self.valid_losses) >= epoch:
            self.summary_writer.add_scalar("Testing Loss", self.test_losses[epoch], epoch)

    def log_epoch_params(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        if not var.tensorboard.log_parameters:
            return
        for name, param in self.named_parameters():
            self.summary_writer.add_histogram(name, param, epoch)

    def log_compgraph(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        if not var.tensorboard.log_computational_graph:
            return
        if epoch == var.n_epochs:
            mb_in = util.merge_dicts(next(iter(train_loader)), train, False)
            _del = ["__index__", "__sampled__"]
            for key, value in mb_in.items(): # Trace requires all inputs to be the same type (Tensor)
                if not isinstance(value, torch.Tensor):
                    try:
                        mb_in[key] = util.to_tensor(value, torch.float)
                    except:
                        _del.append(key)
            for key in _del:
#                print("Deleting \"%s\" from the minibatch as it could not converted to tensor" % (key))
                if key in mb_in:
                    del mb_in[key]
            warnings.filterwarnings("ignore", "^Converting a tensor to a.*")
            warnings.filterwarnings("ignore", "^TracerWarning*")
            self.summary_writer.add_graph(self, mb_in, trace_kwargs={"strict": False})

    # Data elements pulled from this dataset for optimization and prediction tasks
    def pull_data(self, dataset, partition, var):
        def print_dict(d):
            for key, val in d.items():
                if isinstance(val, np.ndarray):
                    print(key, val.shape)
                else:
                    print(key, val)
        common_data = self.pull_common_data(dataset, partition, var)
        model_data = self.pull_model_data(dataset, partition, var)
        data = util.merge_dicts(common_data, model_data)
        #
        csd, msd = common_data.get("__sampled__", None), model_data.get("__sampled__", None)
        cid, mid = common_data.get("__inputs__", None), model_data.get("__inputs__", None)
        cod, mod = common_data.get("__outputs__", None), model_data.get("__outputs__", None)
        #    handle __sampled__ lists
        if not (csd is None or msd is None):
            data["__sampled__"] = list(set(csd + msd))
        elif not csd is None:
            data["__sampled__"] = csd
        elif not msd is None:
            data["__sampled__"] = msd
        #    handle __inputs__ lists
        if not (cid is None or mid is None):
            data["__inputs__"] = list(set(cid + mid))
        elif not cid is None:
            data["__inputs__"] = cid
        elif not mid is None:
            data["__inputs__"] = mid
        #    handle __outputs__ lists
        if not (cod is None or mod is None):
            data["__outputs__"] = list(set(cod + mod))
        elif not cod is None:
            data["__outputs__"] = cod
        elif not mod is None:
            data["__outputs__"] = mod
        if self.debug:
            print(util.make_msg_block("%s data" % (partition)))
            print_dict(data)
        return self.verify_pulled_data(data, var)

    # Data used by everything
    def pull_common_data(self, dataset, partition, var):
        princ_data = dataset.get(var.principle_data_type)
        data = {}
        if var.loss_kwargs.get("mask_loss", False):
            data["loss_mask"] = princ_data.get(var.principle_data_form).get("response_gtmask", partition)
            data["__sampled__"] = ["loss_mask"]
            data["__outputs__"] = ["loss_mask"]
        return data

    # Data used by this model
    def pull_model_data(self, dataset, partition, var):
        princ_data = dataset.get(var.principle_data_type)
        data = {}
        data["__sampled__"] = ["x", "y"]
        data["__outputs__"] = ["y"]
        data["x"] = princ_data.transformed.get(var.principle_data_form).get("predictor_features", partition)
        data["y"] = princ_data.transformed.get(var.principle_data_form).get("response_features", partition)
        data["n_temporal_out"] = var.temporal_mapping[1]
        return data

    # Check for data validity
    def verify_pulled_data(self, data, var):
        return data

    # Defines what must happen before optimization/prediction tasks can begin
    def prepare(self, data, device, revert=False):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.prepare_model(device, revert), self.prepare_data(data, device, revert)

    # Defines model preparation for optimization/prediction tasks
    def prepare_model(self, device, revert=False):
        if revert:
            self = util.to_device(self, util.get_device("cpu"))
        else:
            self = util.to_device(self, util.get_device(device))
        return self

    # Defines data preparation for optimization/prediction tasks
    def prepare_data(self, data, device, revert=False):
        def get_type(value):
            if isinstance(value, np.ndarray):
                if issubclass(value.dtype.type, np.integer):
                    _type = torch.long
                elif issubclass(value.dtype.type, np.floating):
                    _type = torch.float
                elif issubclass(value.dtype.type, np.bool_):
                    _type = torch.bool
                else:
                    raise ValueError(
                        "Do not know how to prepare \"%s\" which has data-type %s" % (key, type(value))
                    )
            elif isinstance(value, torch.Tensor):
                _type = value.dtype
            elif isinstance(value, list):
                _type = [get_type(_value) for _value in value]
            elif isinstance(value, int):
                _type = torch.long
            elif isinstance(value, float):
                _type = torch.float
            elif isinstance(value, str):
                _type = None
            elif value is None:
                _type = None
            else:
                raise ValueError("Do not know how to prepare \"%s\" which has data-type %s" % (key, type(value)))
            return _type
        if revert: # Back to CPU
            for key, value in data.items():
                device = util.get_device("cpu")
                if isinstance(value, torch.Tensor): # PyTorch.Tensor: put on CPU and cast to NumPy.ndarray
                    value = util.to_ndarray(util.to_device(value, device))
                else: # Everything else: do nothing
                    pass
                data[key] = value
        else: # To GPU or keep on CPU
            for key, value in data.items():
                if key in ["__index__", "__sampled__", "__inputs__", "__outputs__", "labels"] or callable(value):
                    continue
                # Determine the type for this data
                _type = get_type(value)
                # Determine the device for this data
                _device = "cpu" if not self.all_to_gpu else device
                if self.mbatches_to_gpu:
                    _device = "cpu" if key in ["x", "y", "xst", "y_st"] else device
                data[key] = util.to_device(util.to_tensor(value, _type), util.get_device(_device))
        return data

    def LoaderDatasetClass(self):
        return model_util.SlidingWindowLoader

    def load(self, chkpt_path, var):
        if not os.path.exists(chkpt_path):
            raise FileNotFoundError(chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location=util.get_device(var.device))
        self.criterion = self.init_criterion(var)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = self.opt_fn_map[var.optimizer](self.parameters(), var.lr)
#       self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["losses"]["train"]
        self.valid_losses = checkpoint["losses"]["valid"]
        self.test_losses = checkpoint["losses"]["test"]
        return self

    def checkpoint(self, path):
        chkpt_dict = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": {"train": self.train_losses, "valid": self.valid_losses, "test": self.test_losses}
        }
        torch.save(chkpt_dict, path)

    def get_id_var(self, var):
        names = [pair[0] for pair in self.id_pairs]
        partitions = [pair[1] for pair in self.id_pairs]
        id_var = var.checkout(names, partitions)
        return id_var

    def curate_id_str(self, var):
        return ArgumentBuilder().view(self.get_id_var(var))

    def get_info_var(self):
        info_names = ["n_parameters"]
        info_values = [self.n_params()]
        info_var = Container().set(info_names, info_values, multi_value=True)
        return info_var

    def curate_info_str(self):
        return ArgumentBuilder().view(self.get_info_var())

    def get_id(self, var, n_id_digits):
        return self.get_id_var(var).hash(n_id_digits)

    def n_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        tab_space = 2 * " "
        lines = []
        lines.append("%s(" % (self.name()))
        modules_str = []
        for name, module in self._modules.items():
            modules_str.append("%s(%s): %s" % (tab_space, name, repr(module).replace("\n", "\n"+tab_space)))
        modules_str = "\n".join(modules_str)
        lines.append(modules_str)
        lines.append(")")
        lines.append("Hyperparameters(")
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "config_name_partition_pairs":
                continue
            str_value = str(value)
            lines.append("%s%s = %s" % (tab_space, name, str_value))
        lines.append(")")
        return "\n".join(lines)


class MLP(torch.nn.Module):

    debug = 0

    def __init__(self, in_size, out_size, hidden_size=16, lin_kwargs={}, act_layer="ReLU", act_kwargs={}, dropout=0.0):
        super(MLP, self).__init__()
        #
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        elif not isinstance(hidden_size, (tuple, list)):
            raise ValueError(type(hidden_size))
        n_layer = len(hidden_size)
        if isinstance(lin_kwargs, dict):
            lin_kwargs = [lin_kwargs] + [lin_kwargs for i in range(n_layer)]# + [lin_kwargs]
        elif not isinstance(lin_kwargs, (tuple, list)):
            raise ValueError(type(lin_kwargs))
        if isinstance(act_layer, str):
            act_layer = [act_layer] + [act_layer for i in range(n_layer)]# + [act_layer]
        elif not isinstance(act_layer, (tuple, list)):
            raise ValueError(type(act_layer))
        if isinstance(act_kwargs, dict):
            act_kwargs = [act_kwargs] + [act_kwargs for i in range(n_layer)]# + [act_kwargs]
        elif not isinstance(act_kwargs, (tuple, list)):
            raise ValueError(type(act_kwargs))
        if isinstance(dropout, float):
            dropout = [dropout] + [dropout for i in range(n_layer)]# + [dropout]
        elif not isinstance(dropout, (tuple, list)):
            raise ValueError(type(dropout))
        #
        self.lins = [torch.nn.Linear(in_size, hidden_size[0], **lin_kwargs[0])]
        self.lins += [
            torch.nn.Linear(hidden_size[i-1], hidden_size[i], **lin_kwargs[i]) for i in range(1, n_layer)
        ]
        self.lins.append(torch.nn.Linear(hidden_size[-1], out_size, **lin_kwargs[-1]))
        self.acts = [model_util.layer_fn_map[act_layer[0]](**act_kwargs[0])]
        self.acts += [model_util.layer_fn_map[act_layer[i]](**act_kwargs[i]) for i in range(1, n_layer)]
        self.acts.append(torch.nn.Identity())
        self.drops = [torch.nn.Dropout(dropout[0])]
        self.drops += [torch.nn.Dropout(dropout[i]) for i in range(1, n_layer)]
        self.drops.append(torch.nn.Dropout(0.0))
        #
        self.lins = torch.nn.ModuleList(self.lins)
        self.acts = torch.nn.ModuleList(self.acts)
        self.drops = torch.nn.ModuleList(self.drops)
        #
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer

    def forward(self, x, **kwargs):
        layer_in = kwargs.get("layer_in", 0)
        layer_out = kwargs.get("layer_out", -1)
        if layer_out < 0:
            layer_out += self.n_layer + 2
        #
        if self.debug:
            print(util.make_msg_block("MLP Forward"))
        if self.debug:
            print("x =", x.shape)
            if self.debug > 1:
                print(x)
            print("layer_in =", layer_in)
            print("layer_out =", layer_out)
        #
        a = x
        for i in range(layer_in, layer_out):
            z = self.lins[i](a)
            a = self.acts[i](z)
            a = self.drops[i](a)
            if self.debug:
                print("a[%d] =" % (i), a.shape)
                if self.debug > 1:
                    print(a)
        return {"yhat": a}

    def __len__(self):
        return self.n_layer


class gcLinear(MessagePassing):

    debug = 0

    def __init__(self, in_size, out_size, conv="std", layer="Linear", bias=True, order=1, n_hops=1, **kwargs):
        super(gcLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.conv = conv
        self.layer = layer
        self.bias = bias
        self.order = order
        self.n_hops = n_hops
        self.kwargs = kwargs
        self.conv_forward = getattr(self, "%s_conv" % (conv))
        getattr(self, "%s_init" % (layer))(in_size, out_size, bias, order, n_hops, **kwargs)
        self.layer_forward = getattr(self, "%s_forward" % (layer))

    def Linear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = torch.nn.Linear(in_size, out_size, bias)
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = torch.nn.Linear(in_size, out_size, bias)

    def kLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = kLinear(in_size, out_size, bias, n_hops)
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = kLinear(in_size, out_size, bias, n_hops)

    def gLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = gLinear(in_size, out_size, bias, kwargs["n_groups"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = gLinear(in_size, out_size, bias, kwargs["n_groups"])

    def kgLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = kgLinear(in_size, out_size, bias, n_hops, kwargs["n_groups"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = kgLinear(in_size, out_size, bias, n_hops, kwargs["n_groups"])

    def vwLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = vwLinear(in_size, out_size, bias, kwargs["embed_size"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = vwLinear(in_size, out_size, bias, kwargs["embed_size"])

    def kvwLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if self.conv == "cheb": n_hops = max(n_hops, 1)
        if order & 1:
            self.lin_i = kvwLinear(in_size, out_size, bias, n_hops, kwargs["embed_size"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = kvwLinear(in_size, out_size, bias, n_hops, kwargs["embed_size"])

    def chebvwLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = chebvwLinear(in_size, out_size, bias, n_hops, kwargs["embed_size"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = chebvwLinear(in_size, out_size, bias, n_hops, kwargs["embed_size"])
        self._n_hops = n_hops
        self.n_hops = 0

    def agcnLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = agcnLinear(in_size, out_size, n_hops, kwargs["embed_size"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = agcnLinear(in_size, out_size, n_hops, kwargs["embed_size"])
        self._n_hops = n_hops
        self.n_hops = 0

    def forward(self, x, edge_index, edge_weight=None, frmt="?", **kwargs):
        if self.debug:
            print(util.make_msg_block("gcLinear Forward"))
            print("x =", x.shape)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if hasattr(self, "lin_i"):
                self.lin_i.debug = self.debug
            if hasattr(self, "lin_o"):
                self.lin_o.debug = self.debug
        # Start
        x, edge_index, edge_weight = self.forward_prehook(x, edge_index, edge_weight, **kwargs)
        V = x.shape[-2]
        if not edge_index is None:
            if frmt == "?":
                if edge_index.shape[-2] == V and edge_index.shape[-1] == V:
                    frmt = "adj"
                elif edge_index.shape[-2] == 2:
                    frmt = "coo"
                else:
                    raise ValueError("Cannot determine format of edge_index with shape=%s" % (edge_index.shape))
        #   Apply linear layer before graph conv
        if self.order & 1:
            if isinstance(self.lin_i, chebvwLinear):
                x = self.layer_forward(self.lin_i, x, A=edge_index, **kwargs)
            else:
                x = self.layer_forward(self.lin_i, x, **kwargs)
        #    Apply graph conv
        x = self.conv_forward(x, edge_index, edge_weight, frmt, **kwargs)
        if self.order & 2:
            if isinstance(self.lin_o, chebvwLinear):
                x = self.layer_forward(self.lin_o, x, A=edge_index, **kwargs)
            else:
                x = self.layer_forward(self.lin_o, x, **kwargs)
        #    Apply linear layer after graph conv
        if self.debug:
            print("*** gcLinear Output ***")
            print("x =", x.shape)
        #   Cleanup
        x, edge_index, edge_weight = self.forward_hook(x, edge_index, edge_weight, **kwargs)
        return x

    def forward_prehook(self, x, edge_index, edge_weight, **kwargs):
        self.__prefrwd_xshape__ = x.shape
        return x, edge_index, edge_weight

    def forward_hook(self, x, edge_index, edge_weight, **kwargs):
        self.__postfrwd_xshape__ = x.shape
        if self.debug > 1:
            print("prefrwd_xshape =", self.__prefrwd_xshape__)
            print("postfrwd_xshape =", self.__postfrwd_xshape__)
        if x.dim() > len(self.__prefrwd_xshape__):
            if self.conv in ["cheb"]: # may require reduction
                x = torch.sum(x, -3)
            else:
                raise NotImplementedError(self.conv)
        return x, edge_index, edge_weight

    def identity_conv(self, x, edge_index, edge_weight, frmt, **kwargs):
        return x

    def std_conv(self, x, edge_index, edge_weight, frmt, **kwargs):
        if not edge_index is None:
            if frmt == "coo":
                x, edge_index = pyt_util.align((x, edge_index), -1)
                for i in range(self.n_hops):
                    x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                x = torch.squeeze(x, 0)
            elif frmt == "adj":
                for i in range(self.n_hops):
                    x = torch.einsum("...NM,...MI->...NI", edge_index, x)
        return x

    def cheb_conv(self, x, edge_index, edge_weight, frmt, **kwargs):
        V = x.shape[-2]
        if edge_index is None:
            x = torch.stack([x for _ in range(self.n_hops)], -3)
        elif frmt == "coo":
            raise NotImplementedError(frmt)
        else:
            supports = [torch.eye(V, device=x.device)]
            if self.n_hops > 0:
                supports.append(edge_index)
            for k in range(2, self.n_hops):
                supports.append(torch.matmul(2 * edge_index, supports[-1]) - supports[-2])
            supports = torch.stack(supports, -3)
            x = torch.einsum("...KNM,...MI->...KNI", supports, x)
        return x

    def Linear_forward(self, layer, x, **kwargs):
        return layer(x)

    def kLinear_forward(self, layer, x, **kwargs):
        return layer(x)

    def gLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["group_index"])

    def kgLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["group_index"])

    def vwLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["embedding"])

    def kvwLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["embedding"])

    def chebvwLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["A"], kwargs["embedding"])

    def agcnLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["embedding"])

    def message(self, x_j, edge_weight):
        """
        Arguments
        ---------
        x_j : FloatTensor with shape=(?, |E|, F)
        edge_weight : (FloatTensor, optional) with shape=(?, |E|)

        Returns
        -------
        x : FloatTensor with shape=(?, |E|, F)

        """
        if edge_weight is None:
            return x_j
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        edge_weight = torch.unsqueeze(edge_weight, -1)
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        return edge_weight * x_j

    def reset_parameters(self):
        if self.order & 1:
            self.lin_i.reset_parameters()
        if self.order & 2:
            self.lin_o.reset_parameters()


class gcGRUCell(torch.nn.Module):

    def __init__(self, in_size, out_size, conv="std", layer="Linear", bias=True, order=1, n_hops=1, shared=True, **kwargs):
        super(gcGRUCell, self).__init__()
        print("gcGRUCell :", in_size, out_size, layer, bias, order, n_hops, shared, kwargs)
        # Layers
        if shared:
            self.lin_rz = gcLinear(in_size+out_size, 2*out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_n = gcLinear(in_size+out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
        else:
            self.lin_ir = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_iz = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_in = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hr = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hz = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hn = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
        # Save vars
        self.in_size = in_size
        self.out_size = out_size
        self.shared = shared
        self.order = order

    def forward(self, x, edge_index=None, edge_weight=None, prev_state=None, **kwargs):
        if self.shared:
            self.lin_rz.debug = self.debug
        else:
            self.lin_ir.debug = self.debug
        # Setup
        if prev_state is None:
            prev_state = torch.zeros((x.size(0), x.size(-2), self.out_size), dtype=x.dtype, device=x.device)
        h = prev_state
        if self.debug:
            print("x =", x.shape)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
            print("prev_state =", h.shape)
        # Start
        if self.shared:
            rz = torch.sigmoid(self.lin_rz(torch.cat((x, h), -1), edge_index, edge_weight, **kwargs))
            r, z = torch.split(rz, self.out_size, -1)
            n = torch.tanh(self.lin_n(torch.cat((x, r * h), -1), edge_index, edge_weight, **kwargs))
        else:
            r = torch.sigmoid(
                self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                self.lin_hr(h, edge_index, edge_weight, **kwargs)
            )
            z = torch.sigmoid(
                self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                self.lin_hz(h, edge_index, edge_weight, **kwargs)
            )
            n = torch.tanh(
                self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
            )
        _h = (1 - z) * n + z * h
        if self.debug:
            print(_h.shape)
        return _h


class gcLSTMCell(torch.nn.Module):

    def __init__(self, in_size, out_size, conv="std", layer="Linear", bias=True, order=1, n_hops=1, shared=True, **kwargs):
        super(gcLSTMCell, self).__init__()
        print("gcLSTMCell :", in_size, out_size, bias, order)
        # Input layers
        if shared:
            self.lin_ifo = gcLinear(in_size+out_size, 3*out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_g = gcLinear(in_size+out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
        else:
            self.lin_ii = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_if = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_ig = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_io = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hi = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hf = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hg = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_ho = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
        # Save vars
        self.in_size = in_size
        self.out_size = out_size
        self.shared = shared
        self.order = order

    def forward(self, x, edge_index, edge_weight=None, prev_state=None, **kwargs):
        if self.shared:
            self.lin_ifo.debug = self.debug
        else:
            self.lin_ii.debug = self.debug
        # Setup
        if prev_state is None:
            zeros = torch.zeros((x.size(0), x.size(-2), self.out_size), dtype=x.dtype, device=x.device)
            prev_state = (zeros, zeros)
        h, c = prev_state
        # Start
        if self.shared:
            xh = torch.cat((x, h), -1)
            ifo = torch.sigmoid(self.lin_ifo(xh, edge_index, edge_weight, **kwargs))
            i, f, o = torch.split(ifo, self.out_size, -1)
            g = torch.tanh(self.lin_g(xh, edge_index, edge_weight, **kwargs))
        else:
            i = torch.sigmoid(
                self.lin_ii(x, edge_index, edge_weight, **kwargs) + \
                self.lin_hi(h, edge_index, edge_weight, **kwargs)
            )
            f = torch.sigmoid(
                self.lin_if(x, edge_index, edge_weight, **kwargs) + \
                self.lin_hf(h, edge_index, edge_weight, **kwargs)
            )
            g = torch.tanh(
                self.lin_ig(x, edge_index, edge_weight, **kwargs) + \
                self.lin_hg(h, edge_index, edge_weight, **kwargs)
            )
            o = torch.sigmoid(
                self.lin_io(x, edge_index, edge_weight, **kwargs) + \
                self.lin_ho(h, edge_index, edge_weight, **kwargs)
            )
        _c = f * c + i * g
        _h = o * torch.tanh(_c)
        if self.debug:
            print(_h.shape, _c.shape)
        return (_h, _c)


class gcstRNNCell(torch.nn.Module):

    def __init__(self, in_size, out_size, xs_size=0, xt_size=0, conv="std", layer="Linear", bias=True, order=1, n_hops=1, shared=True, fusion="add", **kwargs):
        super(gcstRNNCell, self).__init__()
        print("gcstRNNCell :", in_size, out_size, xs_size, xt_size, layer, bias, order, n_hops, shared, kwargs)
        # Layers
        n_modal = 1
        if shared:
            self.lin_ih = gcLinear(
                in_size + xs_size + xt_size + out_size, out_size, 
                conv, layer, bias, order, n_hops, **kwargs
            )
        else:
            # layers for x
            self.lin_i = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for h
            self.lin_h = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for xs
            if xs_size > 0:
                self.lin_s = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                n_modal += 1
            # layers for xt
            if xt_size > 0:
                self.lin_t = torch.nn.Linear(xt_size, out_size, bias)
                n_modal += 1
            if fusion == "add":
                self.fuse_w = torch.ones((n_modal,))
            elif fusion == "attn":
                self.fuse_w = torch.nn.Parameter(torch.rand((n_modal,)))
            else:
                raise NotImplementedError(fusion)
        # Save vars
        self.in_size = in_size
        self.out_size = out_size
        self.xs_size = xs_size
        self.xt_size = xt_size
        self.shared = shared
        self.fusion = fusion
        self.n_modal = n_modal
        self.order = order

    def forward(self, x, edge_index=None, edge_weight=None, prev_state=None, **kwargs):
        if self.shared:
            self.lin_ih.debug = self.debug
        else:
            self.lin_i.debug = self.debug
        xs, xt = kwargs.get("xs", None), kwargs.get("xt", None)
        # Setup
        if prev_state is None:
            prev_state = torch.zeros((x.size(0), x.size(-2), self.out_size), dtype=x.dtype, device=x.device)
        h = prev_state
        if self.debug:
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
            print("prev_state =", h.shape)
        # Start
        if self.shared:
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, xt, x), -1)
            elif not xs is None and self.xs_size > 0:
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, x), -1)
            elif not xt is None and self.xt_size > 0:
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xt, x), -1)
            inner = self.lin_ih(torch.cat((x, h), -1), edge_index, edge_weight, **kwargs)
        else:
            s_w, t_w, st_w = self.get_fuse_w()
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                inner = (
                    s_w * self.lin_s(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_t(xt)[:,None,:] + \
                    st_w * self.lin_i(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_h(h, edge_index, edge_weight, **kwargs)
                )
            elif not xs is None and self.xs_size > 0:
                inner = (
                    s_w * self.lin_s(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_i(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_h(h, edge_index, edge_weight, **kwargs)
                )
            elif not xt is None and self.xt_size > 0:
                inner = (
                    t_w * self.lin_t(xt)[:,None,:] + \
                    st_w * self.lin_i(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_h(h, edge_index, edge_weight, **kwargs)
                )
            else:
                inner = (
                    self.lin_i(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_h(h, edge_index, edge_weight, **kwargs)
                )
        _h = torch.tanh(inner)
        if self.debug:
            print(_h.shape)
        return _h

    def get_fuse_w(self):
        fuse_w = self.fuse_w
        if self.fusion == "attn":
            fuse_w = torch.nn.functional.softmax(fuse_w, -1)
        if self.xs_size < 1 and self.xt_size < 1:
            fuse_w = (1, 1, fuse_w[0])
        elif self.xs_size < 1:
            fuse_w = (1, fuse_w[0], fuse_w[1])
        elif self.xt_size < 1:
            fuse_w = (fuse_w[0], 1, fuse_w[1])
        return fuse_w


class gcstGRUCell(torch.nn.Module):

    def __init__(self, in_size, out_size, xs_size=0, xt_size=0, conv="std", layer="Linear", bias=True, order=1, n_hops=1, shared=True, fusion="add", **kwargs):
        super(gcstGRUCell, self).__init__()
        print("gcstGRUCell :", in_size, out_size, xs_size, xt_size, layer, bias, order, n_hops, shared, kwargs)
        # Layers
        n_modal = 1
        if shared:
            self.lin_rz = gcLinear(
                in_size + xs_size + xt_size + out_size, 
                2 * out_size, 
                conv, layer, bias, order, n_hops, **kwargs
            )
            self.lin_n = gcLinear(
                in_size + xs_size + xt_size + out_size, 
                out_size, conv, layer, bias, order, n_hops, **kwargs
            )
        else:
            # layers for x
            self.lin_ir = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_iz = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_in = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for h
            self.lin_hr = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hz = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hn = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for xs
            if xs_size > 0:
                self.lin_sr = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                self.lin_sz = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                self.lin_sn = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                n_modal += 1
            # layers for xt
            if xt_size > 0:
                self.lin_tr = torch.nn.Linear(xt_size, out_size, bias)
                self.lin_tz = torch.nn.Linear(xt_size, out_size, bias)
                self.lin_tn = torch.nn.Linear(xt_size, out_size, bias)
                n_modal += 1
            if fusion == "add":
                self.fuse_w = torch.ones((n_modal,))
            elif fusion == "attn":
                self.fuse_w = torch.nn.Parameter(torch.rand((n_modal,)))
            else:
                raise NotImplementedError(fusion)
        # Save vars
        self.in_size = in_size
        self.out_size = out_size
        self.xs_size = xs_size
        self.xt_size = xt_size
        self.shared = shared
        self.fusion = fusion
        self.n_modal = n_modal
        self.order = order

    def forward(self, x, edge_index=None, edge_weight=None, prev_state=None, **kwargs):
        if self.shared:
            self.lin_rz.debug = self.debug
        else:
            self.lin_ir.debug = self.debug
        xs, xt = kwargs.get("xs", None), kwargs.get("xt", None)
        # Setup
        if prev_state is None:
            prev_state = torch.zeros((x.size(0), x.size(-2), self.out_size), dtype=x.dtype, device=x.device)
        h = prev_state
        if self.debug:
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
            print("prev_state =", h.shape)
        # Start
        if self.shared:
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, xt, x), -1)
            elif not xs is None and self.xs_size > 0:
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, x), -1)
            elif not xt is None and self.xt_size > 0:
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xt, x), -1)
            rz = torch.sigmoid(self.lin_rz(torch.cat((x, h), -1), edge_index, edge_weight, **kwargs))
            r, z = torch.split(rz, self.out_size, -1)
            n = torch.tanh(self.lin_n(torch.cat((x, r * h), -1), edge_index, edge_weight, **kwargs))
        else:
            s_w, t_w, st_w = self.get_fuse_w()
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                r = torch.sigmoid(
                    s_w * self.lin_sr(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_tr(xt)[:,None,:] + \
                    st_w * self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    s_w * self.lin_sz(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_tz(xt)[:,None,:] + \
                    st_w * self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    s_w * self.lin_sn(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_tn(xt)[:,None,:] + \
                    st_w * self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
            elif not xs is None and self.xs_size > 0:
                r = torch.sigmoid(
                    s_w * self.lin_sr(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    s_w * self.lin_sz(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    s_w * self.lin_sn(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
            elif not xt is None and self.xt_size > 0:
                r = torch.sigmoid(
                    t_w * self.lin_tr(xt)[:,None,:] + \
                    st_w * self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    t_w * self.lin_tz(xt)[:,None,:] + \
                    st_w * self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    t_w * self.lin_tn(xt)[:,None,:] + \
                    st_w * self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
            else:
                r = torch.sigmoid(
                    self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
        _h = (1 - z) * n + z * h
        if self.debug:
            print(_h.shape)
        return _h

    def get_fuse_w(self):
        fuse_w = self.fuse_w
        if self.fusion == "attn":
            fuse_w = torch.nn.functional.softmax(fuse_w, -1)
        if self.xs_size < 1 and self.xt_size < 1:
            fuse_w = (1, 1, fuse_w[0])
        elif self.xs_size < 1:
            fuse_w = (1, fuse_w[0], fuse_w[1])
        elif self.xt_size < 1:
            fuse_w = (fuse_w[0], 1, fuse_w[1])
        return fuse_w


class gcstLSTMCell(torch.nn.Module):

    def __init__(self, in_size, out_size, xs_size=0, xt_size=0, conv="std", layer="Linear", bias=True, order=1, n_hops=1, shared=True, fusion="add", **kwargs):
        super(gcstLSTMCell, self).__init__()
        print("gcstLSTMCell :", in_size, out_size, xs_size, xt_size, layer, bias, order, n_hops, shared, kwargs)
        # Layers
        n_modal = 1
        if shared:
            self.lin_ifo = gcLinear(
                in_size + xs_size + xt_size + out_size, 
                3 * out_size, 
                conv, layer, bias, order, n_hops, **kwargs
            )
            self.lin_g = gcLinear(
                in_size + xs_size + xt_size + out_size, 
                out_size, conv, layer, bias, order, n_hops, **kwargs
            )
        else:
            # layers for x
            self.lin_ii = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_if = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_ig = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_io = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for h
            self.lin_hi = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hf = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hg = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_ho = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for xs
            if xs_size > 0:
                self.lin_si = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                self.lin_sf = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                self.lin_sg = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                self.lin_so = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                n_modal += 1
            # layers for xt
            if xt_size > 0:
                self.lin_ti = torch.nn.Linear(xt_size, out_size, bias)
                self.lin_tf = torch.nn.Linear(xt_size, out_size, bias)
                self.lin_tg = torch.nn.Linear(xt_size, out_size, bias)
                self.lin_to = torch.nn.Linear(xt_size, out_size, bias)
                n_modal += 1
            if fusion == "add":
                self.fuse_w = torch.ones((n_modal,))
            elif fusion == "attn":
                self.fuse_w = torch.nn.Parameter(torch.rand((n_modal,)))
            else:
                raise NotImplementedError(fusion)
        # Save vars
        self.in_size = in_size
        self.out_size = out_size
        self.xs_size = xs_size
        self.xt_size = xt_size
        self.shared = shared
        self.fusion = fusion
        self.n_modal = n_modal
        self.order = order

    def forward(self, x, edge_index=None, edge_weight=None, prev_state=None, **kwargs):
        if self.shared:
            self.lin_ifo.debug = self.debug
        else:
            self.lin_ii.debug = self.debug
        xs, xt = kwargs.get("xs", None), kwargs.get("xt", None)
        # Setup
        if prev_state is None:
            zeros = torch.zeros((x.size(0), x.size(-2), self.out_size), dtype=x.dtype, device=x.device)
            prev_state = (zeros, zeros)
        h, c  = prev_state
        if self.debug:
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
            print("prev_state =", h.shape)
        # Start
        if self.shared:
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, xt, x), -1)
            elif not xs is None and self.xs_size > 0:
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, x), -1)
            elif not xt is None and self.xt_size > 0:
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xt, x), -1)
            ifo = torch.sigmoid(self.lin_ifo(torch.cat((x, h), -1), edge_index, edge_weight, **kwargs))
            i, f, o = torch.split(ifo, self.out_size, -1)
            g = torch.tanh(self.lin_g(torch.cat((x, h), -1), edge_index, edge_weight, **kwargs))
        else:
            s_w, t_w, st_w = self.get_fuse_w()
            # spatial and temporal given
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                i = torch.sigmoid(
                    s_w * self.lin_si(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_ti(xt)[:,None,:] + \
                    st_w * self.lin_ii(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hi(h, edge_index, edge_weight, **kwargs)
                )
                f = torch.sigmoid(
                    s_w * self.lin_sf(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_tf(xt)[:,None,:] + \
                    st_w * self.lin_if(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hf(h, edge_index, edge_weight, **kwargs)
                )
                g = torch.tanh(
                    s_w * self.lin_sg(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_tg(xt)[:,None,:] + \
                    st_w * self.lin_ig(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hg(h, edge_index, edge_weight, **kwargs)
                )
                o = torch.sigmoid(
                    s_w * self.lin_so(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    t_w * self.lin_to(xt)[:,None,:] + \
                    st_w * self.lin_io(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_ho(h, edge_index, edge_weight, **kwargs)
                )
            elif not xs is None and self.xs_size > 0: # only spatial given
                i = torch.sigmoid(
                    s_w * self.lin_si(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_ii(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hi(h, edge_index, edge_weight, **kwargs)
                )
                f = torch.sigmoid(
                    s_w * self.lin_sf(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_if(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hf(h, edge_index, edge_weight, **kwargs)
                )
                g = torch.tanh(
                    s_w * self.lin_sg(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_ig(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hg(h, edge_index, edge_weight, **kwargs)
                )
                o = torch.sigmoid(
                    s_w * self.lin_so(xs, edge_index, edge_weight, **kwargs)[None,:,:] + \
                    st_w * self.lin_io(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_ho(h, edge_index, edge_weight, **kwargs)
                )
            elif not xt is None and self.xt_size > 0: # only temporal given
                i = torch.sigmoid(
                    t_w * self.lin_ti(xt)[:,None,:] + \
                    st_w * self.lin_ii(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hi(h, edge_index, edge_weight, **kwargs)
                )
                f = torch.sigmoid(
                    t_w * self.lin_tf(xt)[:,None,:] + \
                    st_w * self.lin_if(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hf(h, edge_index, edge_weight, **kwargs)
                )
                g = torch.tanh(
                    t_w * self.lin_tg(xt)[:,None,:] + \
                    st_w * self.lin_ig(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hg(h, edge_index, edge_weight, **kwargs)
                )
                o = torch.sigmoid(
                    t_w * self.lin_to(xt)[:,None,:] + \
                    st_w * self.lin_io(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_ho(h, edge_index, edge_weight, **kwargs)
                )
            else: # only spatiotemporal given
                i = torch.sigmoid(
                    self.lin_ii(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hi(h, edge_index, edge_weight, **kwargs)
                )
                f = torch.sigmoid(
                    self.lin_if(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hf(h, edge_index, edge_weight, **kwargs)
                )
                g = torch.tanh(
                    self.lin_ig(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hg(h, edge_index, edge_weight, **kwargs)
                )
                o = torch.sigmoid(
                    self.lin_io(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_ho(h, edge_index, edge_weight, **kwargs)
                )
        _c = f * c + i * g
        _h = o * torch.tanh(_c)
        if self.debug:
            print(_h.shape, _c.shape)
        return (_h, _c)

    def get_fuse_w(self):
        fuse_w = self.fuse_w
        if self.fusion == "attn":
            fuse_w = torch.nn.functional.softmax(fuse_w, -1)
        if self.xs_size < 1 and self.xt_size < 1:
            fuse_w = (1, 1, fuse_w[0])
        elif self.xs_size < 1:
            fuse_w = (1, fuse_w[0], fuse_w[1])
        elif self.xt_size < 1:
            fuse_w = (fuse_w[0], 1, fuse_w[1])
        return fuse_w


class RNN(Model):

    def __init__(self, in_size, out_size, n_rnn_layers=1, rnn_layer="LSTM", rnn_kwargs={}, dropout=0.0):
        super(RNN, self).__init__()
        self.supported_layers = ["RNN", "GRU", "LSTM"]
        assert rnn_layer in self.supported_layers, "Layer \"%s\" not supported" % (rnn_layer)
        rnn_kwargs["num_layers"] = n_rnn_layers
        rnn_kwargs["batch_first"] = True
        rnn_kwargs["dropout"] = dropout
        self.rnn_forward = getattr(self, "%s_forward" % (rnn_layer))
        self.rnn = self.layer_fn_map[rnn_layer](in_size, out_size, **rnn_kwargs)
        # Save all vars
        self.in_size = in_size
        self.out_size = out_size
        self.n_layers = n_rnn_layers
        self.rnn_layer = rnn_layer
        self.rnn_kwargs = rnn_kwargs
        self.dropout = dropout

    def forward(self, **inputs):
        x, xs, x_pc = inputs["x"], inputs.get("xs", None), inputs.get("x_pc", None)
        assert len(x.shape) == 3
        init_state = inputs.get("init_state", None)
        if self.debug:
            print(util.make_msg_block("RNN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
        if self.debug and not x_pc is None:
            print("x_pc =", x_pc.shape)
            print(util.memory_of(x_pc))
        a, prev_state = self.rnn_forward(self.rnn, x=x, xs=xs, x_pc=x_pc, prev_state=init_state)
        return {"yhat": a, "final_state": prev_state}

    def RNN_forward(self, rnn, **inputs):
        hidden_state = rnn(inputs["x"], inputs["prev_state"])
        return hidden_state, hidden_state

    def GRU_forward(self, rnn, **inputs):
        hidden_state = rnn(inputs["x"], inputs["prev_state"])
        return hidden_state, hidden_state

    def LSTM_forward(self, rnn, **inputs):
        hidden_state, cell_state = rnn(inputs["x"], inputs["prev_state"])
        return hidden_state, (hidden_state, cell_state)

    def reset_parameters(self):
        self.rnn.reset_parameters()


class RNNCell(Model):

    debug = 0

    def __init__(self, in_size, out_size, n_rnn_layers=1, rnn_layer="LSTM", rnn_kwargs={}, dropout=0.0):
        super(RNNCell, self).__init__()
        print("RNNCell :", in_size, out_size, n_rnn_layers, rnn_layer, rnn_kwargs, dropout)
        self.supported_layers = ["RNN", "GRU", "LSTM", "pcRNN", "pcGRU", "pcLSTM", "mLSTM", "gGRU", "gLSTM", "gcGRU", "gcLSTM", "gcstRNN", "gcstGRU", "gcstLSTM", "AGCRN"]
        assert rnn_layer in self.supported_layers, "Layer \"%s\" not supported" % (rnn_layer)
        # Instantiate model layers
        def init_cell(in_size, out_size, rnn_layer, **rnn_kwargs):
            if rnn_layer in ["pcRNN", "pcGRU", "pcLSTM"]:
                cell = self.layer_fn_map["%sCell" % (rnn_layer[2:])](in_size+1, out_size, **rnn_kwargs)
            elif rnn_layer in ["mRNN", "mGRU", "mLSTM"]:
                cell = {
                    "mRNN": None, 
                    "mGRU": None, 
                    "mLSTM": mLSTMCell, 
                }[rnn_layer](in_size, out_size, **rnn_kwargs)
            elif rnn_layer in ["gRNN", "gGRU", "gLSTM"]:
                cell = {
                    "gRNN": None, 
                    "gGRU": gGRUCell, 
                    "gLSTM": gLSTMCell, 
                }[rnn_layer](in_size, out_size, **rnn_kwargs)
            elif rnn_layer in ["gcRNN", "gcGRU", "gcLSTM"]:
                cell = {
                    "gcRNN": None, 
                    "gcGRU": gcGRUCell, 
                    "gcLSTM": gcLSTMCell, 
                }[rnn_layer](in_size, out_size, **rnn_kwargs)
            elif rnn_layer in ["gcstRNN", "gcstGRU", "gcstLSTM"]:
                cell = {
                    "gcstRNN": gcstRNNCell, 
                    "gcstGRU": gcstGRUCell, 
                    "gcstLSTM": gcstLSTMCell, 
                }[rnn_layer](in_size, out_size, **rnn_kwargs)
            elif rnn_layer in ["AGCRN"]:
                return AGCRNCell(
                    rnn_kwargs["n_nodes"], 
                    in_size, 
                    out_size, 
                    rnn_kwargs.get("cheb_k", 2), 
                    rnn_kwargs.get("embed_size", 10)
                )
            else:
                cell = self.layer_fn_map["%sCell" % (rnn_layer)](in_size, out_size, **rnn_kwargs)
            return cell
        self.cell_forward = getattr(self, "%s_forward" % (rnn_layer))
        self.cells = [init_cell(in_size, out_size, rnn_layer, **rnn_kwargs)]
        self.cells += [init_cell(out_size, out_size, rnn_layer, **rnn_kwargs) for i in range(1, n_rnn_layers)]
        self.drops = [torch.nn.Dropout(dropout) for i in range(n_rnn_layers)]
        self.cells = torch.nn.ModuleList(self.cells)
        self.drops = torch.nn.ModuleList(self.drops)
        # Save all vars
        self.in_size = in_size
        self.out_size = out_size
        self.n_layers = n_rnn_layers
        self.rnn_layer = rnn_layer
        self.rnn_kwargs = rnn_kwargs
        self.dropout = dropout

    def forward(self, **inputs):
        for i in range(self.n_layers):
            self.cells[i].debug = self.debug
        x, xs, xt = inputs["x"], inputs.get("xs", None), inputs.get("xt", None)
        temporal_dim = inputs.get("temporal_dim", -2)
        init_state, n_steps = inputs.get("init_state", None), inputs.get("n_steps", x.shape[temporal_dim])
        if self.debug:
            print(util.make_msg_block("RNN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
            if self.debug > 1:
                print(x)
            if not xs is None:
                print("xs =", xs.shape)
                print(util.memory_of(xs))
            if not xt is None:
                print("xt =", xt.shape)
                print(util.memory_of(xt))
        autoregress = False
        if n_steps != x.shape[temporal_dim]: # encode autoregressively
            assert x.shape[temporal_dim] == 1, "Encoding a sequence from %d to %d time-steps is ambiguous" % (
                x.shape[temporal_dim], n_steps
            )
            autoregress = True
        outputs = {}
        get_idx_fn = self.index_select
#        get_idx_fn = self.transpose_select
        A = [None] * n_steps
        a = get_idx_fn(x, 0, temporal_dim)
        xt_t = None
        for i in range(self.n_layers):
            prev_state = init_state
            for t in range(n_steps):
                x_t = (a if autoregress else get_idx_fn(x, t, temporal_dim))
                if not xt is None:
                    xt_t = get_idx_fn(xt, t, temporal_dim)
                inputs_t = util.merge_dicts(inputs, {"x": x_t, "xt": xt_t, "prev_state": prev_state})
                a, prev_state = self.cell_forward(self.cells[i], **inputs_t)
                if self.debug:
                    print("Step-%d Embedding =" % (t+1), a.shape)
                    print(util.memory_of(a))
                A[t] = a
            a = torch.stack(A, temporal_dim)
            a = self.drops[i](a)
            x = a
        outputs["yhat"] = a
        outputs["final_state"] = prev_state
        return outputs

    def RNN_forward(self, cell, **inputs):
        hidden_state = cell(inputs["x"], inputs["prev_state"])
        return hidden_state, hidden_state

    def GRU_forward(self, cell, **inputs):
        hidden_state = cell(inputs["x"], inputs["prev_state"])
        return hidden_state, hidden_state

    def LSTM_forward(self, cell, **inputs):
        hidden_state, cell_state = cell(inputs["x"], inputs["prev_state"])
        return hidden_state, (hidden_state, cell_state)

    def pcLSTM_forward(self, cell, **inputs):
        x, x_pc = inputs["x"], inputs["x_pc"]
        if x_pc.shape[0] < x.shape[0]:
            x_pc = x_pc.repeat((x.shape[0] // x_pc.shape[0], 1))
        hidden_state, cell_state = cell(torch.cat((x, x_pc), -1), inputs["prev_state"])
        return hidden_state, (hidden_state, cell_state)

    def mLSTM_forward(self, cell, **inputs):
        hidden_state, cell_state = cell(inputs["x"], inputs.get("xs", None), inputs.get("prev_state", None))
        return hidden_state, (hidden_state, cell_state)

    def gGRU_forward(self, cell, **inputs):
        hidden_state = cell(
            inputs["x"], inputs.get("xs", None), inputs.get("group_index", None), inputs.get("prev_state", None)
        )
        return hidden_state, hidden_state

    def gLSTM_forward(self, cell, **inputs):
        hidden_state, cell_state = cell(
            inputs["x"], inputs.get("xs", None), inputs.get("group_index", None), inputs.get("prev_state", None)
        )
        return hidden_state, (hidden_state, cell_state)

    def AGCRN_forward(self, cell, **inputs):
        x, xs, prev_state = inputs["x"], inputs.get("xs", None), inputs.get("prev_state", None)
        if prev_state is None:
            prev_state = cell.init_hidden_state(x.shape[0])
        if self.debug:
            print("x =", x.shape)
            print("prev_state =", prev_state.shape)
            print("xs =", xs.shape)
        hidden_state = cell(x, prev_state, xs)
        return hidden_state, hidden_state

    def gcGRU_forward(self, cell, **inputs):
        x, prev_state = inputs["x"], inputs.get("prev_state", None)
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        if self.debug:
            print("x =", x.shape)
            if not prev_state is None:
                print("prev_state =", (prev_state[0].shape, prev_state[1].shape))
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
        _inputs = util.remap_dict(inputs, {"xs": "embedding"})
        hidden_state = cell(**_inputs)
        return hidden_state, hidden_state

    def gcLSTM_forward(self, cell, **inputs):
        x, prev_state = inputs["x"], inputs.get("prev_state", None)
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        if self.debug:
            print("x =", x.shape)
            if not prev_state is None:
                print("prev_state =", (prev_state[0].shape, prev_state[1].shape))
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
        _inputs = util.remap_dict(inputs, {"xs": "embedding"})
        (hidden_state, cell_state) = cell(**_inputs)
        return hidden_state, (hidden_state, cell_state)

    def gcstRNN_forward(self, cell, **inputs):
        x, prev_state = inputs["x"], inputs.get("prev_state", None)
        xs, xt = inputs.get("xs", None), inputs.get("xt", None)
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        if self.debug:
            print(util.make_msg_block("gcstRNNCell Forward"))
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not prev_state is None:
                print("prev_state =", (prev_state[0].shape, prev_state[1].shape))
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
        if not "embedding" in inputs:
            inputs["embedding"] = inputs.get("xs", None)
        hidden_state = cell(**inputs)
        return hidden_state, hidden_state

    def gcstGRU_forward(self, cell, **inputs):
        x, prev_state = inputs["x"], inputs.get("prev_state", None)
        xs, xt = inputs.get("xs", None), inputs.get("xt", None)
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        if self.debug:
            print(util.make_msg_block("gcstGRUCell Forward"))
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not prev_state is None:
                print("prev_state =", (prev_state[0].shape, prev_state[1].shape))
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
        if not "embedding" in inputs:
            inputs["embedding"] = inputs.get("xs", None)
        hidden_state = cell(**inputs)
        return hidden_state, hidden_state

    def gcstLSTM_forward(self, cell, **inputs):
        x, prev_state = inputs["x"], inputs.get("prev_state", None)
        xs, xt = inputs.get("xs", None), inputs.get("xt", None)
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        if self.debug:
            print(util.make_msg_block("gcstLSTMCell Forward"))
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not prev_state is None:
                print("prev_state =", (prev_state[0].shape, prev_state[1].shape))
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
        if not "embedding" in inputs:
            inputs["embedding"] = inputs.get("xs", None)
        (hidden_state, cell_state) = cell(**inputs)
        return hidden_state, (hidden_state, cell_state)

    def transpose_select(self, x, idx, dim):
        return torch.transpose(x, 0, dim)[idx,:,:]

    def index_select(self, x, idx, dim):
        return torch.index_select(x, dim, torch.tensor(idx, device=x.device)).squeeze(dim)

    def reset_parameters(self):
        for cell in self.cells:
            if hasattr(cell, "reset_parameters"):
                cell.reset_parameters()


class FlowConv(Model, MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(FlowConv, self).__init__()
        print("FlowConv :", in_channels, out_channels, kwargs)
        add_self_loops = kwargs.get("add_self_loops", False)
        normalize = kwargs.get("normalize", 0)
        node_layer = kwargs.get("node_layer", "Linear")
        node_act = kwargs.get("node_act", "Identity")
        edge_layer = kwargs.get("edge_layer", "Identity")
        edge_act = kwargs.get("edge_act", "Identity")
        nn_order = kwargs.get("nn_order", 1)
        n_hops = kwargs.get("n_hops", 1)
        bias = kwargs.get("bias", True)
        dropout = kwargs.get("dropout", 0.0)
        # Init layers
        #    Node NN
        self._forward = self.default_forward
        if node_layer in ["RNN", "GRU", "LSTM", "gRNN", "gGRU", "gLSTM", "mRNN", "mGRU", "mLSTM", "AGCRN"]:
            self.node_nn = RNNCell(
                in_channels, out_channels, kwargs.get("n_rnn_layers", 1), node_layer, kwargs.get("rnn_kwargs", {}), dropout
            )
            self.node_nn_forward = self.RNN_foward
        elif node_layer in ["gcRNN", "gcGRU", "gcLSTM", "gcstRNN", "gcstGRU", "gcstLSTM"]:
            self.node_nn = RNNCell(
                in_channels, out_channels, kwargs.get("n_rnn_layers", 1), node_layer, kwargs.get("rnn_kwargs", {}), dropout
            )
            self.node_nn_forward = self.gcRNN_forward
        elif node_layer == "gLinear":
            self.node_nn = gLinear(in_channels, out_channels, kwargs.get("n_groups", 3))
            self.node_nn_forward = gLinear_forward
        elif node_layer == "Linear":
            self.node_nn = torch.nn.Linear(in_channels, out_channels)
            self.node_nn_forward = Linear_forward
        elif edge_layer == "Identity":
            self.node_nn_forward = Identity_forward
        else:
            raise NotImplementedError(node_layer)
        if node_act == "Softmax":
            self.node_act = self.layer_fn_map[node_act](-1)
        else:
            self.node_act = self.layer_fn_map[node_act]()
        #    Edge NN
        self.edge_nn_is_rnn = 0
        if edge_layer in ["RNN", "GRU", "LSTM", "gRNN", "gGRU", "gLSTM", "mRNN", "mGRU", "mLSTM", "AGCRN"]:
            self.edge_nn = RNNCell(
                1, 1, kwargs.get("n_rnn_layers", 1), node_layer, kwargs.get("rnn_kwargs", {}), dropout
            )
            self.edge_nn_is_rnn = 1
            self.edge_nn_forward = self.RNN_forward
        elif edge_layer == "gLinear":
            self.edge_nn = gLinear(1, 1, kwargs.get("n_groups", 3))
            self.edge_nn_forward = self.gLinear_forward
        elif edge_layer == "Linear":
            self.edge_nn = torch.nn.Linear(1, 1)
            self.edge_nn_forward = self.Linear_forward
        elif edge_layer == "Identity":
            self.edge_nn = torch.nn.Identity()
            self.edge_nn_forward = self.Identity_forward
        else:
            raise NotImplementedError(edge_layer)
        if edge_act == "Softmax":
            self.edge_act = self.layer_fn_map[edge_act](-1)
        elif edge_act == "gcSoftmax":
            self.edge_act = gcSoftmax()
        else:
            self.edge_act = self.layer_fn_map[edge_act]()
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
#        self.reset_parameters()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.nn_order = nn_order
        self.n_hops = n_hops

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        """

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(?, 2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(?, |E|)

        """
        # Prepare shapes for broadcasting
        if edge_index is None:
            pass
        elif edge_weight is None:
            x, edge_index = pyt_util.align((x, edge_index), -1)
        else:
            x, edge_index, edge_weight = pyt_util.align((x, edge_index, torch.unsqueeze(edge_weight, -1)), -1)
            edge_weight = torch.squeeze(edge_weight, -1)
        return self._forward(x, edge_index, edge_weight, **kwargs)

    def default_forward(self, x, edge_index, edge_weight=None, **kwargs):
        # Node NN Forward: pre-propagation
        if self.nn_order & 1:
            x = self.node_nn_forward(self.node_nn, self.node_act, x, edge_index, edge_weight, **kwargs)
        # Propagate n_hops times to aggregate n_hop neighborhoods
        edge_index, edge_weight = self.edge_nn_forward(
            self.edge_nn, self.edge_act, x, edge_index, edge_weight, 2, **kwargs
        )
        edge_weight = self.normalize_edge_weight(x, edge_index, edge_weight)
        for i in range(self.n_hops):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        # Node NN Forward: post-propagation
        if self.nn_order & 2:
            x = self.node_nn_forward(self.node_nn, self.node_act, x, edge_index, edge_weight, **kwargs)
        if not self.bias is None:
            x += self.bias
        if self.debug:
            print("x =", x.shape)
            if self.debug > 1:
                print(x)
                input()
        return x

    def Identity_forward(self, nn, act, x, edge_index, edge_weight=None, src=1, **kwargs):
        if src == 1:
            return act(nn(x))
        elif src == 2:
            if isinstance(act, gcSoftmax):
                return act(edge_index, nn(edge_weight))
            return edge_index, edge_weight
        else:
            raise ValueError(src)

    def Linear_forward(self, nn, act, x, edge_index, edge_weight=None, src=1, **kwargs):
        if src == 1: # node forward
            return act(nn(x))
        elif src == 2: # edge forward
            if edge_weight is None: return edge_weight
            edge_weight = torch.unsqueeze(edge_weight, -1)
            edge_weight = act(nn(edge_weight, group_index=group_index))
            return torch.squeeze(edge_weight, -1)
        else:
            raise ValueError(src)

    def gLinear_forward(self, nn, act, x, edge_index, edge_weight=None, src=1, **kwargs):
        group_index = kwargs.get("group_index", None)
        if src == 1: # node forward
            return act(nn(x, group_index=group_index))
        elif src == 2: # edge forward
            if edge_weight is None: return edge_weight
            edge_weight = torch.unsqueeze(edge_weight, -1)
            edge_weight = act(nn(edge_weight, group_index=group_index))
            return torch.squeeze(edge_weight, -1)
        else:
            raise ValueError(src)

    def RNN_forward(self, nn, act, x, edge_index, edge_weight=None, src=1, **kwargs):
        xs, group_index = kwargs.get("xs", None), kwargs.get("group_index", None)
        if src == 1:
            N, T, V, F = x.shape
            if nn.rnn_layer in ["RNN", "GRU", "LSTM"]:
                x = torch.transpose(x, 1, 2)
                x = torch.reshape(x, (-1, T, F))
                x = nn(x=x, xs=xs, group_index=group_index)["yhat"]
                x = torch.reshape(x, (N, V, T, self.out_channels))
                x = torch.transpose(x, 1, 2)
                return act(x)
            else:
                return act(nn(x=x, xs=xs, group_index=group_index, temporal_dim=1)["yhat"])
        elif src == 2:
            if edge_weight is None: return edge_weight
            raise ValueError(src)
        else:
            raise ValueError(src)

    def gcRNN_forward(self, nn, act, x, edge_index, edge_weight=None, src=1, **kwargs):
        nn.debug = self.debug
        if self.debug:
            print(util.make_msg_block("gcRNN Forward"))
            print("x =", x.shape)
            if self.debug > 1:
                print(x)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
                if self.debug > 1:
                    print(edge_index)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
                if self.debug > 1:
                    print(edge_weight)
        if src == 1:
            V = x.shape[-2]
            if not edge_index is None:
                edge_index = pyt_util.squeeze_not(edge_index, (-2, -1))
            return act(nn(
                x=x, 
                edge_index=edge_index, 
                edge_weight=edge_weight, 
                temporal_dim=1, 
                **kwargs, 
            )["yhat"])
        elif src == 2:
            if edge_weight is None: return edge_weight
            raise ValueError(src)
        else:
            raise ValueError(src)

    def message(self, x_j, edge_weight):
        """
        Arguments
        ---------
        x_j : FloatTensor with shape=(?, |E|, F)
        edge_weight : (FloatTensor, optional) with shape=(?, |E|)

        Returns
        -------
        x : FloatTensor with shape=(?, |E|, F)

        """
        if edge_weight is None:
            return x_j
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        edge_weight = torch.unsqueeze(edge_weight, -1)
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        return edge_weight * x_j

    def normalize_edge_weight(self, x, edge_index, edge_weight):
        if not edge_weight is None and self.normalize:
            if self.debug:
                print(util.make_msg_block("Normalizing Edge Weight"))
            if self.debug:
                print("x =", x.shape, "=")
                if self.debug > 1:
                    print(x)
            if self.debug:
                print("edge_index =", edge_index.shape, "=")
                if self.debug > 1:
                    print(edge_index)
            if self.debug:
                print("edge_weight =", edge_weight.shape, "=")
                if self.debug > 1:
                    print(edge_weight)
            edge_weight = pyg_util.normalize_edge_weight(edge_index, edge_weight, self.normalize, self.debug)
            if self.debug:
                print("edge_weight_normalized =", edge_weight.shape, "=")
                if self.debug > 1:
                    print(edge_weight)
        return edge_weight

    def reset_parameters(self):
        if hasattr(self.node_nn, "reset_parameters"):
            self.node_nn.reset_parameters()
        if hasattr(self.edge_nn, "reset_parameters"):
            self.edge_nn.reset_parameters()
        if not self.bias is None:
            self.bias = torch.nn.init.zeros_(self.bias)


class GNN(Model):

    mp_kwarg_names = ["aggr", "aggr_kwargs", "flow", "node_dim", "decomposed_layers"]

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=16,
        n_hops=3,
        gcn_layer="GCNConv",
        gcn_kwargs={},
        act_layer="ReLU",
        act_kwargs={},
        dropout=0.0,
        use_edge_weights=True,
        use_edge_attributes=True,
    ):
        super(GNN, self).__init__()
        self.gcnlayer_fn_map["FlowConv"] = FlowConv
        # Setup
        if gcn_layer in self.gcnlayer_supported_map:
            self.use_edge_weights = use_edge_weights and "edge_weight" in self.gcnlayer_supported_map[gcn_layer]
            self.use_edge_attributes = use_edge_attributes and "edge_attr" in self.gcnlayer_supported_map[gcn_layer]
        else:
            self.use_edge_weights = False
            self.use_edge_attributes = False
        allow_default_init, allow_default_forward = False, False
        #   Initialization function for each gcn layer
        self.gcnlayer_init_map = {}
        for gcnlayer in self.gcnlayer_fn_map.keys():
            self.gcnlayer_init_map[gcnlayer] = self.default_init
            if hasattr(self, "%s_init" % (gcnlayer)):
                self.gcnlayer_init_map[gcnlayer] = getattr(self, "%s_init" % (gcnlayer))
            elif not allow_default_init:
                raise NotImplementedError("Unknown init() for layer \"%s\"" % (gcnlayer))
        #   Forward function for each gcn layer
        self.gcnlayer_forward_map = {}
        for gcnlayer in self.gcnlayer_fn_map.keys():
            self.gcnlayer_forward_map[gcnlayer] = self.default_forward
            if hasattr(self, "%s_forward" % (gcnlayer)):
                self.gcnlayer_forward_map[gcnlayer] = getattr(self, "%s_forward" % (gcnlayer))
            elif not allow_default_forward:
                raise NotImplementedError("Unknown forward() for layer \"%s\"" % (gcnlayer))
        # Broadcast arguments to number of layers
        if isinstance(gcn_kwargs, dict):
            gcn_kwargs = [gcn_kwargs for _ in range(n_hops)]
        elif not isinstance(gcn_kwargs, list):
            raise ValueError(
                "Input gcn_kwargs may be dict or list of dict with len=n_hops. Received %s" % (str(gcn_kwargs))
            )
        if isinstance(act_layer, str):
            act_layer = [act_layer for _ in range(n_hops)]
        elif not isinstance(act_layer, list):
            raise ValueError(
                "Input act_layer may be str or list of str with len=n_hops. Received %s" % (str(act_layer))
            )
        if isinstance(act_kwargs, dict):
            act_kwargs = [act_kwargs for _ in range(n_hops)]
        elif not isinstance(act_kwargs, list):
            raise ValueError(
                "Input act_kwargs may be dict or list of dict with len=n_hops. Received %s" % (str(act_kwargs))
            )
        if isinstance(dropout, float):
            dropout = [dropout for _ in range(n_hops)]
        elif not isinstance(dropout, list):
            raise ValueError(
                "Input dropout may be float or list of float with len=n_hops. Received %s" % (str(dropout))
            )
        # Instantiate model layers
        self.name_layer_map = {}
        self.gcnlayer_init_map[gcn_layer](
            in_size,
            out_size,
            hidden_size,
            n_hops,
            gcn_layer,
            gcn_kwargs,
            act_layer,
            act_kwargs,
            dropout,
        )
        self.gnn_forward = self.gcnlayer_forward_map[gcn_layer]
        # Save all vars
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.n_hops = n_hops
        self.gcn_layer = gcn_layer
        self.gcn_kwargs = gcn_kwargs
        self.act_layer = act_layer
        self.act_kwargs = act_kwargs
        self.dropout = dropout

    def default_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GCN layer, activation layer, and dropout layer repeated K times

        Arguments
        ---------
        in_size : int
            number of features F existing on each node for input X with shape=(?, |V|, F)
        out_size : int
            embedding dimension F' for neighborhood embedding X' with shape=(?, |V|, F')
        hidden_size : int
            embedding dimension of interior layers
        n_hops : int
            maximum number of edges (hops) away that nodes will be considered as part of a neighborhood
        gcn_layer : str
            name of the graph convolutional operator to be used for embedding neighborhoods
        gcn_kwargs : list of dict with shape=(n_hops,)
            key-word args for each GCN layer (see <gcn_layer>_init() for specifics)
        act_layer : list of str with shape=(n_hops,)
            name of activation layer to be applied to neighorhood embeddings
        act_kwargs : list of dict with shape=(n_hops,)
            key-word args for each activation layer
        dropout : list of float with shape=(n_hops,)
            dropout probability to be applied to post-activation neighborhood embeddings

        Returns
        -------
        nothing : instead populates dict name_layer_map which contains all layers of the model

        Notes
        -----

        """
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](_in_size, _out_size, **gcn_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def AGNNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of AGNNConv layers - The graph attentional propagation layer from the “Attention-based Graph Neural Network for Semi-Supervised Learning” paper

        Arguments
        ---------
        gcn_kwargs : list of dicts with shape=(n_hops,)
            requires_grad (bool, optional) - If set to False, beta will not be trainable. (default: True)
            add_self_loops (bool, optional) - If set to False, will not add self-loops to the input graph. (default: True)
            **kwargs (optional) - Additional arguments of torch_geometric.nn.conv.MessagePassing.
        else: see default_init()

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.AGNNConv
        source:
            https://arxiv.org/abs/1803.03735

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["requires_grad", "add_self_loops"] + self.mp_kwarg_names
            )
        _in_size, _out_size = in_size, out_size
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # GCN layer(s) init
        if in_size != hidden_size:
            _in_size = hidden_size
        if hidden_size != out_size:
            _out_size = hidden_size
        self.default_init(_in_size, _out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def APPNP_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of APPNP layers - The approximate personalized propagation of neural predictions layer from the “Predict then Propagate: Graph Neural Networks meet Personalized PageRank” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            K (int) – Number of iterations K.
            alpha (float) – Teleport probability a.
            dropout (float, optional) – Dropout probability of edges during training. (default: 0)
            cached (bool, optional) – If set to True, the layer will cache the computation of D^(-1/2)AD^(-1/2) on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            normalize (bool, optional) – Whether to add self-loops and apply symmetric normalization. (default: True)
            **kwargs (optional) - Additional arguments of torch_geometric.nn.conv.MessagePassing.
        else : see default_init()

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.APPNP
        source:
            https://arxiv.org/abs/1810.05997

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["K", "alpha", "dropout", "cached", "add_self_loops", "normalize"] + self.mp_kwarg_names
            )
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # GCN layer(s)
        _gcn_kwargs = util.merge_dicts(
            {
                "K": n_hops,
                "alpha": 0.1,
            },
            gcn_kwargs[-1],
        )
        name, layer = "gcn", self.gcnlayer_fn_map[gcn_layer](**_gcn_kwargs)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Activation layer
        name, layer = "act", self.layer_fn_map[act_layer[-1]](**(act_kwargs[-1]))
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Dropout layer
        name, layer = "drop", self.layer_fn_map["Dropout"](dropout[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def ARMAConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of ARMAConv layers - The ARMA graph convolutional operator from the “Graph Neural Networks with Convolutional ARMA Filters” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            num_stacks (int, optional) – Number of parallel stacks K. (default: 1).
            num_layers (int, optional) – Number of layers T. (default: 1)
            act (callable, optional) – Activation function sig. (default: torch.nn.ReLU())
            shared_weights (int, optional) – If set to True the layers in each stack will share the same parameters. (default: False)
            dropout (float, optional) – Dropout probability of the skip connection. (default: 0.)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
        else : see default_init()

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ARMAConv
        source:
            https://arxiv.org/abs/1901.01343

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["num_stacks", "num_layers", "act", "shared_weights", "dropout", "bias"] + self.mp_kwarg_names
            )
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # GCN layer(s)
        _gcn_kwargs = util.merge_dicts(
            {
                "num_layers": n_hops,
            },
            gcn_kwargs[-1],
        )
        name, layer = "gcn", self.gcnlayer_fn_map[gcn_layer](hidden_size, hidden_size, **_gcn_kwargs)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Activation layer
        name, layer = "act", self.layer_fn_map[act_layer[-1]](**act_kwargs[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Dropout layer
        name, layer = "drop", self.layer_fn_map["Dropout"](dropout[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def CGConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of CGConv layers - The crystal graph convolutional operator from the “Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            dim (int, optional) – Edge feature dimensionality. (default: 0)
            aggr (string, optional) – The aggregation operator to use ("add", "mean", "max"). (default: "add")
            batch_norm (bool, optional) – If set to True, will make use of batch normalization. (default: False)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
        else : see default_init()

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.CGConv
        source:
            https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["dim", "aggr", "batch_norm", "bias"] + self.mp_kwarg_names
            )
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        for i in range(n_hops):
            # GCN layer
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](hidden_size, **gcn_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def ChebConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of ChebConv layers - The chebyshev spectral graph convolutional operator from the “Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            K (int) – Chebyshev filter size K.
            normalization (str, optional) – The normalization scheme for the graph Laplacian (default: "sym"):
                1. None: No normalization
                2. "sym": Symmetric normalization
                3. "rw": Random-walk normalization
                lambda_max should be a torch.Tensor of size [num_graphs] in a mini-batch scenario and a scalar/zero-dimensional tensor when operating on single graphs. You can pre-compute lambda_max via the torch_geometric.transforms.LaplacianLambdaMax transform.
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
        else : see default_init()

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv
        source:
            https://arxiv.org/abs/1606.09375

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["K", "normalization", "bias"] + self.mp_kwarg_names
            )
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # GCN layer(s)
        _gcn_kwargs = util.merge_dicts(
            {
                "K": n_hops + 1, # K (int) – Chebyshev filter size K.
            },
            gcn_kwargs[-1],
        )
        name, layer = "gcn", self.gcnlayer_fn_map[gcn_layer](hidden_size, hidden_size, **_gcn_kwargs)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Activation layer
        name, layer = "act", self.layer_fn_map[act_layer[-1]](**act_kwargs[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Dropout layer
        name, layer = "drop", self.layer_fn_map["Dropout"](dropout[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def ClusterGCNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of ClusterGCNConv layers - The ClusterGCN graph convolutional operator from the “Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            diag_lambda (float, optional) – Diagonal enhancement value lambda. (default: 0.)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ClusterGCNConv
        source:
            https://arxiv.org/abs/1905.07953

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["diag_lambda", "add_self_loops", "bias"] + self.mp_kwarg_names
            )
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def DenseGCNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of DenseGCNConv layers - The GCNConv operator that accepts a dense adjacency matrix A

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            improved (bool, optional) – If set to True, the layer computes A as A + 2I. (default: False)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#dense-convolutional-layers
        source:
            https://arxiv.org/abs/1609.02907

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["improved", "bias"] + self.mp_kwarg_names
            )
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def DenseGINConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of DenseGINConv layers - The GINConv operator that accepts a dense adjacency matrix A

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            nn (torch.nn.Module) – A neural network  that maps node features x of shape [-1, in_channels] to shape [-1, out_channels], e.g., defined by torch.nn.Sequential.
            eps (float, optional) – (Initial) -value. (default: 0.)
            train_eps (bool, optional) – If set to True,  will be a trainable parameter. (default: False)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#dense-convolutional-layers
        source:
            https://arxiv.org/abs/1810.00826

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["nn", "eps", "train_eps"] + self.mp_kwarg_names
            )
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "nn": self.layer_fn_map["Sequential"](
                        self.layer_fn_map["Linear"](_in_size, _out_size),
                        self.layer_fn_map[act_layer[i]](**act_kwargs[i]),
                        self.layer_fn_map["Dropout"](dropout[i]),
                    ),
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](**_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def DenseGraphConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of DenseGraphConv layers - The GraphConv operator that accepts a dense adjacency matrix A

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "add")
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#dense-convolutional-layers
        source:
            https://arxiv.org/abs/1810.02244

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["aggr", "bias"] + self.mp_kwarg_names
            )
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def DenseSAGEConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of SAGEConv layers - The GraphSAGE operator that accepts a dense adjacency matrix A

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            normalize (bool, optional) – If set to True, output features will be -normalized, i.e., X_i/||X_i||2. (default: False)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
        source:
            https://arxiv.org/abs/1706.02216

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["normalize", "bias"] + self.mp_kwarg_names
            )
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def DNAConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of DNAConv layers - The dynamic neighborhood aggregation operator from the “Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            heads (int, optional) – Number of multi-head-attentions. (default: 1)
            groups (int, optional) – Number of groups to use for all linear projections. (default: 1)
            dropout (float, optional) – Dropout probability of attention coefficients. (default: 0.)
            cached (bool, optional) – If set to True, the layer will cache the computation of D^(-1/2)AD^(-1/2) on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
            normalize (bool, optional) – Whether to add self-loops and apply symmetric normalization. (default: True)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.DNAConv
        source:
            https://arxiv.org/abs/1904.04849

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["heads", "groups", "dropout", "cached", "normalize", "add_self_loops", "bias"] + self.mp_kwarg_names
            )
        self.default_init(in_size, hidden_size, hidden_size, n_hops-1, "GCNConv", gcn_kwargs[:-1], act_layer[:-1], act_kwargs[:-1], dropout[:-1])
        # Projection layer: maps input -> hidden
        if n_hops == 1 and in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # GCN layer
        name, layer = "dna", self.gcnlayer_fn_map[gcn_layer](hidden_size, **gcn_kwargs[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Activation layer
        name, layer = "act", self.layer_fn_map[act_layer[-1]](**act_kwargs[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Dropout layer
        name, layer = "drop", self.layer_fn_map["Dropout"](dropout[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def DynamicEdgeConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of DynamicEdgeConv layers - The dynamic edge convolutional operator from the “Dynamic Graph CNN for Learning on Point Clouds” paper (see torch_geometric.nn.conv.EdgeConv), where the graph is dynamically constructed using nearest neighbors in the feature space.

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            nn (torch.nn.Module) – A neural network  that maps pair-wise concatenated node features x of shape :obj:`[-1, 2 * in_channels] to shape [-1, out_channels], e.g. defined by torch.nn.Sequential.
            k (int) – Number of nearest neighbors.
            aggr (string) – The aggregation operator to use ("add", "mean", "max"). (default: "max")
            num_workers (int) – Number of workers to use for k-NN computation. Has no effect in case batch is not None, or the input lies on the GPU. (default: 1)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.DynamicEdgeConv
        source:
            https://arxiv.org/abs/1801.07829

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["nn", "k", "aggr", "num_workers"] + self.mp_kwarg_names
            )
        # Init layers
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "nn": self.layer_fn_map["Sequential"](
                        self.layer_fn_map["Linear"](2*_in_size, _out_size),
                        self.layer_fn_map[act_layer[i]](**act_kwargs[i]),
                        self.layer_fn_map["Dropout"](dropout[i]),
                    ),
                    "k": 3,
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](**_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def ECConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of ECConv layers - Alias of NNConv

        Arguments
        ---------
        see NNConv_init()

        Returns
        -------
        see default_init()

        """
        self.NNConv_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def EdgeConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of EdgeConv layers - The edge convolutional operator from the “Dynamic Graph CNN for Learning on Point Clouds” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            nn (torch.nn.Module) – A neural network  that maps pair-wise concatenated node features x of shape [-1, 2 * in_channels] to shape [-1, out_channels], e.g., defined by torch.nn.Sequential.
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "max")

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv
        source:
            https://arxiv.org/abs/1801.07829

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["nn", "aggr"] + self.mp_kwarg_names
            )
        # Init layers
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "nn": self.layer_fn_map["Sequential"](
                        self.layer_fn_map["Linear"](2*_in_size, _out_size),
                        self.layer_fn_map[act_layer[i]](**act_kwargs[i]),
                        self.layer_fn_map["Dropout"](dropout[i]),
                    ),
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](**_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def EGConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of EGConv layers - The Efficient Graph Convolution from the “Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions” paper.

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            aggregators (List[str], optional) – Aggregators to be used. Supported aggregators are "sum", "mean", "symnorm", "max", "min", "std", "var". Multiple aggregators can be used to improve the performance. (default: ["symnorm"])
            num_heads (int, optional) – Number of heads  to use. Must have out_channels % num_heads == 0. It is recommended to set num_heads >= num_bases. (default: 8)
            num_bases (int, optional) – Number of basis weights  to use. (default: 4)
            cached (bool, optional) – If set to True, the layer will cache the computation of the edge index with added self loops on first execution, along with caching the calculation of the symmetric normalized edge weights if the "symnorm" aggregator is being used. This parameter should only be set to True in transductive learning scenarios. (default: False)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EGConv
        source:
            https://arxiv.org/abs/2104.01481

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["aggregators", "num_heads", "num_bases", "cached", "add_self_loops", "bias"] + self.mp_kwarg_names
            )
        # Init layers
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def FAConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of FAConv layers - The Frequency Adaptive Graph Convolution operator from the “Beyond Low-Frequency Information in Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            eps (float, optional) – -value. (default: 0.1)
            dropout (float, optional) – Dropout probability of the normalized coefficients which exposes each node to a stochastically sampled neighborhood during training. (default: 0).
            cached (bool, optional) – If set to True, the layer will cache the computation of sqrt(d_i*d_j) on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            normalize (bool, optional) – Whether to add self-loops (if add_self_loops is True) and compute symmetric normalization coefficients on the fly. If set to False, edge_weight needs to be provided in the layer’s forward() method. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FAConv
        source:
            https://arxiv.org/abs/2101.00797

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["eps", "dropout", "cached", "add_self_loops", "normalize"] + self.mp_kwarg_names
            )
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # Init layers
        for i in range(n_hops):
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "normalize": not self.use_edge_weights,
                    # If edge_weight not given in forward(), normalization=True will create it
                    # If normlize == True => edge_weight == None
                    # If normalize == False => edge_weight != None
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](hidden_size, **_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def FastRGCNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of FastRGCNConv layers - Runtime efficient RGCNConv

        Arguments
        ---------
        see RGCNConv_init()

        Returns
        -------
        see default_init()

        """
        self.RGCNConv_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def FeaStConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of FeaStConv layers - The (translation-invariant) feature-steered convolutional operator from the “FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            heads (int, optional) – Number of attention heads H. (default: 1)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FeaStConv
        source:
            https://arxiv.org/abs/1706.05206

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["heads", "add_self_loops", "bias"] + self.mp_kwarg_names
            )
        # Init layers
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def FlowConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of FlowConv layers - 

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            normalize (bool, optional) – Whether to normalize edge weights. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
        source:

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            if 0:
                gcn_kwargs[i] = util.filter_dict(
                    gcn_kwargs[i], 
                    ["add_self_loops", "normalize", "bias"] + self.mp_kwarg_names
                )
        # Init layers
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def FiLMConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of FiLMConv layers - The FiLM graph convolutional operator from the “GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            num_relations (int, optional) – Number of relations. (default: 1)
            nn (torch.nn.Module, optional) – The neural network  that maps node features x_i of shape [-1, in_channels] to shape [-1, 2 * out_channels]. If set to None,  will be implemented as a single linear layer. (default: None)
            act (callable, optional) – Activation function . (default: torch.nn.ReLU())
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "mean")

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FiLMConv
        source:
            https://arxiv.org/abs/1906.12192

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["num_relations", "nn", "act", "aggr"] + self.mp_kwarg_names
            )
        # Init layers
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](_in_size, _out_size, **gcn_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def GATConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GATConv layers - The graph attentional operator from the “Graph Attention Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            heads (int, optional) – Number of multi-head-attentions. (default: 1)
            concat (bool, optional) – If set to False, the multi-head attentions are averaged instead of concatenated. (default: True)
            negative_slope (float, optional) – LeakyReLU angle of the negative slope. (default: 0.2)
            dropout (float, optional) – Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training. (default: 0)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            edge_dim (int, optional) – Edge feature dimensionality (in case there are any). (default: None)
            fill_value (float or Tensor or str, optional) – The way to generate edge features of self-loops (in case edge_dim != None). If given as float or torch.Tensor, edge features of self-loops will be directly given by fill_value. If given as str, edge features of self-loops are computed by aggregating all features of edges that point to the specific node, according to a reduce operation. ("add", "mean", "min", "max", "mul"). (default: "mean")
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
        source:
            https://arxiv.org/abs/1710.10903

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                [
                    "heads", 
                    "concat", 
                    "negative_slope", 
                    "dropout", 
                    "add_self_loops", 
                    "edge_dim", 
                    "fill_value", 
                    "bias"
                ] + self.mp_kwarg_names
            )
        # Init layers
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def GatedGraphConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GatedGraphConv layers - The gated graph convolution operator from the “Gated Graph Sequence Neural Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            num_layers (int) – The sequence length L.
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "add")
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GatedGraphConv
        source:
            https://arxiv.org/abs/1511.05493

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["num_layers", "aggr", "bias"] + self.mp_kwarg_names
            )
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # GCN layer
        _gcn_kwargs = util.merge_dicts(
            {
                "num_layers": n_hops,
            },
            gcn_kwargs[-1]
        )
        name, layer = "gcn", self.gcnlayer_fn_map[gcn_layer](hidden_size, **_gcn_kwargs)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Activation layer
        name, layer = "act", self.layer_fn_map[act_layer[-1]](**act_kwargs[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Dropout layer
        name, layer = "drop", self.layer_fn_map["Dropout"](dropout[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def GATv2Conv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GATv2Conv layers - The GATv2 operator from the “How Attentive are Graph Attention Networks?” paper, which fixes the static attention problem of the standard GATConv layer. Since the linear layers in the standard GAT are applied right after each other, the ranking of attended nodes is unconditioned on the query node. In contrast, in GATv2, every node can attend to any other node.

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            heads (int, optional) – Number of multi-head-attentions. (default: 1)
            concat (bool, optional) – If set to False, the multi-head attentions are averaged instead of concatenated. (default: True)
            negative_slope (float, optional) – LeakyReLU angle of the negative slope. (default: 0.2)
            dropout (float, optional) – Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training. (default: 0)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            edge_dim (int, optional) – Edge feature dimensionality (in case there are any). (default: None)
            fill_value (float or Tensor or str, optional) – The way to generate edge features of self-loops (in case edge_dim != None). If given as float or torch.Tensor, edge features of self-loops will be directly given by fill_value. If given as str, edge features of self-loops are computed by aggregating all features of edges that point to the specific node, according to a reduce operation. ("add", "mean", "min", "max", "mul"). (default: "mean")
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
            share_weights (bool, optional) – If set to True, the same matrix will be applied to the source and the target node of every edge. (default: False)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv
        source:
            https://arxiv.org/abs/2105.14491

        """
        self.GATConv_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def GCN2Conv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GCN2Conv layers - The graph convolutional operator with initial residual connections and identity mapping (GCNII) from the “Simple and Deep Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            alpha (float) – The strength of the initial residual connection alpha.
            theta (float, optional) – The hyperparameter  to compute the strength of the identity mapping beta=log(theta/l + 1). (default: None)
            layer (int, optional) – The layer l in which this module is executed. (default: None)
            shared_weights (bool, optional) – If set to False, will use different weight matrices for the smoothed representation and the initial residual (“GCNII*”). (default: True)
            cached (bool, optional) – If set to True, the layer will cache the computation of D^(-1/2)AD^(-1/2) on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
            normalize (bool, optional) – Whether to add self-loops and apply symmetric normalization. (default: True)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCN2Conv
        source:
            https://arxiv.org/abs/2007.02133

        """
        # Filter to keep only accepted key-word args
        for i in range(len(gcn_kwargs)):
            gcn_kwargs[i] = util.filter_dict(
                gcn_kwargs[i], 
                ["num_layers", "aggr", "bias"] + self.mp_kwarg_names
            )
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # Init layers
        for i in range(n_hops):
            # GCN layer
            #   settings acquired from section "6.3 Inductive Learning"
            _gcn_kwargs = util.merge_dicts(
                {
                    "alpha": 0.5,
                    "theta": 1.0,
                    "layer": i+1,
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](hidden_size, **_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def GCNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GCNConv layers - The graph convolutional operator from the “Semi-supervised Classification with Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            improved (bool, optional) – If set to True, the layer computes A as A + 2I. (default: False)
            cached (bool, optional) – If set to True, the layer will cache the computation of D^(-1/2)AD^(-1/2) on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            normalize (bool, optional) – Whether to add self-loops and compute symmetric normalization coefficients on the fly. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        source:
            https://arxiv.org/abs/1609.02907

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def GENConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GENConv layers - The GENeralized Graph Convolution (GENConv) from the “DeeperGCN: All You Need to Train Deeper GCNs” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            aggr (str, optional) – The aggregation scheme to use ("softmax", "powermean", "add", "mean", max). (default: "softmax")
            t (float, optional) – Initial inverse temperature for softmax aggregation. (default: 1.0)
            learn_t (bool, optional) – If set to True, will learn the value t for softmax aggregation dynamically. (default: False)
            p (float, optional) – Initial power for power mean aggregation. (default: 1.0)
            learn_p (bool, optional) – If set to True, will learn the value p for power mean aggregation dynamically. (default: False)
            msg_norm (bool, optional) – If set to True, will use message normalization. (default: False)
            learn_msg_scale (bool, optional) – If set to True, will learn the scaling factor of message normalization. (default: False)
            norm (str, optional) – Norm layer of MLP layers ("batch", "layer", "instance") (default: batch)
            num_layers (int, optional) – The number of MLP layers. (default: 2)
            eps (float, optional) – The epsilon value of the message construction function. (default: 1e-7)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GENConv
        source:
            https://arxiv.org/abs/2006.07739

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def GeneralConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GeneralConv layers - A general GNN layer adapted from the “Design Space for Graph Neural Networks” paper.

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            in_edge_channels (int, optional) – Size of each input edge. (default: None)
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "mean")
            skip_linear (bool, optional) – Whether apply linear function in skip connection. (default: False)
            directed_msg (bool, optional) – If message passing is directed; otherwise, message passing is bi-directed. (default: True)
            heads (int, optional) – Number of message passing ensembles. If heads > 1, the GNN layer will output an ensemble of multiple messages. If attention is used (attention=True), this corresponds to multi-head attention. (default: 1)
            attention (bool, optional) – Whether to add attention to message computation. (default: False)
            attention_type (str, optional) – Type of attention: "additive", "dot_product". (default: "additive")
            l2_normalize (bool, optional) – If set to True, output features will be l2-normalized, i.e., X_i/||X_i||2. (default: False)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GeneralConv
        source:
            https://arxiv.org/abs/2011.08843

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def GINConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GINConv layers - The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            nn (torch.nn.Module) – A neural network  that maps node features x of shape [-1, in_channels] to shape [-1, out_channels], e.g., defined by torch.nn.Sequential.
            eps (float, optional) – (Initial) -value. (default: 0.)
            train_eps (bool, optional) – If set to True,  will be a trainable parameter. (default: False)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv
        source:
            https://arxiv.org/abs/1810.00826

        """
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "nn": self.layer_fn_map["Sequential"](
                        self.layer_fn_map["Linear"](_in_size, _out_size),
                        self.layer_fn_map[act_layer[i]](**act_kwargs[i]),
                        self.layer_fn_map["Dropout"](dropout[i]),
                    ),
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](**_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def GINEConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GINEConv layers - The modified GINConv operator from the “Strategies for Pre-training Graph Neural Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            nn (torch.nn.Module) – A neural network  that maps node features x of shape [-1, in_channels] to shape [-1, out_channels], e.g., defined by torch.nn.Sequential.
            eps (float, optional) – (Initial) -value. (default: 0.)
            train_eps (bool, optional) – If set to True,  will be a trainable parameter. (default: False)
            edge_dim (int, optional) – Edge feature dimensionality. If set to None, node and edge feature dimensionality is expected to match. Other-wise, edge features are linearly transformed to match node feature dimensionality. (default: None)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINEConv
        source:
            https://arxiv.org/abs/1905.12265

        """
        self.GINConv_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def GMMConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GMMConv layers - The gaussian mixture model convolutional operator from the “Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            dim (int) – Pseudo-coordinate dimensionality.
            kernel_size (int) – Number of kernels .
            separate_gaussians (bool, optional) – If set to True, will learn separate GMMs for every pair of input and output channel, inspired by traditional CNNs. (default: False)
            aggr (string, optional) – The aggregation operator to use ("add", "mean", "max"). (default: "mean")
            root_weight (bool, optional) – If set to False, the layer will not add transformed root node features to the output. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GMMConv
        source:
            https://arxiv.org/abs/1611.08402

        """
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            if i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "dim": 3,
                    "kernel_size": 3,
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](_in_size, _out_size, **_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def GraphConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GraphConv layers - The graph neural network operator from the “Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "add")
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv
        source:
            https://arxiv.org/abs/1810.02244

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def GravNetConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of GravNetConv layers - The GravNet operator from the “Learning Representations of Irregular Particle-detector Geometry with Distance-weighted Graph Networks” paper, where the graph is dynamically constructed using nearest neighbors.

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            space_dimensions (int) – The dimensionality of the space used to construct the neighbors; referred to as S in the paper.
            propagate_dimensions (int) – The number of features to be propagated between the vertices; referred to as F_LR in the paper.
            k (int) – The number of nearest neighbors.

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GravNetConv
        source:
            https://arxiv.org/abs/1902.07987

        """
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            if i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "space_dimensions": hidden_size,
                    "propagate_dimensions": hidden_size,
                    "k": 3,
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](_in_size, _out_size, **_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def HGTConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of HGTConv layers - The Heterogeneous Graph Transformer (HGT) operator from the “Heterogeneous Graph Transformer” paper.

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            metadata (Tuple[List[str], List[Tuple[str, str, str]]]) – The metadata of the heterogeneous graph, i.e. its node and edge types given by a list of strings and a list of string triplets, respectively. See torch_geometric.data.HeteroData.metadata() for more information.
            heads (int, optional) – Number of multi-head-attentions. (default: 1)
            group (string, optional) – The aggregation scheme to use for grouping node embeddings generated by different relations. ("sum", "mean", "min", "max"). (default: "sum")

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HGTConv
        source:
            https://arxiv.org/abs/2003.01332

        """
        fn_name = "%s.%s" % (self.__class__.__name__, inspect.currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (fn_name))

    def HypergraphConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of HypergraphConv layers - The hypergraph convolutional operator from the “Hypergraph Convolution and Hypergraph Attention” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            use_attention (bool, optional) – If set to True, attention will be added to this layer. (default: False)
            heads (int, optional) – Number of multi-head-attentions. (default: 1)
            concat (bool, optional) – If set to False, the multi-head attentions are averaged instead of concatenated. (default: True)
            negative_slope (float, optional) – LeakyReLU angle of the negative slope. (default: 0.2)
            dropout (float, optional) – Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training. (default: 0)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HypergraphConv
        source:
            https://arxiv.org/abs/1901.08150

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def LEConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of LEConv layers - The local extremum graph neural network operator from the “ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True).

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.LEConv
        source:
            https://arxiv.org/abs/1911.07979

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def MFConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of MFConv layers - The graph neural network operator from the “Convolutional Networks on Graphs for Learning Molecular Fingerprints” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            max_degree (int, optional) – The maximum node degree to consider when updating weights (default: 10)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.MFConv
        source:
            https://arxiv.org/abs/1509.09292

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def NNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of NNConv layers - The continuous kernel-based convolutional operator from the “Neural Message Passing for Quantum Chemistry” paper. This convolution is also known as the edge-conditioned convolution from the “Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            edge_size (int) - The number of features D that define each edge in edge_attr with shape=(|M|, D)
            nn (torch.nn.Module) – A neural network  that maps edge features edge_attr of shape [-1, num_edge_features] to shape [-1, in_channels * out_channels], e.g., defined by torch.nn.Sequential.
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "add")
            root_weight (bool, optional) – If set to False, the layer will not add the transformed root node features to the output. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.NNConv
        source:
            https://arxiv.org/abs/1704.01212

        """
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            edge_size = 1
            if "edge_size" in gcn_kwargs[i]:
                edge_size = gcn_kwargs[i].pop("edge_size")
            _gcn_kwargs = util.merge_dicts(
                {
                    "nn": self.layer_fn_map["Sequential"](
                        self.layer_fn_map["Linear"](edge_size, _in_size * _out_size),
                        self.layer_fn_map[act_layer[i]](**act_kwargs[i]),
                    ),
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](_in_size, _out_size, **_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        self.edge_size = edge_size

    def PANConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of PANConv layers - The path integral based convolutional operator from the “Path Integral Based Convolution and Pooling for Graph Neural Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            filter_size (int) – The filter size L.

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PANConv
        source:
            https://arxiv.org/abs/2006.16811

        """
        # Projection layer: maps input -> hidden
        if in_size != hidden_size:
            name, layer = "in_proj", self.layer_fn_map["Linear"](in_size, hidden_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        for i in range(n_hops):
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "filter_size": n_hops,
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](hidden_size, hidden_size, **_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
        # Projection layer: maps hidden -> output
        if hidden_size != out_size:
            name, layer = "out_proj", self.layer_fn_map["Linear"](hidden_size, out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def PDNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of PDNConv layers - The pathfinder discovery network convolutional operator from the “Pathfinder Discovery Networks for Neural Message Passing” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            edge_dim (int) – Edge feature dimensionality.
            hidden_channels (int) – Hidden edge feature dimensionality.
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            normalize (bool, optional) – Whether to add self-loops and compute symmetric normalization coefficients on the fly. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PDNConv
        source:
            https://arxiv.org/pdf/2010.12878.pdf

        """
        edge_size = in_size
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            if i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "edge_dim": edge_size,
                    "hidden_channels": hidden_size,
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](_in_size, _out_size, **_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_fn_map[act_layer[i]](**act_kwargs[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_fn_map["Dropout"](dropout[i])
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def PNAConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of PNAConv layers - The Principal Neighbourhood Aggregation graph convolution operator from the “Principal Neighbourhood Aggregation for Graph Nets” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            aggregators (list of str) – Set of aggregation function identifiers, namely "sum", "mean", "min", "max", "var" and "std".
            scalers (list of str) – Set of scaling function identifiers, namely "identity", "amplification", "attenuation", "linear" and "inverse_linear".
            deg (Tensor) – Histogram of in-degrees of nodes in the training set, used by scalers to normalize.
            edge_dim (int, optional) – Edge feature dimensionality (in case there are any). (default None)
            towers (int, optional) – Number of towers (default: 1).
            pre_layers (int, optional) – Number of transformation layers before aggregation (default: 1).
            post_layers (int, optional) – Number of transformation layers after aggregation (default: 1).
            divide_input (bool, optional) – Whether the input features should be split between towers or not (default: False).
            act (str or Callable, optional) – Pre- and post-layer activation function to use. (default: "relu")
            act_kwargs (Dict[str, Any], optional) – Arguments passed to the respective activation function defined by act. (default: None)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PNAConv
        source:
            https://arxiv.org/abs/2004.05718

        """
        fn_name = "%s.%s" % (self.__class__.__name__, inspect.currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (fn_name))

    def PointConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of PointConv layers - Alias of PointNetConv

        Arguments
        ---------
        see PointNetConv_init()

        Returns
        -------
        see default_init()

        """
        self.PointNetConv_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def PointNetConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of PointNetConv layers - The PointNet set layer from the “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation” and “PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space” papers

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            local_nn (torch.nn.Module, optional) – A neural network  that maps node features x and relative spatial coordinates pos_j - pos_i of shape [-1, in_channels + num_dimensions] to shape [-1, out_channels], e.g., defined by torch.nn.Sequential. (default: None)
            global_nn (torch.nn.Module, optional) – A neural network  that maps aggregated node features of shape [-1, out_channels] to shape [-1, final_out_channels], e.g., defined by torch.nn.Sequential. (default: None)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PointNetConv
        source:
            https://arxiv.org/abs/1612.00593

        """
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            if i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            _gcn_kwargs = util.merge_dicts(
                {
                    "local_nn": self.layer_fn_map["Sequential"](
                        self.layer_fn_map["Linear"](_in_size+3, _out_size),
                        self.layer_fn_map[act_layer[i]](**act_kwargs[i]),
                        self.layer_fn_map["Dropout"](dropout[i]),
                    ),
                    "global_nn": self.layer_fn_map["Sequential"](
                        self.layer_fn_map["Linear"](_out_size, _out_size),
                        self.layer_fn_map[act_layer[i]](**act_kwargs[i]),
                        self.layer_fn_map["Dropout"](dropout[i]),
                    ),
                },
                gcn_kwargs[i],
            )
            name, layer = "gcn_%d" % (i), self.gcnlayer_fn_map[gcn_layer](**_gcn_kwargs)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def PPFConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of PPFConv layers - The PPFNet operator from the “PPFNet: Global Context Aware Local Features for Robust 3D Point Matching” paper

        Arguments
        ---------
        see PointNetConv_init()

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.PPFConv
        source:
            https://arxiv.org/abs/1802.02669

        """
        self.PointNetConv_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def ResGatedGraphConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of RGCNConv layers - The relational graph convolutional operator from the “Modeling Relational Data with Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            act (callable, optional) – Gating function . (default: torch.nn.Sigmoid())
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
            root_weight (bool, optional) – If set to False, the layer will not add transformed root node features to the output. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ResGatedGraphConv
        source:
            https://arxiv.org/abs/1711.07553

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def RGCNConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of RGCNConv layers - The relational graph convolutional operator from the “Modeling Relational Data with Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            num_relations (int) – Number of relations.
            num_bases (int, optional) – If set, this layer will use the basis-decomposition regularization scheme where num_bases denotes the number of bases to use. (default: None)
            num_blocks (int, optional) – If set, this layer will use the block-diagonal-decomposition regularization scheme where num_blocks denotes the number of blocks to use. (default: None)
            aggr (string, optional) – The aggregation scheme to use ("add", "mean", "max"). (default: "mean")
            root_weight (bool, optional) – If set to False, the layer will not add transformed root node features to the output. (default: True)
            is_sorted (bool, optional) – If set to True, assumes that edge_index is sorted by edge_type. This avoids internal re-sorting of the data and can improve runtime and memory efficiency. (default: False)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv
        source:
            https://arxiv.org/abs/1703.06103

        """
        _gcn_kwargs = [
            util.merge_dicts(
                {
                    "num_relations": 1,
                },
                gcn_kwargs[i],
            ) for i in range(n_hops)
        ]
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, _gcn_kwargs, act_layer, act_kwargs, dropout)

    def SAGEConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of SAGEConv layers - The GraphSAGE operator from the “Inductive Representation Learning on Large Graphs” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            aggr (string or Aggregation, optional) – The aggregation scheme to use. Any aggregation of torch_geometric.nn.aggr can be used, e.g., "mean", "max", or "lstm". (default: "mean")
            normalize (bool, optional) – If set to True, output features will be -normalized, i.e., X_i/||X_i||2. (default: False)
            root_weight (bool, optional) – If set to False, the layer will not add transformed root node features to the output. (default: True)
            project (bool, optional) – If set to True, the layer will apply a linear transformation followed by an activation function before aggregation (as described in Eq. (3) of the paper). (default: False)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
        source:
            https://arxiv.org/abs/1706.02216

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def SignedConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of SignedConv layers - The signed graph convolutional operator from the “Signed Graph Convolutional Network” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            first_aggr (bool) – Denotes which aggregation formula to use.
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SignedConv
        source:
            https://arxiv.org/abs/1808.06354

        """
        _gcn_kwargs = [
            util.merge_dicts(
                {
                    "first_aggr": True,
                },
                gcn_kwargs[i],
            ) for i in range(n_hops)
        ]
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, _gcn_kwargs, act_layer, act_kwargs, dropout)

    def SplineConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of SplineConv layers - The spline-based convolutional operator from the “SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            dim (int) – Pseudo-coordinate dimensionality.
            kernel_size (int or [int]) – Size of the convolving kernel.
            is_open_spline (bool or [bool], optional) – If set to False, the operator will use a closed B-spline basis in this dimension. (default True)
            degree (int, optional) – B-spline basis degrees. (default: 1)
            aggr (string, optional) – The aggregation operator to use ("add", "mean", "max"). (default: "mean")
            root_weight (bool, optional) – If set to False, the layer will not add transformed root node features to the output. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SplineConv
        source:
            https://arxiv.org/abs/1711.08920

        """
        _gcn_kwargs = [
            util.merge_dicts(
                {
                    "dim": hidden_size,
                    "kernel_size": 3,
                },
                gcn_kwargs[i],
            ) for i in range(n_hops)
        ]
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, _gcn_kwargs, act_layer, act_kwargs, dropout)

    def SuperGATConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of SuperGATConv layers - The self-supervised graph attentional operator from the “How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            heads (int, optional) – Number of multi-head-attentions. (default: 1)
            concat (bool, optional) – If set to False, the multi-head attentions are averaged instead of concatenated. (default: True)
            negative_slope (float, optional) – LeakyReLU angle of the negative slope. (default: 0.2)
            dropout (float, optional) – Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training. (default: 0)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
            attention_type (string, optional) – Type of attention to use. ('MX', 'SD'). (default: 'MX')
            neg_sample_ratio (float, optional) – The ratio of the number of sampled negative edges to the number of positive edges. (default: 0.5)
            edge_sample_ratio (float, optional) – The ratio of samples to use for training among the number of training edges. (default: 1.0)
            is_undirected (bool, optional) – Whether the input graph is undirected. If not given, will be automatically computed with the input graph when negative sampling is performed. (default: False)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SuperGATConv
        source:
            https://openreview.net/forum?id=Wi5KUNlqWty

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def TransformerConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of TransformerConv layers - The graph transformer operator from the “Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            heads (int, optional) – Number of multi-head-attentions. (default: 1)
            concat (bool, optional) – If set to False, the multi-head attentions are averaged instead of concatenated. (default: True)
            beta (bool, optional) – If set, will combine aggregation and skip information (default: False)
            dropout (float, optional) – Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training. (default: 0)
            edge_dim (int, optional) – Edge feature dimensionality (in case there are any). Edge features are added to the keys after linear transformation, that is, prior to computing the attention dot product. They are also added to final values after the same linear transformation. (default None)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
            root_weight (bool, optional) – If set to False, the layer will not add the transformed root node features to the output and the option beta is set to False. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
        source:
            https://arxiv.org/abs/2009.03509

        """
        self.default_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def TAGConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of TAGConv layers - The topology adaptive graph convolutional networks operator from the “Topology Adaptive Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            K (int, optional) – Number of hops K. (default: 3)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
            normalize (bool, optional) – Whether to apply symmetric normalization. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TAGConv
        source:
            https://arxiv.org/abs/1710.10370

        """
        _gcn_kwargs = util.merge_dicts(
            {
                "K": n_hops,
            },
            gcn_kwargs[-1],
        )
        name, layer = "gcn", self.gcnlayer_fn_map[gcn_layer](in_size, out_size, **_gcn_kwargs)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Activation layer
        name, layer = "act", self.layer_fn_map[act_layer[-1]](**act_kwargs[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Dropout layer
        name, layer = "drop", self.layer_fn_map["Dropout"](dropout[-1])
        setattr(self, name, layer)
        self.name_layer_map[name] = layer

    def SGConv_init(self, in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout):
        """ Initialization for GNN composed of SGConv layers - The simple graph convolutional operator from the “Simplifying Graph Convolutional Networks” paper

        Arguments
        ---------
        gcn_kwargs : dict to supply key-word args for GCN layer that has following key-word args
            K (int, optional) – Number of hops K. (default: 1)
            cached (bool, optional) – If set to True, the layer will cache the computation of (D^(-1/2)AD^(-1/2))^(K)X on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
            add_self_loops (bool, optional) – If set to False, will not add self-loops to the input graph. (default: True)
            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

        Returns
        -------
        see default_init()

        Notes
        -----
        documentation:
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SGConv
        source:
            https://arxiv.org/abs/1902.07153

        """
        self.TAGConv_init(in_size, out_size, hidden_size, n_hops, gcn_layer, gcn_kwargs, act_layer, act_kwargs, dropout)

    def forward(self, **kwargs):
        x, edge_index, edge_weight = kwargs.get("x"), kwargs.get("edge_index"), kwargs.get("edge_weight", None)
        if not self.use_edge_weights:
            edge_weight = None
        if self.debug:
            print(util.make_msg_block("GNN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
            if self.debug > 1:
                print(x)
        if self.debug and not edge_index is None:
            print("edge_index =", edge_index.shape)
            print(util.memory_of(edge_index))
        if self.debug and not edge_weight is None:
            print("edge_weight =", edge_weight.shape)
            print(util.memory_of(edge_weight))
        outputs = {}
        outputs["yhat"] = self.gnn_forward(**kwargs)
        return outputs

    def gcn_forward(self, gcn, **kwargs):
        _kwargs = util.copy_dict(kwargs)
        if not self.use_edge_weights and "edge_weight" in _kwargs:
            _kwargs.pop("edge_weight")
        if not self.use_edge_attributes and "edge_attr" in _kwargs:
            _kwargs.pop("edge_attr")
        return gcn(**kwargs)

    def default_forward(self, **kwargs):
        a = kwargs.get("x")
        i = 0
        # GNN layer(s) forward
        _kwargs = util.copy_dict(kwargs)
        for name, layer in self.name_layer_map.items():
            if name.startswith("gcn"):
                _kwargs["x"] = a
                a = self.gcn_forward(layer, **_kwargs)
                if self.debug:
                    print("GNN %d-Hop Encoding =" % (i+1), a.shape)
                    print(util.memory_of(a))
                i += 1
            else:
                a = layer(a)
        return a

    def AGNNConv_forward(self, **kwargs):
        """ Forward for AGNNConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def APPNP_forward(self, **kwargs):
        """ Forward for APPNPConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def ARMAConv_forward(self, **kwargs):
        """ Forward for ARMAConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight"])
        return self.default_forward(**_kwargs)

    def CGConv_forward(self, **kwargs):
        """ Forward for CGConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr"])
        return self.default_forward(**_kwargs)

    def ChebConv_forward(self, **kwargs):
        """ Forward for ChebConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)
        batch : (FloatTensor, optional) with shape=(|V|,)
        lambda_max : (FloatTensor, optional) with shape=(|G|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight", "batch", "lambda_max"])
        return self.default_forward(**_kwargs)

    def ClusterGCNConv_forward(self, **kwargs):
        """ Forward for ClusterGCNConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def DenseGCNConv_forward(self, **kwargs):
        """ Forward for DenseGCNConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) - node features X
        adj : FloatTensor with shape=(?, |V|, |V|) - adjacency matrix A
        mask : (BoolTensor, optional) with shape=(?, |V|) - Mask matrix M indicating the valid nodes for each graph. (default: None)
        add_loop : (bool, optional) – If set to False, the layer will not automatically add self-loops to the adjacency matrices. (default: True)

        Returns
        -------
        a (Tensor) - output with shape=(?, |V|, F')

        """
        if "edge_index" in kwargs:
            x = kwargs.get("x")
            edge_index = kwargs.get("edge_index")
            if not hasattr(self, "adj"):
                adj = torch.zeros((x.shape[-2], x.shape[-2]), device=x.device)
                setattr(self, "adj", adj)
            adj = getattr(self, "adj")
            adj[:,:] = 0
            adj[(edge_index[0],edge_index[1])] = 1
            kwargs["adj"] = adj
        elif not "adj" in kwargs:
            raise ValueError("Missing required adjacency matrix adj")
        _kwargs = util.filter_dict(kwargs, ["x", "adj", "mask", "add_loop"])
        return self.default_forward(**_kwargs)

    def DenseGINConv_forward(self, **kwargs):
        """ Forward for DenseGINConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) - node features X
        adj : FloatTensor with shape=(?, |V|, |V|) - adjacency matrix A
        mask : (BoolTensor, optional) with shape=(?, |V|) - Mask matrix M indicating the valid nodes for each graph. (default: None)
        add_loop : (bool, optional) – If set to False, the layer will not automatically add self-loops to the adjacency matrices. (default: True)

        Returns
        -------
        a (Tensor) - output with shape=(?, |V|, F')

        """
        return self.DenseGCNConv_forward(**kwargs)

    def DenseGraphConv_forward(self, **kwargs):
        """ Forward for DenseGraphConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(N, |V|, F) - node features X
        adj : FloatTensor with shape=(N, |V|, |V|) - adjacency matrix A
        mask : (BoolTensor, optional) with shape=(N, |V|) - Mask matrix M indicating the valid nodes for each graph. (default: None)

        Returns
        -------
        a (Tensor) - output with shape=(N, |V|, F')

        """
        return self.DenseGCNConv_forward(**kwargs)

    def DenseSAGEConv_forward(self, **kwargs):
        """ Forward for DenseSAGEConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) - node features X
        adj : FloatTensor with shape=(?, |V|, |V|) - adjacency matrix A
        mask : (BoolTensor, optional) with shape=(?, |V|) - Mask matrix M indicating the valid nodes for each graph. (default: None)

        Returns
        -------
        a (Tensor) - output with shape=(?, N, F')

        """
        return self.DenseGCNConv_forward(**kwargs)

    def DNAConv_forward(self, **kwargs):
        """ Forward for DNAConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (LongTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        a = kwargs.get("x")
        i = 0
        A = []
        # GNN layer(s) forward
        for name, layer in self.name_layer_map.items():
            if name.startswith("gcn"):
                A.append(a)
                _kwargs = util.filter_dict(kwargs, ["edge_index", "edge_weight"])
                _kwargs["x"] = a
                a = self.gcn_forward(layer, **_kwargs)
                if self.debug:
                    print("GNN %d-Hop Encoding =" % (i+1), a.shape)
                    print(util.memory_of(a))
                i += 1
            elif name == "dna":
                if self.n_hops == 1: # only DNA as gcn layer
                    A = [a]
                _kwargs = util.filter_dict(kwargs, ["edge_index", "edge_weight"])
                _kwargs["x"] = torch.stack(A, 1)
                a = self.gcn_forward(layer, **_kwargs)
            else:
                a = layer(a)
        return a

    def DynamicEdgeConv_forward(self, **kwargs):
        """ Forward for DynamicEdgeConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        batch : (LongTensor, optional) with shape=(|V|,) || tuple(batch, batch) if bipartite

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "batch"])
        return self.default_forward(**_kwargs)

    def ECConv_forward(self, **kwargs):
        """ Forward for ECConv GNN - alias of NNConv

        Arguments
        ---------
        see NNConv_forward()

        Returns
        -------
        see NNConv_forward()

        """
        return self.NNConv_forward(**kwargs)

    def EdgeConv_forward(self, **kwargs):
        """ Forward for EdgeConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def EGConv_forward(self, **kwargs):
        """ Forward for EGConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def FAConv_forward(self, **kwargs):
        """ Forward for EGConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        x_0 : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (LongTensor, optional) with shape=(2, |E|)
        return_attention_weights : (bool, optional)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')
        (edge_index, attention_weights) : (shape=(2, |E|), shape=(|E|,)) if return_attention_weights=True

        """
        _kwargs = util.filter_dict(kwargs, ["x", "x_0", "edge_index", "edge_weight", "return_attention_weights"])
        if self.gcn_0.normalize: # normalize == True => edge_weight == None
            if "edge_weight" in _kwargs:
                _kwargs.pop("edge_weight")
        elif not "edge_weight" in _kwargs: # normalize == False => edge_weight != None
            _kwargs["edge_weight"] = torch.ones(
                _kwargs["edge_index"].shape[:-2] + _kwargs["edge_index"].shape[-1:], 
                device=_kwargs["edge_index"].device
            )
        if not "x_0" in _kwargs:
            _kwargs["x_0"] = _kwargs["x"]
        a, a_0 = _kwargs.get("x"), _kwargs.get("x_0")
        i = 0
        # GNN layer(s) forward
        for name, layer in self.name_layer_map.items():
            if name.startswith("gcn"):
                _kwargs["x"] = a
                a = self.gcn_forward(layer, **_kwargs)
                if self.debug:
                    print("GNN %d-Hop Encoding =" % (i+1), a.shape)
                    print(util.memory_of(a))
                i += 1
            else:
                a = layer(a)
                if name == "in_proj": # must update x_0 so that x and x_0 have the same shape=(|V|, F)
                    _kwargs["x_0"] = layer(a_0)
        return a

    def FastRGCNConv_forward(self, **kwargs):
        """ Forward for FastRGCNConv GNN - alias of RGCNConv

        Arguments
        ---------
        see RGCNConv_forward()

        Returns
        -------
        see RGCNConv_forward()

        """
        return self.RGCNConv_forward(**kwargs)

    def FeaStConv_forward(self, **kwargs):
        """ Forward for FeaStConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def FlowConv_forward(self, **kwargs):
        """ Forward for FlowConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(?, |E|)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        if 0:
            _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight"])
        else:
            _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight", "xs", "xt", "group_index"])
        return self.default_forward(**_kwargs)

    def FiLMConv_forward(self, **kwargs):
        """ Forward for FiLMConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_type : (LongTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_type"])
        return self.default_forward(**_kwargs)

    def GATConv_forward(self, **kwargs):
        """ Forward for GATConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (LongTensor, optional) with shape=(|E|,)
        size : (tuple(int, int), optional)
        return_attention_weights : (bool, optional)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')
        (a, edge_index, attention_weights) : (shape=(|V|, F'), shape=(2, |E|), shape=(|E|, H)) if return_attention_weights=True

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size", "return_attetion_weights"])
        return self.default_forward(**_kwargs)

    def GatedGraphConv_forward(self, **kwargs):
        """ Forward for GatedGraphConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def GATv2Conv_forward(self, **kwargs):
        """ Forward for GatedGraphConv GNN

        Arguments
        ---------
        see GATConv_forward()

        Returns
        -------
        see GATConv_forward()

        """
        return self.GATConv_forward(**kwargs)

    def GCN2Conv_forward(self, **kwargs):
        """ Forward for GCN2Conv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        x_0 : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "x_0", "edge_index", "edge_weight"])
        if not "x_0" in _kwargs:
            _kwargs["x_0"] = _kwargs["x"]
        a, a_0 = _kwargs.get("x"), _kwargs.get("x_0")
        i = 0
        # GNN layer(s) forward
        for name, layer in self.name_layer_map.items():
            if name.startswith("gcn"):
                _kwargs["x"] = a
                a = self.gcn_forward(layer, **_kwargs)
                if self.debug:
                    print("GNN %d-Hop Encoding =" % (i+1), a.shape)
                    print(util.memory_of(a))
                i += 1
            else:
                a = layer(a)
                if name == "in_proj": # must update x_0 so that x and x_0 have the same shape=(|V|, F)
                    _kwargs["x_0"] = layer(a_0)
        return a

    def GCNConv_forward(self, **kwargs):
        """ Forward for GCN2Conv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight"])
        return self.default_forward(**_kwargs)

    def GENConv_forward(self, **kwargs):
        """ Forward for GENConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size"])
        return self.default_forward(**_kwargs)

    def GeneralConv_forward(self, **kwargs):
        """ Forward for GeneralConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size"])
        return self.default_forward(**_kwargs)

    def GINConv_forward(self, **kwargs):
        """ Forward for GINConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "size"])
        return self.default_forward(**_kwargs)

    def GINEConv_forward(self, **kwargs):
        """ Forward for GINEConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size"])
        x = _kwargs.get("x")
        edge_index = _kwargs.get("edge_index")
        # edge_attr can only be None if edge_index is a SparseTensor - see arguments in GINEConv_init()
        if not "edge_attr" in _kwargs and not isinstance(edge_index, SparseTensor):
            if not hasattr(self, "edge_attr"): # node and edge feature dimensionality must match
                self.edge_attr = torch.zeros((edge_index.shape[-1], x.shape[-1]), device=edge_index.device)
            _kwargs["edge_attr"] = self.edge_attr
        return self.default_forward(**_kwargs)

    def GMMConv_forward(self, **kwargs):
        """ Forward for GMMConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size"])
        return self.default_forward(**_kwargs)

    def GraphConv_forward(self, **kwargs):
        """ Forward for GraphConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight", "size"])
        return self.default_forward(**_kwargs)

    def GravNetConv_forward(self, **kwargs):
        """ Forward for GravNetConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        batch : (LongTensor, optional) with shape=(|V|,) || tuple(batch, batch) if bipartite

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "batch"])
        return self.default_forward(**_kwargs)

    def HGTConv_forward(self, **kwargs):
        """ Forward for HGTConv GNN

        Arguments
        ---------
        x_dict : {str: FloatTensor} with shape=(|V|, F) for each node type
        edge_index_dict : {str: LongTensor} with shape=(2, |E|) for each node type

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x_dict", "edge_index_dict"])
        return self.default_forward(**_kwargs)

    def HypergraphConv_forward(self, **kwargs):
        """ Forward for HypergraphConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        hyperedge_index : LongTensor with shape=(|V|, |E|)
        hyperedge_weight : (FloatTensor, optional) with shape=(|E|,)
        hyperedge_attr : (FloatTensor, optional) with shape=(|E|, D)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "hyperedge_index", "hyperedge_weight", "hyperedge_attr"])
        return self.default_forward(**_kwargs)

    def LEConv_forward(self, **kwargs):
        """ Forward for LEConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight"])
        return self.default_forward(**_kwargs)

    def MFConv_forward(self, **kwargs):
        """ Forward for MFConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "size"])
        return self.default_forward(**_kwargs)

    def NNConv_forward(self, **kwargs):
        """ Forward for NNConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size"])
        if not "edge_attr" in _kwargs: # Must have edge_attr even though PyG says its optional
            edge_index = _kwargs.get("edge_index")
            if not hasattr(self, "edge_attr"):
                self.edge_attr = torch.ones((edge_index.shape[-1], self.edge_size), device=edge_index.device)
            _kwargs["edge_attr"] = self.edge_attr
        return self.default_forward(**_kwargs)

    def PANConv_forward(self, **kwargs):
        """ Forward for PANConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        a = _kwargs.get("x")
        i = 0
        # GNN layer(s) forward
        _kwargs = util.copy_dict(kwargs)
        for name, layer in self.name_layer_map.items():
            if name.startswith("gcn"):
                _kwargs["x"] = a
                a, M = self.gcn_forward(layer, **_kwargs)
                if self.debug:
                    print("GNN %d-Hop Encoding =" % (i+1), a.shape)
                    print(util.memory_of(a))
                i += 1
            else:
                a = layer(a)
        return a

    def PDNConv_forward(self, **kwargs):
        """ Forward for PDNConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr"])
        return self.default_forward(**_kwargs)

    def PNAConv_forward(self, **kwargs):
        """ Forward for PNAConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr"])
        return self.default_forward(**_kwargs)

    def PointConv_forward(self, **kwargs):
        """ Forward for PointConv GNN - alias of PointNetConv

        Arguments
        ---------
        see PointNetConv_forward()

        Returns
        -------
        see PointNetConv_forward()

        """
        return self.PointNetConv_forward(**kwargs)

    def PointNetConv_forward(self, **kwargs):
        """ Forward for PointNetConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        pos : FloatTensor with shape=(|V|, 3) || tuple(pos, pos) if bipartite
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "pos", "edge_index"])
        return self.default_forward(**_kwargs)

    def PPFConv_forward(self, **kwargs):
        """ Forward for PPFConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        pos : FloatTensor with shape=(|V|, 3) || tuple(pos, pos) if bipartite
        normal : FloatTensor with shape=(|V|, 3) || tuple(pos, pos) if bipartite
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "pos", "normal", "edge_index"])
        return self.default_forward(**_kwargs)

    def ResGatedGraphConv_forward(self, **kwargs):
        """ Forward for PPFConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index"])
        return self.default_forward(**_kwargs)

    def RGCNConv_forward(self, **kwargs):
        """ Forward for RGCNConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_type : (LongTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_type"])
        edge_index = _kwargs.get("edge_index")
        # edge_type can only be None if edge_index is a SparseTensor - see arguments in RGCNConv_init()
        if not "edge_type" in _kwargs and not isinstance(edge_index, SparseTensor):
            if not hasattr(self, "edge_type"):
                self.edge_type = torch.zeros((edge_index.shape[-1],), dtype=torch.long, device=edge_index.device)
            _kwargs["edge_type"] = self.edge_type
        return self.default_forward(**_kwargs)

    def SAGEConv_forward(self, **kwargs):
        """ Forward for SAGEConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "size"])
        return self.default_forward(**_kwargs)

    def SGConv_forward(self, **kwargs):
        """ Forward for SGConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight"])
        return self.default_forward(**_kwargs)

    def SignedConv_forward(self, **kwargs):
        """ Forward for SignedConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F) || tuple(x, x) if bipartite
        pos_edge_index : LongTensor with shape=(2, |E^(+)|)
        neg_edge_index : LongTensor with shape=(2, |E^(-)|)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "pos_edge_index", "neg_edge_index"])
        if 0:
            if not "pos_edge_index" in _kwargs:
                _kwargs["pos_edge_index"] = kwargs["edge_index"]
            if not "neg_edge_index" in _kwargs:
                _kwargs["neg_edge_index"] = kwargs["edge_index"]
        return self.default_forward(**_kwargs)

    def SplineConv_forward(self, **kwargs):
        """ Forward for SplineConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (FloatTensor, optional) with shape=(|E|, D)
        size : (tuple(int, int), optional)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size"])
        return self.default_forward(**_kwargs)

    def SuperGATConv_forward(self, **kwargs):
        """ Forward for SuperGATConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        neg_edge_index : (LongTensor, optional) with shape=(2, |E^(-)|)
        batch : (LongTensor, optional) with shape=(|V|,)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "size"])
        return self.default_forward(**_kwargs)

    def TAGConv_forward(self, **kwargs):
        """ Forward for TAGConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(|E|,)

        Returns
        -------
        a : FloatTensor with shape=(?, |V|, F')

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_weight"])
        if 0:
            print(_kwargs.keys())
            for key, value in _kwargs.items():
                print(key, value.shape)
                if len(value.shape) < 3:
                    print(value)
            sys.exit(1)
        return self.default_forward(**_kwargs)

    def TransformerConv_forward(self, **kwargs):
        """ Forward for TransformerConv GNN

        Arguments
        ---------
        x : FloatTensor with shape=(|V|, F) || tuple(x, x) if bipartite
        edge_index : LongTensor with shape=(2, |E|)
        edge_attr : (LongTensor, optional) with shape=(|E|,)
        return_attention_weights : (bool, optional)

        Returns
        -------
        a : FloatTensor with shape=(|V|, F')
        (a, edge_index, attention_weights) : (shape=(|V|,  F'), shape=(2, |E|), shape=(|E|, H)) if return_attention_weights=True

        """
        _kwargs = util.filter_dict(kwargs, ["x", "edge_index", "edge_attr", "return_attetion_weights"])
        return self.default_forward(**_kwargs)

    def reset_parameters(self):
        for name, layer in self.name_layer_map.items():
            if hasattr(layer, "reset_parameters"):
#                print("\t", name)
                layer.reset_parameters()

    def ___str__(self):
        tab_space = 4 * " "
        lines = []
        lines.append("%s(" % (__class__.__name__))
        lines.append("%sin_size=%s\n" % (str(self.in_size)))
        lines.append("%sout_size=%s\n" % (str(self.out_size)))
        lines.append("%shidden_size=%s\n" % (str(self.hidden_size)))
        lines.append("%sn_hops=%s\n" % (str(self.n_hops)))
        lines.append("%sgcn_layer=%s\n" % (str(self.gcn_layer)))
        lines.append("%sgcn_kwargs=%s\n" % (str(self.gcn_kwargs)))
        lines.append("%sact_layer=%s\n" % (str(self.act_layer)))
        lines.append("%sact_kwargs=%s\n" % (str(self.act_kwargs)))
        lines.append("%sdropout=%.2f\n" % (str(self.dropout)))
        lines.append(")")
        return "\n".join(lines)


class GraphConstructor(Model):

    def __init__(self, graph_construction_method=["top-k", "Minkowski-2", 1.0], exog_alpha=0.5, cache_exog=True):
        super(GraphConstructor, self).__init__()
        if graph_construction_method is None:
            pass
        elif graph_construction_method[0] == "random":
            graph_construction_method.insert(None, 1)
        self.method = graph_construction_method
        self.exog_alpha = exog_alpha
        self.cache_exog = cache_exog

    def forward(self, **inputs):
#        self.debug = 1
        if self.debug:
            print(util.make_msg_block("GraphConstructor Forward"))
        outputs = {}
        if not self.method is None:
            sims = self.compute_similarities(**inputs)["sims"]
            _inputs = util.merge_dicts(inputs, {"sims": sims})
            outputs = self.sims_to_edges(**_inputs)
        return outputs

    def _compute_similarities(self, A, B):
        if self.debug:
            print("A =", A.shape)
            print(util.memory_of(A))
            if self.debug > 1:
                print(A)
            print("B =", B.shape)
            print(util.memory_of(B))
            if self.debug > 1:
                print(B)
        # Handle args
        if len(A.shape) < 2:
            raise NotImplementedError(A.shape)
        if len(B.shape) < 2:
            raise NotImplementedError(A.shape)
        # Setup
        sims = None
        sim_fns = self.method[1]
        if isinstance(sim_fns, str):
            sim_fns = [sim_fns]
        elif not isinstance(sim_fns, list):
            raise ValueError(sim_fns)
        # Start
        if "Minkowski" in sim_fns[0]:
            p = float(sim_fns[0].split("-")[-1])
            if len(A.shape) == 2:
                sims = torch.exp(
                    -torch.cdist(torch.unsqueeze(A, 0), torch.unsqueeze(B, 0), p) / A.shape[-1]
                )
                sims = torch.squeeze(sims, 0)
            else:
                sims = torch.exp(-torch.cdist(A, B, p) / A.shape[-1])
        elif sim_fns[0] == "cosine":
            numer = torch.einsum("...ij,...kj->...ik", A, B)
            norm = torch.sqrt(torch.outer(torch.sum(torch.pow(A, 2), -1), torch.sum(torch.pow(B, 2), -1)))
            # Two options:
            #   1. sims = torch.abs(cosims) : so negative similarity (sims=[-1,0) is NOT discounted
            #   2. sims = torch.clamp(cosims) : so negative similarity is completely discounted
            cosims = numer / torch.max(norm, torch.tensor(1e-8))
        elif sim_fns[0] == "dot":
            sims = torch.einsum("...ij,...kj->...ik", A, B)
        else:
            raise NotImplementedError(sim_fns[0])
        for fn in sim_fns[1:]:
            if fn == "ReLU":
                sims = F.relu(sims)
            elif fn == "Softmax":
                sims = F.softmax(sims, -1)
            else:
                raise NotImplementedError(fn)
        return sims

    def compute_similarities(self, **inputs):
        # Compute a similarity metric between all spatial element pairs
        #   unpack the inputs
        x, m, b = inputs["x"], inputs.get("m", None), inputs.get("b", None)
        reduce_fn, reduce_dim = inputs.get("reduce_fn", torch.mean), inputs.get("reduce_dim", None)
        agg_fn, agg_weight = inputs.get("agg_fn", sum), inputs.get("agg_weight", 1.0)
        outputs = {}
        if not isinstance(x, list):
            x = [x]
        if not isinstance(reduce_fn, list):
            reduce_fn = [reduce_fn for _ in x]
        if not isinstance(reduce_dim, list):
            reduce_dim = [reduce_dim for _ in x]
        if not isinstance(agg_weight, list) and not isinstance(agg_weight, torch.Tensor):
            agg_weight = [agg_weight for _ in x]
        sims = []
        for _x, _reduce_fn, _reduce_dim in zip(x, reduce_fn, reduce_dim):
            sim = self._compute_similarities(_x, _x)
            if not _reduce_dim is None:
                sim = _reduce_fn(sim, _reduce_dim)
            sims.append(sim)
            if self.debug:
                print("Similarity =", sim.shape)
                print(util.memory_of(sim))
                if self.debug > 1:
                    print(sim)
        sims = pyt_util.align(sims, -1)
        for i in range(len(sims)):
            sims[i] = agg_weight[i] * sims[i]
        sim = agg_fn(sims)
        if not m is None:
            sim = sim * m
        if not b is None:
            sim = sim + b
        if self.debug:
            print("Similarity =", sim.shape)
            print(util.memory_of(sim))
            if self.debug > 1:
                print(sim)
        outputs["sims"] = sim
        return outputs

    def _sims_to_edges(self, sims):
        n_spatial, n_spatial = sims.shape[-2:]
        edge_index, edge_weight, W = None, None, None
        if self.method[0] =="top-k":
            k = self.method[-1]
            if isinstance(k, float):
                k = int(k * n_spatial**2 + 0.5)
            values, indices = torch.topk(torch.reshape(sims, sims.shape[:-2] + (-1,)), k)
            edge_index = torch.transpose(pyt_util.unravel_index(indices, sims.shape[-2:]), -2, -1)
            edge_weight = values
        elif self.method[0] == "k-nn":
            n_spatial, n_spatial = sims.shape[-2:]
            k = self.method[-1]
            if isinstance(k, float):
                k = int(k * n_spatial**2 + 0.5)
            values, indices = torch.topk(sims, k)
            edge_index = torch.stack(
                [
                    torch.arange(n_spatial, device=sims.device).repeat_interleave(k), 
                    torch.reshape(indices, sims.shape[:-2] + (-1,))
                ]
            )
            edge_weight = torch.reshape(values, sims.shape[:-2] + (-1,))
        elif self.method[0] == "threshold":
            comparator, threshold = self.method[-2:]
            indices = torch.where(util.comparator_fn_map[comparator](sims, threshold))
            edge_index = torch.stack(indices, 0)
            edge_weight = sims[indices]
        elif self.method[0] == "range":
            lower, upper = self.method[-2:]
            indices = torch.where(torch.logical_and(sims >= lower, sims <= upper))
            edge_index = torch.stack(indices, 0)
            edge_weight = sims[indices]
        elif self.method[0] == "random":
            raise NotImplementedError(self.method)
        else:
            raise NotImplementedError("Unknown edge construction heuristic \"%s\"" % (self.method[0]))
        if edge_index.shape[-1] < n_spatial**2: # Creating graph with some sparsity - apply masking to get W
            M = torch.zeros((n_spatial, n_spatial), device=sims.device)
            M[...,edge_index[0],edge_index[1]] = 1
            W = M * sims
        else: # Creating fully-connected graph
            W = sims
        return edge_index, edge_weight, W

    def sims_to_edges(self, **inputs):
        sims = inputs["sims"]
        orig_edge_index = inputs.get("orig_edge_index", None)
        excl_edge_index = inputs.get("excl_edge_index", None)
        ignore_self_loops = inputs.get("ignore_self_loops", True)
        prune = inputs.get("prune", None)
        # Start
        orig_edge_weight = None
        outputs = {}
        if len(sims.shape) == 2: # Construct one set of edges
            n_spatial, n_spatial = sims.shape
            if not orig_edge_index is None: # Save similarities on original edges
                orig_edge_weight = sims[orig_edge_index[0],orig_edge_index[1]]
                if self.debug:
                    print("Original Edge Weight =", orig_edge_weight.shape)
                    print(util.memory_of(orig_edge_weight))
                    if self.debug > 1:
                        print(orig_edge_weight)
            if ignore_self_loops:
                if self.debug:
                    print(util.make_msg_block("Masking Self Loops", "-"))
                sims.fill_diagonal_(0) # Ignore self-similarity to avoid self-loops
            if not excl_edge_index is None: # Ignore excluded edge similarities
                if self.debug:
                    print(util.make_msg_block("Masking Excluded Edges", "-"))
                excl_edge_weight = sims[excl_edge_index[0],excl_edge_index[1]]
                sims[excl_edge_index[0],excl_edge_index[1]] = 0
            edge_index, edge_weight, W = self._sims_to_edges(sims)
        elif len(sims.shape) == 3: # Construct a set of edges for each time-step
            n_temporal, n_spatial, n_spatial = sims.shape
            if not orig_edge_index is None: # Save similarities on original edges
                orig_edge_weight = sims[:,orig_edge_index[0],orig_edge_index[1]]
                outputs["orig_edge_weight"] = orig_edge_weight
                if self.debug:
                    print("Original Edge Weight =", orig_edge_weight.shape)
                    print(util.memory_of(orig_edge_weight))
                    if self.debug > 1:
                        print(orig_edge_weight)
            if ignore_self_loops:
                if self.debug:
                    print(util.make_msg_block("Masking Self Loops", "-"))
                for i in range(n_temporal): # Ignore self-similarity to avoid self-loops
                    sims[i].fill_diagonal_(0)
            if not excl_edge_index is None: # Ignore excluded edge similarities
                if self.debug:
                    print(util.make_msg_block("Masking Excluded Edges", "-"))
                excl_edge_weight = sims[:excl_edge_index[0],excl_edge_index[1]]
                sims[:,excl_edge_index[0],excl_edge_index[1]] = 0
            edge_index, edge_weight, W = self._sims_to_edges(sims)
        elif len(sims.shape) == 4: # Construct a set of edges for each time-step
            n_sample, n_temporal, n_spatial, n_spatial = sims.shape
            if not orig_edge_index is None: # Save similarities on original edges
                orig_edge_weight = sims[:,:,orig_edge_index[0],orig_edge_index[1]]
                outputs["orig_edge_weight"] = orig_edge_weight
                if self.debug:
                    print("Original Edge Weight =", orig_edge_weight.shape)
                    print(util.memory_of(orig_edge_weight))
                    if self.debug > 1:
                        print(orig_edge_weight)
            if ignore_self_loops:
                if self.debug:
                    print(util.make_msg_block("Masking Self Loops", "-"))
                sims = (1 - torch.eye(n_spatial, device=sims.device)) * sims
            if not excl_edge_index is None: # Ignore excluded edge similarities
                if self.debug:
                    print(util.make_msg_block("Masking Excluded Edges", "-"))
                excl_edge_weight = sims[:,:excl_edge_index[0],excl_edge_index[1]]
                sims[:,:,excl_edge_index[0],excl_edge_index[1]] = 0
            edge_index, edge_weight, W = self._sims_to_edges(sims)
        if not prune is None:
            if prune[0] == "weight":
                if self.debug:
                    print(util.make_msg_block("Applying Secondary Pruning", "-"))
                comparator, value = prune[1:]
                keep = torch.where(torch.logical_not(util.comparator_fn_map[comparator](edge_weight, value)))
                edge_index = torch.index_select(edge_index, -1, keep[0])
                edge_weight = torch.index_select(edge_weight, -1, keep[0])
            else:
                raise NotImplementedError(prune)
        if self.debug:
            print("Edge Index =", edge_index.shape)
            print(util.memory_of(edge_index))
            if self.debug > 1:
                print(edge_index)
            print("Edge Weight =", edge_weight.shape)
            print(util.memory_of(edge_weight))
            if self.debug > 1:
                print(edge_weight)
            if not W is None:
                print("W =", W.shape)
                if self.debug > 1:
                    print(W)
        outputs["edge_index"] = edge_index
        outputs["edge_weight"] = edge_weight
        outputs["orig_edge_weight"] = orig_edge_weight
        outputs["W"] = W
        return outputs

    def reset_parameters(self):
        pass


class AdaEdgeGNN(Model):

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        n_nodes=-1, 
        n_edges=-1,
        gnn_kwargs={},
        gc_kwargs={},
        learn_edge_weights=True,
    ):
        super(AdaEdgeGNN, self).__init__()
        # Instantiate model Layers
        self.gnn = GNN(in_size, out_size, hidden_size, **gnn_kwargs)
        self.gc = GraphConstructor(**gc_kwargs)
        self.edges_adapted, self.n_added_edges = False, 0
        if not self.gc.method is None:
            if self.gc.method[0] == "threshold" or self.gc.method[-1] > 0:
                self.edges_adapted = True
                c = self.gc.method[-1]
                self.n_added_edges = c if isinstance(c, int) else int(c * n_nodes**2)
        learn_edge_weights = learn_edge_weights and self.gnn.use_edge_weights
        if learn_edge_weights:
            n_learned_edges = n_edges + self.n_added_edges
            if n_learned_edges < 1:
                raise ValueError(
                    "Input n_learned_edges=%d must be positive if learn_edge_weights=True" % (n_learned_edges)
                )
            self.edge_weight = torch.nn.Parameter(torch.ones(n_learned_edges))
        # Save vars
        self.in_size, self.out_size = in_size, out_size
        self.hidden_size, self.n_hops = self.gnn.hidden_size, self.gnn.n_hops
        self.gcn_layer, self.act_layer = self.gnn.gcn_layer, self.gnn.act_layer
        self.dropout = self.gnn.dropout
        self.gc_method, self.exog_alpha = self.gc.method, self.gc.exog_alpha
        self.learn_edge_weights = learn_edge_weights

    def forward(self, **inputs):
#        self.debug = 1
        self.gnn.debug = self.debug
        self.gc.debug = self.debug
        gnn_kwargs, gc_kwargs = inputs.get("gnn_kwargs", None), inputs.get("gc_kwargs", None)
        x = gnn_kwargs["x"]
        edge_index, edge_weight = gnn_kwargs.get("edge_index", None), gnn_kwargs.get("edge_weight", None)
        if self.debug:
            print(util.make_msg_block("AdaEdgeGNN Forward"))
        if self.debug:
            print("x =", x.shape)
            if self.debug > 1:
                print(x)
        outputs = {}
        if self.edges_adapted:
            if self.debug:
                print(util.make_msg_block("Adapting Edges + GNN", "-"))
            gc_outputs = self.gc(**gc_kwargs)
            added_edge_index, added_edge_weight = gc_outputs["edge_index"], gc_outputs["edge_weight"]
            if "reduce_dim" in gc_kwargs:
                added_edge_index = torch.unsqueeze(added_edge_index, gc_kwargs["reduce_dim"])
                added_edge_weight = torch.unsqueeze(added_edge_weight, gc_kwargs["reduce_dim"])
            outputs["added_edge_index"] = added_edge_index
            outputs["added_edge_weight"] = added_edge_weight
            outputs["W"] = gc_outputs.get("W", None)
            if self.learn_edge_weights:
                ada_edge_index, _ = self.adapt_edges(edge_index, added_edge_index)
                ada_edge_weight = self.edge_weight
            elif edge_weight is None:
                ada_edge_index, ada_edge_weight = self.adapt_edges(
                    edge_index, added_edge_index, gc_outputs.get("orig_edge_weight", None), added_edge_weight
                )
            else:
                ada_edge_index, ada_edge_weight = self.adapt_edges(
                    edge_index, added_edge_index, edge_weight, added_edge_weight
                )
            if 0:
                print("ada_edge_index =", ada_edge_index.shape)
                print(ada_edge_index)
                print("ada_edge_weight =", ada_edge_weight.shape)
                print(ada_edge_weight)
            gnn_kwargs = util.merge_dicts(
                gnn_kwargs, 
#                {"edge_index": ada_edge_index, "edge_weight": ada_edge_weight}
                {"edge_index": gc_outputs["W"]}
            )
            a = self.gnn(**gnn_kwargs)["yhat"]
        else:
            if self.debug:
                print(util.make_msg_block("Standard GNN", "-"))
            if self.learn_edge_weights:
                if self.debug:
                    print(util.make_msg_block("Subbing Learned Weights", "-"))
                gnn_kwargs["edge_weight"] = self.edge_weight
            if self.gcn_layer == "FlowConv":
                W = torch.zeros((x.shape[-2], x.shape[-2]), device=x.device)
                if not edge_index is None:
                    if edge_weight is None:
                        W[edge_index[1],edge_index[0]] = 1
                    else:
                        W[edge_index[1],edge_index[0]] = edge_weight
            gnn_kwargs = util.merge_dicts(gnn_kwargs, {"edge_index": W})
            a = self.gnn(**gnn_kwargs)["yhat"]
        if self.debug:
            print("AdaEdgeGNN Encoding =", a.shape)
            print(util.memory_of(a))
        if self.debug and False:
            sys.exit(1)
        outputs["yhat"] = a
        return outputs

    def adapt_edges(self, orig_edge_index, added_edge_index, orig_edge_weight=None, added_edge_weight=None):
        if orig_edge_index is None:
            return added_edge_index, added_edge_weight
        if 0:
            print("orig_edge_index =", orig_edge_index.shape)
            print(orig_edge_index)
            print("added_edge_index =", added_edge_index.shape)
            print(added_edge_index)
        edge_index = pyt_util.maybe_expand_then_cat((orig_edge_index, added_edge_index), -1)
        if not (orig_edge_weight is None or added_edge_weight is None):
            if 0:
                print("orig_edge_weight =", orig_edge_weight.shape)
                print(orig_edge_weight)
                print("added_edge_weight =", added_edge_weight.shape)
                print(added_edge_weight)
            edge_weight = pyt_util.maybe_expand_then_cat((orig_edge_weight, added_edge_weight), -1)
        else:
            edge_weight = None
        if 0:
            print("edge_index =", edge_index.shape)
            print(edge_index)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
                print(edge_weight)
        if self.debug:
            print("Adapted Edge Index =", edge_index.shape)
            print(util.memory_of(edge_index))
            if self.debug > 1:
                print(edge_index)
        if self.debug and not edge_weight is None:
            print("Adapted Edge Weight =", edge_weight.shape)
            print(util.memory_of(edge_weight))
            if self.debug > 1:
                print(edge_weight)
        return edge_index, edge_weight

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.gc.reset_parameters()
        if self.learn_edge_weights:
            torch.nn.init.uniform_(self.edge_weight)
            if 1:
                pass
            else:
                self.edge_weight_query_proj.reset_parameters()
                self.edge_weight_key_proj.reset_parameters()


class TemporalMapper(Model):

    def __init__(self, in_size, out_size, temporal_mapper="last", temporal_mapper_kwargs={}):
        super(TemporalMapper, self).__init__()
        # Setup
        self.supported_methods = ["last", "last_n", "attention"]
        assert temporal_mapper in self.supported_methods, "Temporal mapping method \"%s\" not supported" % (method)
        self.method_init_map = {}
        for method in self.supported_methods:
            self.method_init_map[method] = getattr(self, "%s_init" % (method))
        self.method_mapper_map = {}
        for method in self.supported_methods:
            self.method_mapper_map[method] = getattr(self, "%s_mapper" % (method))
        # Instantiate method
        self.method_init_map[temporal_mapper](in_size, out_size, temporal_mapper_kwargs)
        self.mapper = self.method_mapper_map[temporal_mapper]

    def last_init(self, in_size, out_size, kwargs={}):
        pass

    def last_n_init(self, in_size, out_size, kwargs={}):
        pass

    def attention_init(self, in_size, out_size, kwargs={}):
        attention_kwargs = util.merge_dicts(
            {"num_heads": 1, "dropout": 0.0, "kdim": in_size, "vdim": in_size},
            kwargs
        )
        self.attn = self.layer_fn_map["MultiheadAttention"](out_size, **attention_kwargs)

    # Preconditions:
    #   inputs={"x": torch.float, "n_temporal_out": int}
    #   x.shape=(n_sample, n_temporal_in, n_predictors)
    # Postconditions:
    #   a.shape=(n_sample, n_temporal_out, n_responses)
    def forward(self, **inputs):
        if self.debug:
            print(util.make_msg_block("TemporalMapper Forward"))
        return self.mapper(**inputs)

    def last_mapper(self, **inputs):
        if self.debug:
            print(util.make_msg_block("last_mapper() forward"))
        x, temporal_dim = inputs["x"], inputs.get("temporal_dim", -2)
        if self.debug:
            print("x =", x.shape)
            print("temporal_dim =", temporal_dim)
        return {
            "yhat": torch.index_select(x, temporal_dim, torch.tensor(x.shape[temporal_dim]-1, device=x.device))
        }

    def last_n_mapper(self, **inputs):
        x, n_temporal_out, temporal_dim = inputs["x"], inputs["n_temporal_out"], inputs.get("temporal_dim", -2)
        if self.debug:
            print("x =", x.shape)
            print("temporal_dim =", temporal_dim)
        t = x.shape[temporal_dim]
        return {
            "yhat": torch.index_select(
                x, temporal_dim, torch.tensor(range(t-n_temporal_out, t), device=x.device)
            )
        }

    def attention_mapper(self, **inputs):
        n_temporal_out, temporal_dim = inputs["n_temporal_out"], inputs.get("temporal_dim", -2)
        if "Q" in inputs:
            Q = inputs["Q"]
        else:
            Q = torch.transpose(inputs["x"], 0, temporal_dim)[-n_temporal_out:]
        if "K" in inputs:
            K = inputs["K"]
        else:
            K = torch.transpose(inputs["x"], 0, temporal_dim)
        if "V" in inputs:
            V = inputs["V"]
        else:
            V = torch.transpose(inputs["x"], 0, temporal_dim)
        if self.debug:
            print("Q =", Q.shape)
            print("K =", K.shape)
            print("V =", V.shape)
        a, w = self.attn(Q, K, V)
        return {"yhat": torch.transpose(a, 0, temporal_dim), "attn_weights": w}

    def reset_parameters(self):
        if hasattr(self, "attn"):
            self.attn._reset_parameters()


class Aggregator(Model):

    def __init__(self, agg_size, aggregation_method="learned_lr"):
        super(Aggregator, self).__init__()
        getattr(self, "%s_init" % (aggregation_method))(agg_size)
        self.agg_fn = getattr(self, "%s_forward" % (aggregation_method))
        # Save all vars
        self.agg_size = agg_size
        self.method = aggregation_method

    def static_l_init(self, agg_size):
        pass

    def static_r_init(self, agg_size):
        pass

    def static_lr_init(self, agg_size):
        pass

    def learned_l_init(self, agg_size):
        self.agg_weights_l = torch.nn.Parameter(torch.ones((agg_size,)))

    def learned_r_init(self, agg_size):
        self.agg_weights_r = torch.nn.Parameter(torch.ones((agg_size,)))

    def learned_lr_init(self, agg_size):
        self.agg_weights_l = torch.nn.Parameter(torch.ones((agg_size,)))
        self.agg_weights_r = torch.nn.Parameter(torch.ones((agg_size,)))

    def learned_reciprocal_l_init(self, agg_size):
        self.agg_weights = torch.nn.Parameter(torch.ones((agg_size,)))
        self.agg_act = self.layer_fn_map["Sigmoid"]()

    def learned_reciprocal_r_init(self, agg_size):
        self.agg_weights = torch.nn.Parameter(torch.ones((agg_size,)))
        self.agg_act = self.layer_fn_map["Sigmoid"]()

    def learned_reciprocal_lr_init(self, agg_size):
        self.agg_weights = torch.nn.Parameter(torch.ones((2, agg_size)))
        self.agg_act = self.layer_fn_map["Softmax"](0)

    def learned_contrastive_init(self, agg_size):
        self.agg_weights = torch.nn.Parameter(torch.ones((agg_size,)))

    def forward(self, **kwargs):
#        self.debug = 1
        L, R, dim = kwargs["L"], kwargs["R"], kwargs.get("dim", -2)
        if self.debug:
            print(util.make_msg_block("Aggregator Forward"))
        if self.debug and not L is None:
            print("L =", L.shape)
            print(util.memory_of(L))
        if self.debug and not R is None:
            print("R =", R.shape)
            print(util.memory_of(R))
        if self.debug:
            print("Aggregation Dim =", dim)
        a = self.agg_fn(L, R, dim)
        if self.debug:
            print("Aggregation =", a.shape)
            print(util.memory_of(a))
        return a

    def learned_l_forward(self, L, R, dim):
        return pyt_util.bcast_safe_view(self.agg_weights_l, L, dim) * L + R

    def learned_r_forward(self, L, R, dim):
        return L + pyt_util.bcast_safe_view(self.agg_weights_r, R, dim) * R

    def learned_lr_forward(self, L, R, dim):
        return pyt_util.bcast_safe_view(self.agg_weights_l, L, dim) * L + pyt_util.bcast_safe_view(self.agg_weights_r, R, dim) * R

    def learned_reciprocal_l_forward(self, L, R, dim):
        weights = bcast_safe_view(self.agg_weights, L, dim)
        return self.agg_act(weights) * L + (1 - self.agg_act(weights)) * R

    def learned_reciprocal_r_forward(self, L, R, dim):
        weights = bcast_safe_view(self.agg_weights, L, dim)
        return (1 - self.agg_act(weights)) * L + self.agg_act(weights) * R

    def learned_reciprocal_lr_forward(self, L, R, dim):
        weights = self.agg_act(self.agg_weights)
        weights_l = pyt_util.bcast_safe_view(weights[0,:], L, dim)
        weights_r = pyt_util.bcast_safe_view(weights[1,:], R, dim)
        return weights_l * L + weights_r * R

    def learned_contrastive_forward(self, L, R, dim):
        return (1 - pyt_util.bcast_safe_view(self.agg_weights, L, dim)) * L + pyt_util.bcast_safe_view(self.agg_weights, R, dim) * R

    def static_l_forward(self, L, R, dim):
        return L

    def static_r_forward(self, L, R, dim):
        return R

    def static_lr_forward(self, L, R, dim):
        return L + R

    def check_args(self, L, R, dim):
        if L.shape != R.shape:
            raise NotImplementedError(
                "Unknown shapes for inputs L and R at %s and %s. Only one-to-one shapes are supported." % (
                    L.shape, R.shape
                )
            )

    def reset_parameters(self):
        pass


class GNN_HyperparameterVariables(Container):

    def __init__(self):
        self.n_hops = 1
        self.gcn_layer = "GCNConv"
        self.gcn_kwargs = {}
        self.act_layer = "Identity"
        self.act_kwargs = {}
        self.dropout = 0.0
        self.use_edge_weights = True
        self.use_edge_attributes = True


class GraphConstructor_HyperparameterVariables(Container):

    def __init__(self):
        self.graph_construction_method = ["top-k", "Minkowski-2", 49]
        self.graph_construction_method = ["top-k", "Minkowski-2", int(1276**2)-1275-1276]
        self.graph_construction_method = ["top-k", "dot-ReLU", 1.0]
        self.exog_alpha = 0.5


class Cluster_HyperparameterVariables(Container):

    def __init__(self):
        self.cluster_method = "Agglomerative"
        self.n_clusters = 3
        self.clustered_representation = "identity"


class AdaEdgeGNN_HyperparameterVariables(Container):

    def __init__(self):
        self.gnn_kwargs = GNN_HyperparameterVariables()
        self.gc_kwargs = GraphConstructor_HyperparameterVariables()
        self.learn_edge_weights = False


class RNN_HyperparameterVariables(Container):

    def __init__(self):
        self.n_rnn_layers = 1
        self.rnn_layer = "LSTM"
        self.rnn_kwargs = {}
        self.dropout = 0.0


class TemporalMapper_HyperparameterVariables(Container):

    def __init__(self):
        self.temporal_mapper = "last"
        self.temporal_mapper_kwargs = {}


class Aggregator_HyperparameterVariables(Container):

    def __init__(self):
        self.aggregation_method = "learned_reciprocal_lr"
