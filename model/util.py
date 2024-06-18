import re
import sys
import torch
import numpy as np

import util
import model.mytorch.util as pyt_util
import model.mytorch_geometric.util as pyg_util


#   init activation -> function map
act_fn_map = util.sort_dict(pyt_util.act_fn_map)

#   init optimization -> class constructor function map
opt_fn_map = util.sort_dict(pyt_util.opt_fn_map)

#   init scheduler -> function map
sched_fn_map = util.sort_dict(pyt_util.sched_fn_map)

#   init initialization -> function map
init_fn_map = util.sort_dict(pyt_util.init_fn_map)

#   init loss -> class constructor function map
loss_fn_map = util.sort_dict(pyt_util.loss_fn_map)

#   init layer -> class constructor function map
layer_fn_map = util.sort_dict(pyt_util.layer_fn_map)

#   init gcn layer -> class constructor function map
gcnlayer_fn_map = util.sort_dict(pyg_util.gcnlayer_fn_map)

#   init gcn layer -> supported feature list map
gcnlayer_supported_map = util.sort_dict(pyg_util.gcnlayer_supported_map)

#   init gcn layer -> requirements dict map
gcnlayer_required_map = util.sort_dict(pyg_util.gcnlayer_required_map)


def masked_softmax(x, mask=None, **kwargs):
    """
    Performs masked softmax, as simply masking post-softmax can be inaccurate
    :param x: [batch_size, num_items]
    :param mask: [batch_size, num_items]
    :return:
    """
    dim = kwargs.get("dim", -1)
    if mask is not None:
        mask = mask.float()
    if mask is not None:
        x_masked = x * mask + (1 - 1 / mask)
    else:
        x_masked = x
    x_max = x_masked.max(dim)[0]
    x_exp = (x - x_max.unsqueeze(-1)).exp()
    if mask is not None:
        x_exp = x_exp * mask.float()
    return x_exp / x_exp.sum(dim).unsqueeze(-1)


def reset_parameters(model):
    for name, module in model.named_modules():
        if module == model: # avoid infinite recursion
            continue
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return model


def init_params_from_model(model_a, model_b): # init parameters of model_a using model_b
    name_param_dict = dict(model_a.named_parameters())
    for name_b, param_b in model_b.named_parameters():
        if name_b in name_param_dict:
            param_a = name_param_map[name_b]
            if param_a.shape != param_b.shape:
                raise ValueError("Shape mismatch for coincident parameter %s from initializer model %s. Expected parameter %s.shape=%s but received %s.shape=%s" % (
                    name_b, model_b.name(), name_a, param_a.shape, name_b, param_b.shape
                    )
                )
            param_a.data.copy_(param_b.data)
    return model_a


def init_params(model, init, seed=None, **kwargs):
    debug = kwargs.get("debug", 0)
    # Start by setting the seed and calling reset_parameters()
    torch.manual_seed(time.time() if seed is None else seed)
    model = reset_parameters(model)
    # Reset seed in-case a different initialization follows
    torch.manual_seed(time.time() if seed is None else seed)
    if init is None: # init from default (reset_parameters()) method
        pass # done by default
    elif isinstance(init, torch.nn.Module): # init from model
        model = init_params_from_model(model, init)
    elif isinstance(init, dict): # init with different initializer for each group
        for name, param in self.named_parameters():
            for regex_name, _init in init.items():
                if re.match(regex_name, name):
                    if _init == "constant_":
                        init_fn_map[_init](param, 0.0)
                    else:
                        try:
                            init_fn_map[_init](param)
                        except ValueError as err:
                            pass
                    break
    elif isinstance(init, str): # init from one initializer using constant_ for bias terms
        if init == "AGCRN":
            for name, p in model.named_parameters():
                if debug:
                    print(name, ":", p.shape)
                if p.dim() > 1:
                    if debug:
                        print("xavier_uniform_()")
                    torch.nn.init.xavier_uniform_(p)
                else:
                    if debug:
                        print("uniform_()")
                    torch.nn.init.uniform_(p)
        else:
            for name, param in model.named_parameters():
                if debug:
                    print(name, ":", param.shape)
                if param.dim() > 1:
                    if debug:
                        print(init)
                    init_fn_map[init](param)
                else:
                    if debug:
                        print("constant_")
                    init_fn_map["constant_"](param, 0.0)
    else:
        raise NotImplementedError(init)
    return model


class StandardLoader(torch.utils.data.Dataset):

    __sampled__ = ["x", "y"]
    debug = 0

    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        if "__sampled__" in data:
            self.__sampled__ = data["__sampled__"]
        if not "x" in data:
            raise ValueError("Input data must contain field \"x\"")
        if not all(item in data for item in self.__sampled__):
            raise ValueError("Input data must contained sampled fields %s" % (str(self.__sampled__)))
        self.data = data
        self.mb = {}

    def __len__(self):
        return self.data["x"].shape[0]

    def __getitem__(self, key):
        if self.debug:
            print("key =", type(key), "= len(%d)" % (len(key)), "=")
            if self.debug > 1:
                print(key)
        if self.debug:
            for item, data in self.data.items():
                print(item, "=", type(data), "=", data.shape, "=")
                if self.debug > 1:
                    print(data)
        for item in self.__sampled__:
            self.mb[item] = self.data[item][key]
        if self.debug:
            for item, sample in self.mb.items():
                print(item, "=", type(sample), "=", sample.shape, "=")
                if self.debug > 1:
                    print(sample)
        self.mb["__index__"] = key
        if self.debug:
            sys.exit(1)
        return self.mb


class PartitionLoader(torch.utils.data.Dataset):

    __sampled__ = ["x", "y"]
    debug = 0

    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        if "__sampled__" in data:
            self.__sampled__ = data["__sampled__"]
        if not ("x" in data and "indices" in data):
            raise ValueError("Input data must contain fields \"x\" and \"indices\"")
        if not all(item in data for item in self.__sampled__):
            raise ValueError("Input data must contained sampled fields %s" % (str(self.__sampled__)))
        self.indices = data["indices"]
        self.data = data
        self.mb = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if self.debug:
            print("key =", type(key), "= len(%d)" % (len(key)), "=")
            if self.debug > 1:
                print(key)
        if self.debug:
            for item, data in self.data.items():
                print(item, "=", type(data), "=", data.shape, "=")
                if self.debug > 1:
                    print(data)
        for item in self.__sampled__:
            self.mb[item] = self.data[item][self.indices[key]]
        if self.debug:
            for item, sample in self.mb.items():
                print(item, "=", type(sample), "=", sample.shape, "=")
                if self.debug > 1:
                    print(sample)
        self.mb["__index__"] = self.indices[key]
        if self.debug:
            sys.exit(1)
        return self.mb


class SlidingWindowLoader(torch.utils.data.Dataset):

    __sampled__ = ["x", "y"]
    __inputs__ = ["x"]
    __outputs__ = ["y"]
    debug = 0

    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        if "__sampled__" in data:
            self.__sampled__ = data["__sampled__"]
        if "__inputs__" in data:
            self.__inputs__ = data["__inputs__"]
        if "__outputs__" in data:
            self.__outputs__ = data["__outputs__"]
        if not ("x" in data and "y" in data):
            raise ValueError("Arugment data must contain fields \"x\" and \"y\"")
        if not all(item in data for item in self.__sampled__):
            raise ValueError("Argument data must contained sampled fields %s" % (str(self.__sampled__)))
        if not all(item in data for item in self.__inputs__):
            raise ValueError("Arugment data must contained input fields %s" % (str(self.__inputs__)))
        if not all(item in data for item in self.__outputs__):
            raise ValueError("Arugment data must contained output fields %s" % (str(self.__outputs__)))
        if len(data["x"].shape) != len(data["y"].shape):
            raise ValueError(
                "Data x and y must have equal dimensionality, received x.shape=%s and y.shape=%s" % (
                    str(data["x"].shape),
                    str(data["y"].shape)
                )
            )
        if len(data["x"].shape) == 3: # Original spatiotemporal format
            self.n_temporal, self.n_spatial, self.n_feature = data["x"].shape
            self.in_indices, self.out_indices = util.input_output_window_indices(
                self.n_temporal,
                var.temporal_mapping[0],
                var.temporal_mapping[1],
                var.horizon
            )
        elif len(data["x"].shape) == 4: # Reduced spatiotemporal format
            self.n_channel, self.n_temporal, self.n_spatial, self.n_feature = data["x"].shape
            self.in_indices, self.out_indices = util.input_output_window_indices(
                self.n_temporal,
                var.temporal_mapping[0],
                var.temporal_mapping[1],
                var.horizon
            )
            if self.debug:
                print(self.in_indices.shape, self.out_indices.shape)
            self.in_indices = np.tile(
                self.in_indices,
                (self.n_channel, 1, 1)
            ) + np.arange(self.n_channel)[:,None,None] * self.in_indices.shape[0]
            self.out_indices = np.tile(
                self.out_indices,
                (self.n_channel, 1, 1)
            ) + np.arange(self.n_channel)[:,None,None] * self.out_indices.shape[0]
            if self.debug:
                print(self.in_indices.shape, self.out_indices.shape)
                print(self.in_indices)
                print(self.out_indices)
            data["x"] = torch.reshape(data["x"], (-1,) + data["x"].shape[2:])
            data["y"] = torch.reshape(data["y"], (-1,) + data["y"].shape[2:])
            if self.debug:
                print(data["x"].shape, data["y"].shape)
            self.in_indices = np.reshape(self.in_indices, (-1,) + self.in_indices.shape[2:])
            self.out_indices = np.reshape(self.out_indices, (-1,) + self.out_indices.shape[2:])
            if self.debug:
                print(self.in_indices.shape)
                print(self.in_indices)
                print(self.out_indices.shape)
                print(self.out_indices)
        else:
            raise NotImplementedError(
                "SlidingWindowLoader only supports 3D and 4D inputs, received x.shape=%s" % (
                    str(data["x"].shape)
                )
            )
        self.data = data
        self.mb = {}

    def __len__(self):
        return self.in_indices.shape[0]

    #  Description:
    #   Pull a single sample or batch of samples from x and y
    # Arguments:
    #   key - index or list of indices to pull sample from
    def __getitem__(self, index):
        if self.debug:
            print("index =", type(index), "= len(%d)" % (len(index)), "=")
            if self.debug > 1:
                print(index)
        if self.debug:
            for key, value in self.data.items():
                print(key, "=", type(value), "=", value.shape, "=")
                if self.debug > 1:
                    print(value)
        for key in self.__sampled__:
            value = self.data[key]
            indices = self.in_indices[index,:]
            if key in self.__outputs__:
                indices = self.out_indices[index,:]
            final_shape = indices.shape + value.shape[1:]
            self.mb[key] = torch.reshape(value[np.reshape(indices, (-1,))], final_shape)
        if self.debug:
            for key, value in self.mb.items():
                print(key, "=", type(value), "=", value.shape, "=")
                if self.debug > 1:
                    print(value)
        self.mb["__index__"] = index
        if self.debug:
            sys.exit(1)
        return self.mb


class EarlyStopper:

    def __init__(self, patience, init_steps=0):
        self.init_steps = init_steps
        self.patience = patience
        self.n_plateau_steps = init_steps
        self.min_loss = sys.float_info.max

    def step(self, loss):
        if loss < self.min_loss:
            self.reset(loss)
        else:
            self.n_plateau_steps += 1
        return self.stop()

    def stop(self):
        return self.patience > 0 and self.n_plateau_steps >= self.patience

    def reset(self, loss=None):
        if not loss is None:
            self.min_loss = loss
        self.n_plateau_steps = 0


class RapidScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, named_parameters, param_gauss_dict={}, last_epoch=-1, verbose=False, debug=1):
        if not isinstance(named_parameters, (tuple, list)):
            raise TypeError(
                "Input named_parameters must be tuple or list. Received %s." % (type(named_parameters))
            )
        pidx_gauss_dict = {}
        for pidx, (name, param) in enumerate(named_parameters):
            for regex_name, gauss in param_gauss_dict.items():
                if re.match(regex_name, name):
                    if debug:
                        print(regex_name, name, pidx, gauss)
                    pidx_gauss_dict[pidx] = gauss
        if debug:
            print(pidx_gauss_dict)
        #
        init_lrs = [param["lr"] for param in optimizer.param_groups]
        for pidx, gauss in pidx_gauss_dict.items():
            if len(gauss) < 3:
                pidx_gauss_dict[pidx] = [gauss[0], gauss[1], gauss[1]]
        self.pidx_gauss_dict = pidx_gauss_dict
        self.init_lrs = init_lrs
        super().__init__(optimizer, last_epoch, verbose)

    def gauss(self, x, u, s):
        return (1.0 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.power((x - u) / s, 2))

    def get_param_lr(self, x, pidx):
        gauss = self.pidx_gauss_dict.get(pidx, None)
        if gauss is None:
            lr = self.init_lrs[pidx]
        else:
            u, sl, sr = gauss
            if x < u:
                if sl is None:
                    p, pf = 1, 1
                else:
                    p = self.gauss(x, u, sl)
                    pf = self.gauss(0, 0, sl)
            else:
                if sr is None:
                    p, pf = 1, 1
                else:
                    p = self.gauss(x, u, sr)
                    pf = self.gauss(0, 0, sr)
            lr = self.init_lrs[pidx] * (p / pf)
        return lr

    def get_lr(self):
        lrs = list(self.init_lrs)
        for pidx, gauss in self.pidx_gauss_dict.items():
            lrs[pidx] = self.get_param_lr(self.last_epoch+1, pidx)
        return lrs

    def plot(self, epochs=200, ax=None):
        if ax is None:
            ax = pyplot.gca()
        for pidx, gauss in self.pidx_gauss_dict.items():
            ax.plot(range(epochs), [self.get_param_lr(i, pidx) for i in range(epochs)], label=pidx)
        return ax


class TriggerScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, named_parameters, param_state_dict={}, max_resets=2, ease_steps=1, last_epoch=-1, verbose=False, debug=0):
        if not isinstance(named_parameters, (tuple, list)):
            raise TypeError(
                "Input named_parameters must be tuple or list. Received %s." % (type(named_parameters))
            )
        pidx_states_dict = {}
        states = []
        for pidx, (name, param) in enumerate(named_parameters):
            for regex_name, state in param_state_dict.items():
                if re.match(regex_name, name):
                    if debug:
                        print(regex_name, name, pidx, state)
                    if not pidx in pidx_states_dict:
                        pidx_states_dict[pidx] = []
                    pidx_states_dict[pidx].append(state)
                    states.append(state)
        n_states = len(np.unique(states))
        if debug:
            print(pidx_states_dict)
        init_lrs = [param["lr"] for param in optimizer.param_groups]
        #
        self.optimizer = optimizer
        self.named_parameters = named_parameters
        self.param_state_dict = param_state_dict
        self.pidx_states_dict = pidx_states_dict
        self.init_lrs = init_lrs
        self.state = 0
        self.n_states = n_states
        self.reset_count = 0
        self.max_resets = max_resets
        self.ease_steps = ease_steps
        self._step_count = 0
        self.last_advance_step = 0
        print(self)
#        super().__init__(optimizer, last_epoch, verbose)

    def get_param_lr(self, state, pidx):
        lr = self.init_lrs[pidx] if state in self.pidx_states_dict[pidx] else 0.0
        if self.last_advance_step > 0:
            lr = min(lr, (self._step_count - self.last_advance_step + 1) * (lr / max(1, self.ease_steps)))
        return lr

    def get_lr(self):
        lrs = list(self.init_lrs)
        for pidx, states in self.pidx_states_dict.items():
            lrs[pidx] = self.get_param_lr(self.state, pidx)
#        print("TriggerScheduler LRs:")
#        print(lrs)
        return lrs

    def advance_state(self):
        self.state = 0 if self.state == self.n_states-1 else self.state+1
        self.reset_count += 0 if self.state else 1
        self.last_advance_step = self._step_count

    def step(self, event, epoch=None):
        self._step_count += 1
        if isinstance(event, EarlyStopper):
            if event.stop():
                self.advance_state()
                print(self)
                if self.reset_count <= self.max_resets:
                    event.reset()
        else:
            raise NotImplementedError("Unknown event type %s." % (type(event)))
        lrs = self.get_lr()
        for pidx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = lrs[pidx]
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def __str__(self):
        lrs = self.get_lr()
        tab = 4 * " "
        lines = []
        lines.append("TriggerScheduler[state=%d/%d](" % (self.state+1, self.n_states))
        for pidx, (name, param) in enumerate(self.named_parameters):
            lines.append("%s%s : %.8f" % (tab, name, lrs[pidx]))
        lines.append(")")
        return "\n".join(lines)
