import os
import sys
import time
import torch
import importlib

import util
from container import Container
from model import Model

from model.DCRNN_PyTorch.model.pytorch.dcrnn_model import DCRNNModel
from model.DCRNN_PyTorch.model.pytorch.dcrnn_cell import LayerParams


class DCRNN(Model):

    def __init__(
        self, 
        input_dim, # input_dim (int) - Number of input features.
        output_dim, # output_dim (int) - Number of output features.
        seq_len, # seq_len (int) - Number of input time-steps
        horizon, # horizon (int) - Number of output time-steps or forecast length
        adj_mx, 
        cl_decay_steps=2000, 
        filter_type="dual_random_walk", 
        l1_decay=0.0, 
        max_diffusion_step=3, 
        num_rnn_layers=2, 
        rnn_units=64, 
        use_curriculum_learning=True, 
    ):
        super(DCRNN, self).__init__()
        if not isinstance(adj_mx, torch.Tensor): # must be a FloatTensor or type errors will follow
            adj_mx = util.to_tensor(adj_mx, torch.float)
        class FakeLogger:
            def __init__(self): pass
            def debug(self, msg): pass
            def info(self, msg): pass
        kwargs = {
            "input_dim": input_dim, 
            "output_dim": output_dim, 
            "seq_len": seq_len, 
            "horizon": horizon, 
            "num_nodes": adj_mx.shape[0], 
            "cl_decay_steps": cl_decay_steps, 
            "filter_type": filter_type, 
            "l1_decay": l1_decay, 
            "max_diffusion_step": max_diffusion_step, 
            "num_rnn_layers": num_rnn_layers, 
            "rnn_units": rnn_units, 
            "use_curriculum_learning": use_curriculum_learning, 
        }
        self.model = DCRNNModel(adj_mx, FakeLogger(), **kwargs)
        # Save all vars
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.batches_seen = 0
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["cl_decay_steps", None], 
            ["filter_type", None],
            ["l1_decay", None], 
            ["max_diffusion_step", None], 
            ["num_rnn_layers", None], 
            ["rnn_units", None], 
            ["use_curriculum_learning", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        y = inputs["y"]
        adj = inputs["adj"]
        n_sample, seq_len, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, F)
        # Start forward
        if self.debug:
            print(util.make_msg_block("DCRNN Foward"))
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        if self.debug:
            print("y =", y.shape, "=")
            if self.debug > 1:
                print(y)
        if self.debug:
            print("adj =", adj.shape, "=")
            if self.debug > 1:
                print(adj)
        self.model.encoder_model.adj_mx = adj
        self.model.decoder_model.adj_mx = adj
        x = torch.reshape(torch.transpose(x, 0, 1), (seq_len, n_sample, -1)) # shape=(T, N, |V|*F)
        y = torch.reshape(torch.transpose(y, 0, 1), (self.horizon, n_sample, -1)) # shape=(T', N, |V|*F')
        if self.debug:
            print("x reshaped =", x.shape, "=")
            if self.debug > 1:
                print(x)
        if self.debug:
            print("y reshaped =", y.shape, "=")
            if self.debug > 1:
                print(y)
        a = self.model.forward(x, y, self.batches_seen) # shape=(T', N, |V|*F')
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        a = torch.reshape(
            torch.transpose(a, 0, 1), (n_sample, self.horizon, n_spatial, self.output_dim)
        ) # shape=(N, T', |V|, F')
        if self.debug:
            print("a reshaped =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            sys.exit(1)
        if self.training:
            self.batches_seen += 1
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        # Acquired from class LayerParams @ model/pytorch/dcrnn_cell.py
        for cell in self.model.encoder_model.dcgru_layers:
            cell._fc_params = LayerParams(cell, "fc")
            cell._gconv_params = LayerParams(cell, "gconv")
        for cell in self.model.decoder_model.dcgru_layers:
            cell._fc_params = LayerParams(cell, "fc")
            cell._gconv_params = LayerParams(cell, "gconv")

    def pull_model_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = ["x", "y"]
        data["__outputs__"] = ["y"]
        # Pull spatiotemporal data
        data["x"] = dataset.spatiotemporal.transformed.original.get("predictor_features", partition)
        data["y"] = dataset.spatiotemporal.transformed.original.get("response_features", partition)
        # Pull graph data
        if not dataset.graph.is_empty():
            if var.graph.edge_weight_feature is None:
                data["adj"] = dataset.graph.original.get("A", partition)
            elif var.graph.edge_weight_feature == "weight":
                data["adj"] = dataset.graph.original.get("W", partition)
            else:
                raise ValueError(
                    "DCRNN can only accept adjacency matrix A or weight matrix W denoted by edge_weight_feature=%s or edge_weight_feature=%s respectively" % (str(None), "weight")
                )
        return data

    def log_compgraph(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        pass


def init(dataset, var):
    spatiotemporal = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    model = DCRNN(
        spatiotemporal.misc.n_predictor,
        spatiotemporal.misc.n_response,
        var.mapping.temporal_mapping[0],
        var.mapping.temporal_mapping[1],
        graph.original.get("A", "train"), 
        hyp_var.cl_decay_steps, 
        hyp_var.filter_type, 
        hyp_var.l1_decay, 
        hyp_var.max_diffusion_step, 
        hyp_var.num_rnn_layers, 
        hyp_var.rnn_units, 
        hyp_var.use_curriculum_learning, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/1707.01926.pdf : section 4 - "EXPERIMENTS"
#       ::: optimizer
#   2. The paper @ https://arxiv.org/pdf/1707.01926.pdf : APPENDIX E - "DETAILED EXPERIMENTAL SETTINGS"
#       ::: max_diffusion_step, num_rnn_layers, rnn_units, lr_scheduler, lr_scheduler_kwargs
#   3. The configuration file data/model/dcrnn_la.yaml in repository @ https://github.com/chnsh/DCRNN_PyTorch
#       ::: mbatch_size, filter_type, use_curriculum_learning 
class HyperparameterVariables(Container):

    def __init__(self):
        self.cl_decay_steps = 2000
        self.filter_type = "dual_random_walk"
        self.l1_decay = 0.0
        self.max_diffusion_step = 3
        self.num_rnn_layers = 2
        self.rnn_units = 64
        self.use_curriculum_learning = True


class TrainingVariables(Container):

    def __init__(self):
        self.mbatch_size = 64
        self.lr = 1e-2
        self.lr_scheduler = "MultiStepLR"
        self.lr_scheduler_kwargs = {"milestones": [20, 30, 40, 50], "gamma": 0.1}
        self.grad_clip = "norm"
        self.grad_clip_kwargs = {"max_norm": 5}
        self.optimizer = "Adam"
