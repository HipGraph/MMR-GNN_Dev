import os
import sys
import time
import torch
import numpy as np

import util
from container import Container
from model import Model

from model.ASTGCN_PyTorch.model.MSTGCN_r import make_model


class MSTGCN(Model):

    def __init__(
        self, 
        in_channels, # in_channels (int) - Number of input features
        num_of_vertices, # num_of_vertices - Number of nodes in graph |V|
        len_input, # len_input (int) - Number of input time-steps
        num_for_predict, # num_for_predict (int) - Number of output time-steps (maybe number of output features?)
        adj_mx, 
        nb_block, 
        K, 
        nb_chev_filter, 
        nb_time_filter, 
        time_strides, 
        device, 
    ):
        super(MSTGCN, self).__init__()
        # Init layers
        self.model = make_model(
            device, 
            nb_block, 
            in_channels, 
            K, 
            nb_chev_filter, 
            nb_time_filter, 
            time_strides, 
            adj_mx, 
            num_for_predict, 
            len_input, 
        )
        # Save all vars
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["nb_block", None], 
            ["K", None], 
            ["nb_chev_filter", None], 
            ["nb_time_filter", None], 
            ["time_strides", None], 
        ]

    def forward(self, inputs):
        self.debug = 0
        # Handle args
        x = inputs["x"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # x.shape=(N, T, |V|, F)
        # Start forward
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        x = torch.movedim(x, (0, 1, 2, 3), (0, 3, 1, 2)) # x.shape=(N, |V|, F, T)
        if self.debug:
            print("x reshaped =", x.shape, "=")
            if self.debug > 1:
                print(x)
        a = self.model(x) # a.shape=(N, |V|, T')
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        a = torch.unsqueeze(torch.transpose(a, 1, 2), -1) # a.shape=(N, T', |V|, 1)
        if self.debug:
            print("a reshaped =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        # Acquired from function make_model() @ model/MSTGCN_r.py
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else: # not given in make_model() but needed for reproducibility
                torch.nn.init.uniform_(p)

    def verify_pulled_data(self, data, var):
        if data["y"].shape[-1] > 1:
            raise ValueError(
                "MSTGCN can only infer a single response variable but \"y\" has shape=%s" % (data["y"].shape)
            )
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    model = MSTGCN(
        spatmp.misc.n_predictor, 
        spatmp.original.get("n_spatial", "train"), 
        var.mapping.temporal_mapping[0], 
        var.mapping.temporal_mapping[1], 
        graph.original.get("A", "train"), 
        hyp_var.nb_block, 
        hyp_var.K, 
        hyp_var.nb_chev_filter, 
        hyp_var.nb_time_filter, 
        hyp_var.time_strides, 
        util.get_device(var.execution.device), 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in respository @ https://github.com/guoshnBJTU/ASTGCN-r-pytorch
#   1. The paper @ https://ojs.aaai.org/index.php/AAAI/article/view/3881 under section "Experiments", sub-section "Settings"
#       ::: K, nb_chev_filter, nb_time_filter, loss, mbatch_size, lr
#   2. The configuration file configurations/METR_LA_astgcn.conf from repository @ https://github.com/guoshnBJTU/ASTGCN-r-pytorch
#       ::: nb_block, n_epochs, time_strides
class HyperparameterVariables(Container):

    def __init__(self):
        self.nb_block = 2
        self.K = 3
        self.nb_chev_filter = 64
        self.nb_time_filter = 64
        self.time_strides = 1 # num_of_hours in configurations/METR_LA_astgcn.conf


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 100
        self.mbatch_size = 64
        self.lr = 1e-4
        self.loss = "MSELoss"
