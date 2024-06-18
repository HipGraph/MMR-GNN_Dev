import os
import sys
import time
import torch

import util
from container import Container
from model import Model

from model.TGCN_PyTorch.models.tgcn import TGCN


class _TGCN(Model):

    def __init__(
        self, 
        adj, 
        n_temporal_out, 
        hidden_dim, # hidden_dim (int): Number of dimensions for hidden state of GRU
    ):
        super(_TGCN, self).__init__()
        self.model = TGCN(adj, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, n_temporal_out)
        self.out_proj_act = torch.nn.Identity()
        # Save all vars
        self.hidden_dim = hidden_dim
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["hidden_dim", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, 1)
        # Start forward
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        x = torch.squeeze(x, -1) # shape=(N, T, |V|)
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        a = self.model(x) # shape=(N, |V|, H)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        z = self.out_proj(a)
        a = self.out_proj_act(z) # shape=(N, |V|, T')
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        a = torch.unsqueeze(torch.transpose(a, 1, 2), -1) # shape=(N, T', |V|, 1)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        self.model.tgcn_cell.graph_conv1.reset_parameters()
        self.model.tgcn_cell.graph_conv2.reset_parameters()
        self.out_proj.reset_parameters()

    def verify_pulled_data(self, data, var):
        if data["x"].shape[-1] > 1 or data["y"].shape[-1] > 1:
            raise ValueError(
                "_TGCN only capable of univariate inference but \"x\" or \"y\" were multi-variate with x.shape=%s and y.shape=%s" % (
                    str(data["x"].shape), str(data["y"].shape)
                )
            )
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    model = _TGCN(
        graph.original.get("A", "train"), 
        var.mapping.temporal_mapping[1], 
        hyp_var.hidden_dim, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/1811.05320.pdf from section 4.3 - "Model Parameters Designing"
#       ::: hidden_dim, lr, n_epoch, optimizer, mbatch_size, loss
#   2. The repository @ https://github.com/martinwhl/T-GCN-PyTorch from "Model Training"
#       ::: l2_reg
class HyperparameterVariables(Container):

    def __init__(self):
        self.hidden_dim = 100 # for SZ-Taxi
        self.hidden_dim = 64 # for Los-Loop


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 3000
        self.n_epochs = 50
        self.mbatch_size = 64
        self.lr = 1e-3
        self.l2_reg = 1.5e-3 # SZ-Taxi
        self.l2_reg = 0.0 # Los-Loop
        self.optimizer = "Adam"
        self.loss = "MSELoss"
