import os
import sys
import time
import torch

import util
from container import Container
from model import Model

from model.AGCRN_PyTorch.model.AGCRN import AGCRN as _AGCRN


class AGCRN(Model):

    def __init__(
        self, 
        input_dim, # input_dim (int) - Number of input features
        output_dim, # output_dim (int) : Number of output features
        num_nodes,  # num_nodes (int) - Number of nodes
        horizon, # horizon (int) - Number of output time-steps
        default_graph=True, # default_graph (bool) - ?
        num_layers=2, # num_layers (int) - Number of layers
        embed_dim=10, # embed_dim (int) - Number of dimensions for node embeddings
        rnn_units=64, # rnn_units (int) - Number of dimensions for rnn embeddings
        cheb_k=2, # cheb_k (int) : ?
    ):
        super(AGCRN, self).__init__()
        # Handle args
        class Args:
            pass
        args = Args()
        args.input_dim = input_dim
        args.output_dim = output_dim
        args.num_nodes = num_nodes
        args.horizon = horizon
        args.default_graph = default_graph
        args.num_layers = num_layers
        args.embed_dim = embed_dim
        args.rnn_units = rnn_units
        args.cheb_k = cheb_k
        # Init layers
        self.model = _AGCRN(args)
        # Save all vars
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["num_layers", None], 
            ["embed_dim", None], 
            ["rnn_units", None], 
            ["cheb_k", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        y = inputs["y"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, F)
        # Start forward
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        a = self.model(x, None, 0.0) # shape=(N, T', |V|, F')
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        # Acquired from repository @ AGCRN/model/Run.py
        print("HELLO?")
        input()
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    model = AGCRN(
        spatmp.misc.n_predictor, 
        spatmp.misc.n_response, 
        spatmp.original.get("n_spatial", "train"), 
        var.mapping.temporal_mapping[1], 
        hyp_var.default_graph, 
        hyp_var.num_layers, 
        hyp_var.embed_dim, 
        hyp_var.rnn_units, 
        hyp_var.cheb_k, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/2007.02842.pdf in APPENDIX A.3 "Implementation Details"
#       ::: num_layers, rnn_units, mbatch_size, lr, embed_dim, optimizer, n_epochs, patience
#   2. Repository @ AGCRN/model/PEMSD4_AGCRN.conf
#       ::: cheb_order, grad_clip, 
class HyperparameterVariables(Container):

    def __init__(self):
        self.default_graph = True
        self.num_layers = 2
        self.embed_dim = 2 # PeMSD8
        self.embed_dim = 10 # PeMSD4
        self.rnn_units = 64
        self.cheb_k = 2


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 100
        self.patience = 15
        self.mbatch_size = 64
        self.lr = 3e-3
        self.grad_clip = None # b/c grad_norm=False in PEMSD4_AGCRN.conf
        self.optimizer = "Adam"
        self.loss = "L1Loss"
        self.initializer = None
