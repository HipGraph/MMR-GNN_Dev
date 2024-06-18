import os
import sys
import time
import torch
import numpy as np

import util
from container import Container
from model import Model

from model.STGCN_PyTorch.model.models import STGCNChebGraphConv
from model.STGCN_PyTorch.script.utility import calc_gso, calc_chebynet_gso


class STGCN(Model):

    def __init__(
        self, 
        n_his, 
        gso, 
        n_vertex, 
        Kt, 
        Ks, 
        act_func, 
        graph_conv_type, 
        enable_bias, 
        droprate, 
        blocks, 
    ):
        super(STGCN, self).__init__()
        # Handle args
        class Args:
            pass
        args = Args()
        args.n_his = n_his
        args.gso = gso
        args.Kt = Kt
        args.Ks = Ks
        args.act_func = act_func
        args.graph_conv_type = graph_conv_type
        args.enable_bias = enable_bias
        args.droprate = droprate
        # Init layers
        self.model = STGCNChebGraphConv(args, blocks, n_vertex)
        # Save all vars
        self.graph_conv_type = graph_conv_type
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["Kt", None], 
            ["Ks", None], 
            ["act_func", None], 
            ["graph_conv_type", None], 
            ["gso_type", None], 
            ["enable_bias", None], 
            ["droprate", None], 
            ["stblock_num", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # x.shape=(N, T, |V|, F)
        # Start forward
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        x = torch.movedim(x, (0, 1, 2, 3), (0, 2, 3, 1)) # x.shape=(N, F, T, |V|)
        if self.debug:
            print("x reshaped =", x.shape, "=")
            if self.debug > 1:
                print(x)
        a = self.model(x) # a.shape=(N, 1, T', |V|)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        a = torch.movedim(a, (0, 1, 2, 3), (0, 3, 1, 2)) # a.shape=(N, T', |V|, 1)
        if self.debug:
            print("a reshaped =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def verify_pulled_data(self, data, var):
        Ti, To = var.temporal_mapping
        if To > 1:
            raise ValueError("STGCN only capable of single-step forecasting but [Ti,To]=[%d,%d]." % (Ti, To))
        return data

    def reset_parameters(self):
        for block in self.model.st_blocks:
            block.tmp_conv1.align.align_conv.reset_parameters()
            block.tmp_conv1.causal_conv.reset_parameters()
            block.graph_conv.align.align_conv.reset_parameters()
            if self.graph_conv_type == "cheb_graph_conv":
                block.graph_conv.cheb_graph_conv.reset_parameters()
            elif self.graph_conv_type == "graph_conv":
                block.graph_conv.graph_conv.reset_parameters()
            else:
                raise ValueError(self.graph_conv_type)
            block.tmp_conv2.align.align_conv.reset_parameters()
            block.tmp_conv2.causal_conv.reset_parameters()
            block.tc2_ln.reset_parameters()
        if self.model.Ko > 1:
            self.model.output.tmp_conv1.align.align_conv.reset_parameters()
            self.model.output.tmp_conv1.causal_conv.reset_parameters()
            self.model.output.fc1.reset_parameters()
            self.model.output.fc2.reset_parameters()
            self.model.output.tc1_ln.reset_parameters()
        elif self.model.Ko == 0:
            self.model.fc1.reset_parameters()
            self.model.fc2.reset_parameters()


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    # Acquired from main.py in repository @ https://github.com/hazdzz/STGCN
    gso = calc_gso(graph.original.get("A", "train"), hyp_var.gso_type)
    if hyp_var.graph_conv_type == 'cheb_graph_conv':
        gso = calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    gso = torch.from_numpy(gso).to(util.get_device(var.execution.device))
    # Acquired from main.py in repository @ https://github.com/hazdzz/STGCN
    #    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    Ti, To = var.mapping.temporal_mapping
    if Ti < 6:
        raise ValueError("STGCN: found input time-steps (Ti=%d) are less than 6 but Ti>=6 must hold for default settings." % (Ti))
    Ko = Ti - (hyp_var.Kt - 1) * 2 * hyp_var.stblock_num
    blocks = []
    blocks.append([spatmp.misc.n_predictor])
    for l in range(hyp_var.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    # Determined from data_transform() of script/dataloader.py in repository @ https://github.com/hazdzz/STGCN
    # Init model
    model = STGCN(
        Ti, 
        gso, 
        spatmp.original.get("n_spatial", "train"), 
        hyp_var.Kt, 
        hyp_var.Ks, 
        hyp_var.act_func, 
        hyp_var.graph_conv_type, 
        hyp_var.enable_bias, 
        hyp_var.droprate, 
        blocks, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/1709.04875.pdf under section 4.3 "Experimental Settings"
#       ::: Kt, Ks, optimizer, n_epochs, mbatch_size, lr, lr_scheduler, lr_scheduler_kwargs
#   2. The Python script main.py in repository @ https://github.com/hazdzz/STGCN
#       ::: act_func, graph_conv_type, gso_type, enable_bias, droprate, stblock_num
class HyperparameterVariables(Container):

    def __init__(self):
        self.Kt = 3
        self.Ks = 3
        self.act_func = "glu"
        self.graph_conv_type = "cheb_graph_conv"
        self.gso_type = "sym_norm_lap"
        self.enable_bias = True
        self.droprate = 0.5
        self.stblock_num = 2


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 50
        self.mbatch_size = 50
        self.lr = 1e-3
        self.lr_scheduler = "StepLR"
        self.lr_scheduler_kwargs = {"step_size": 5, "gamma": 0.7}
        self.optimizer = "RMSprop"
