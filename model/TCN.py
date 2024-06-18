import os
import sys
import torch
import math

import util
from container import Container
from model import Model


from model.TCN_PyTorch.TCN.adding_problem.model import TCN as _TCN


class TCN(Model):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        # Init layers
        self.model = _TCN(input_size, output_size, num_channels, kernel_size, dropout)
        # Save all vars
        self.input_size = input_size
        self.output_size = output_size
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["num_channels", None], 
            ["kernel_size", None], 
            ["dropout", None],
        ]

    def forward(self, inputs):
#        self.debug = 1
        x = inputs["x"]
        n_temporal_out = inputs["n_temporal_out"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, F)
        # Start forward
        if self.debug:
            print(util.make_msg_block("TCN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
        x = torch.movedim(x, (0, 1, 2, 3), (0, 3, 1, 2)) # shape=(N, |V|, F, T)
        x = torch.reshape(x, (-1, n_predictor, n_temporal_in)) # shape=(N*|V|, F, T)
        if self.debug:
            print("x reshaped =", x.shape)
            print(util.memory_of(x))
        if n_temporal_out > 1: # Apply forward auto-regressively
            a = x
            for i in range(n_temporal_out):
                a_i = self.model(a[:,:,-n_temporal_in:]) # shape=(N*|V|, F')
                a = torch.concat((a, torch.unsqueeze(a_i, -1)), -1)
            a = a[:,:,-n_temporal_out:] # shape=(N*|V|, F', T')
        else:
            a = self.model(x) # shape=(N*|V|, F')
            a = torch.unsqueeze(a, -1) # shape=(N*|V|, F', T')
        if self.debug:
            print("TCN Output =", a.shape)
            print(util.memory_of(a))
        a = torch.transpose(a, -2, -1) # shape=(N*|V|, T', F')
        a = torch.reshape(a, (n_sample, n_spatial, n_temporal_out, self.output_size)) # shape=(N, |V|, T', F')
        a = torch.transpose(a, 1, 2) # shape=(N, T', |V|, F')
        if self.debug:
            print("TCN Output Reshaped =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        # NOT REPRODUCIBLE DUE TO torch.nn.utils.weight_norm()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for layer in self.model.tcn.network:
            layer.init_weights()
        self.model.init_weights()


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    hyp_var = var.models.get(model_name()).hyperparameters
    if isinstance(hyp_var.num_channels, int):
        n_levels = int(math.ceil(math.log(var.mapping.temporal_mapping[0], 2)))
        hyp_var.num_channels = [hyp_var.num_channels for i in range(n_levels)]
    elif not isinstance(hyp_var.num_channels, list):
        raise TypeError(hyp_var.num_channels)
    model = TCN(
        spatmp.misc.n_predictor,
        spatmp.misc.n_response,
        hyp_var.num_channels, 
        hyp_var.kernel_size,
        hyp_var.dropout,
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.num_channels = 16
        self.kernel_size = 2
        self.dropout = 0.0


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 10
        self.mbatch_size = 32
        self.lr = 4e-3
        self.optimizer = "Adam"
        self.initializer = None
