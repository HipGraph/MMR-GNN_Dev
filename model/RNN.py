import os
import sys
import torch

import util
from container import Container
from model import Model, TemporalMapper
from model import RNNCell
from model import RNN_HyperparameterVariables
from model import TemporalMapper_HyperparameterVariables


class RNN(Model):

    debug = 0

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=16,
        rnn_kwargs={},
        mapper_kwargs={}, 
        project_input=False, 
    ):
        super(RNN, self).__init__()
        # Instantiate model layers
        if project_input:
            self.in_proj = torch.nn.Linear(in_size, hidden_size)
            self.enc = RNNCell(hidden_size, hidden_size, **rnn_kwargs)
        else:
            self.enc = RNNCell(in_size, hidden_size, **rnn_kwargs)
        self.map = TemporalMapper(hidden_size, hidden_size, **mapper_kwargs)
        self.dec = RNNCell(hidden_size, hidden_size, **rnn_kwargs)
        self.out_proj = torch.nn.Linear(hidden_size, out_size)
        self.out_proj_act = torch.nn.Identity()
        # Save all vars
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.project_input = project_input
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["hidden_size", None],
            ["rnn_kwargs", None],
            ["mapper_kwargs", None],
            ["project_input", None], 
        ]

    def forward(self, inputs):
        self.enc.debug = self.debug
        self.map.debug = self.debug
        self.dec.debug = self.debug
        x, n_temporal_out = inputs["x"], inputs["n_temporal_out"]
        n_samples, n_temporal_in, n_spatial, in_size = x.shape
        if self.debug:
            print(util.make_msg_block("RNN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
            print("n_temporal_out =", n_temporal_out)
            print(util.memory_of(n_temporal_out))
        # Start forward
        a = x
        #    Input projection forward
        if self.project_input:
            a = self.in_proj(a)
            if self.debug:
                print("x Projected =", a.shape)
                print(util.memory_of(a))
        #    Encoding layer forward
        a = torch.reshape(torch.transpose(a, 1, 2), (-1, n_temporal_in, a.shape[-1]))
        if self.debug:
            print("x Reshaped =", a.shape)
            print(util.memory_of(a))
        a = self.enc(x=a)["yhat"]
        if self.debug:
            print("RNN Encoding =", a.shape)
            print(util.memory_of(a))
        #    Temporal mapper forward
        a = self.map(x=a, n_temporal_out=n_temporal_out, temporal_dim=-2)["yhat"]
        if self.debug:
            print("Encoding Remapped =", a.shape)
            print(util.memory_of(a))
        #    Decoding layer forward
        a = self.dec(x=a, n_steps=n_temporal_out)["yhat"]
        if self.debug:
            print("RNN Decoding =", a.shape)
            print(util.memory_of(a))
        a = torch.reshape(a, (n_samples, n_spatial, n_temporal_out, self.hidden_size))
        a = torch.transpose(a, 1, 2)
        if self.debug:
            print("Decoding Reshaped =", a.shape)
            print(util.memory_of(a))
        #    Output layer forward
        z = self.out_proj(a)
        a = self.out_proj_act(z)
        if self.debug:
            print("Output Projected =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.map.reset_parameters()
        self.dec.reset_parameters()
        self.out_proj.reset_parameters()


def init(dataset, var):
    spatiotemporal = dataset.spatiotemporal
    hyp_var = var.models.get(model_name()).hyperparameters
    model = RNN(
        spatiotemporal.misc.n_predictor,
        spatiotemporal.misc.n_response,
        hyp_var.hidden_size,
        hyp_var.rnn_kwargs.to_dict(), 
        hyp_var.mapper_kwargs.to_dict(), 
        hyp_var.project_input, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.hidden_size = 16
        self.project_input = False
        self.rnn_kwargs = RNN_HyperparameterVariables()
        self.rnn_kwargs.rnn_layer = "RNN"
        self.mapper_kwargs = TemporalMapper_HyperparameterVariables()


class TrainingVariables(Container):

    def __init__(self):
        self.initializer = None
