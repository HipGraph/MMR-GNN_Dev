import os
import sys
import time
import torch

import util
from container import Container
from model import Model

from model.StemGNN_PyTorch.models.base_model import Model as _StemGNN


class StemGNN(Model):

    def __init__(
        self, 
        units, # units (int) - Number of nodes in the graph
        time_step, # time_step (int) - Number of input time-steps or input window size
        horizon, # horizon (int) - Number of output time-steps or output window size
        stack_cnt=2, #
        multi_layer=5, # multi_layer (int) -
        dropout_rate=0.5, #
        leaky_rate=0.2, #
    ):
        super(StemGNN, self).__init__()
        self.model = _StemGNN(units, stack_cnt, time_step, multi_layer, horizon, dropout_rate, leaky_rate)
        # Save all vars
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["stack_cnt", None], 
            ["multi_layer", None], 
            ["dropout_rate", None], 
            ["leaky_rate", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape
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
        a, attention = self.model(x) # shape=(N, T', |V|)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            print("attention =", attention.shape, "=")
            if self.debug > 1:
                print(attention)
        a = torch.unsqueeze(a, -1)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        # Acquired from classes StockBlockLayer and Model @ ./models/base_model.py
        for block in self.model.stock_block:
            torch.nn.init.xavier_normal_(block.weight)
        torch.nn.init.xavier_uniform_(self.model.weight_key.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.model.weight_query.data, gain=1.414)

    def verify_pulled_data(self, data, var):
        if data["x"].shape[-1] > 1 or data["y"].shape[-1] > 1:
            raise ValueError(
                "StemGNN only capable of univariate inference but \"x\" or \"y\" were multi-variate with x.shape=%s and y.shape=%s" % (
                    str(data["x"].shape), str(data["y"].shape)
                )
            )
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    model = StemGNN(
        spatmp.original.get("n_spatial", "train"), 
        var.mapping.temporal_mapping[0], 
        var.mapping.temporal_mapping[1], 
        hyp_var.stack_cnt, 
        hyp_var.multi_layer, 
        hyp_var.dropout_rate, 
        hyp_var.leaky_rate, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/2103.07719.pdf under section 5 "Experiments", sub-section 5.1 "Setup"
#       ::: optimizer, n_epochs, lr, lr_scheduler, lr_scheduler_kwargs
#   2. The script main.py in repository @ https://github.com/microsoft/StemGNN
#       ::: multi_layer, dropout_rate, leaky_rate, mbatch_size
#   3. The script models/handler.py in repository @ https://github.com/microsoft/StemGNN
#       ::: stack_cnt, loss
class HyperparameterVariables(Container):

    def __init__(self):
        self.stack_cnt = 2
        self.multi_layer = 5
        self.dropout_rate = 0.5
        self.leaky_rate = 0.2


class TrainingVariables(Container):

    def __init__(self):
        self.n_epoch = 50
        self.mbatch_size = 32
        self.lr = 1e-3
        self.lr_scheduler = "StepLR"
        self.lr_scheduler_kwargs = {"step_size": 5, "gamma": 0.7}
        self.optimizer = "RMSprop"
        self.loss = "MSELoss"
        self.initializer = None
