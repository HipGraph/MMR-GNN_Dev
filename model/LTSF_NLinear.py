import os
import sys
import torch

import util
from container import Container
from model import Model

from model.LTSF_PyTorch.models.NLinear import Model as _Model


class LTSF_NLinear(Model):

    debug = 0

    def __init__(self, cfg):
        super(LTSF_NLinear, self).__init__()
        print(cfg)
        # Instantiate model layers
        self.model = _Model(cfg)
        # Save all vars
        self.cfg = cfg
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["individual", None],
        ]

    def forward(self, inputs):
        x = inputs["x"]
        #
        x = torch.squeeze(x, -1)
        a = self.model(x)[:,:,:,None]
        if self.debug:
            print("LTSF-NLinear Output =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    #
    cfg = Container().copy(hyp_var)
    cfg.enc_in = spatmp.original.train__n_spatial
    cfg.seq_len  = var.mapping.temporal_mapping[0]
    cfg.pred_len = var.mapping.temporal_mapping[1]
    model = LTSF_NLinear(cfg)
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.individual = True


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 10
        self.patience = 3
        self.mbtach_size = 16
        self.lr = 0.01 # is actually 0.005 but doubling it to counteract lr decay at epoch=1 which the authors manually avoid
        self.loss = "MSELoss"
        self.optimizer = "Adam"
        self.lr_scheduler = "ExponentialLR"
        self.lr_scheduler_kwargs = {"gamma": 0.5}
