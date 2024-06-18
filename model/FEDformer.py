import os
import sys
import torch
import numpy as np
import pandas as pd

import util
from container import Container
from model import Model

from model.FEDformer_PyTorch.models.FEDformer import Model as _Model
from model.FEDformer_PyTorch.utils.timefeatures import time_features


class FEDformer(Model):

    debug = 0

    def __init__(self, cfg):
        super(FEDformer, self).__init__()
        print(cfg)
        # Instantiate model layers
        self.model = _Model(cfg)
        # Save all vars
        self.cfg = cfg
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["version", None], 
            ["mode_select", None], 
            ["modes", None], 
            ["d_model", None], 
            ["n_heads", None], 
            ["e_layers", None], 
            ["d_layers", None], 
            ["d_ff", None], 
            ["moving_avg", None], 
            ["factor", None], 
            ["distill", None], 
            ["dropout", None], 
            ["embed", None], 
            ["activation", None], 
            ["freq", None], 
        ]

    def forward(self, inputs):
        x_enc = inputs["x_enc"]
        x_dec = inputs["x_dec"]
        x_mark_enc = inputs["x_mark_enc"]
        x_mark_dec = inputs["x_mark_dec"]
        n_temporal_out = inputs["n_temporal_out"]
        B, T, N, C = x_enc.shape
        #
        x_enc = torch.squeeze(x_enc, -1)
        x_dec = torch.squeeze(x_dec, -1)
        if self.debug:
            print(x_enc.shape)
            print(x_mark_enc.shape)
            print(x_dec.shape)
            print(x_mark_dec.shape)
        if 0:
            x_dec = torch.cat((x_dec, x_enc[:,-self.cfg.label_len:,:]), 1)
            x_mark_dec = torch.cat((x_mark_dec, x_mark_enc[:,-self.cfg.label_len:]), 1)
        else: # correct way according to: https://github.com/MAZiqing/FEDformer/blob/master/exp/exp_main.py lines 121-123
            x_dec = torch.cat(
                (x_enc[:,-self.cfg.label_len:,:], torch.zeros_like(x_dec[:,-self.cfg.label_len:,:])), 
                1
            )
            x_mark_dec = torch.cat((x_mark_enc[:,-self.cfg.label_len:], x_mark_dec), 1)
        if self.debug:
            print(x_dec.shape)
            print(x_mark_dec.shape)
        a = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)[:,:,:,None]
        if self.debug:
            print("FEDformer Output =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def pull_model_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = ["x_enc", "x_dec", "y", "x_mark_enc", "x_mark_dec"]
        data["__inputs__"] = ["x_enc", "x_mark_enc"]
        data["__outputs__"] = ["y", "x_dec", "x_mark_dec"]
        spatmp = dataset.spatiotemporal
        # Pull spatiotemporal data
        data["x"] = spatmp.transformed.original.get("predictor_features", partition)
        data["x_enc"] = spatmp.transformed.original.get("predictor_features", partition)
        data["x_dec"] = spatmp.transformed.original.get("predictor_features", partition)
        data["y"] = spatmp.transformed.original.get("response_features", partition)
        # Pull temporal data
        data["x_mark_enc"] = time_features(
            pd.to_datetime(
                spatmp.original.get("temporal_labels", partition), 
                format=spatmp.misc.temporal_label_format
            ), 
            freq=self.cfg.freq
        ).T
        data["x_mark_dec"] = time_features(
            pd.to_datetime(
                spatmp.original.get("temporal_labels", partition), 
                format=spatmp.misc.temporal_label_format
            ), 
            freq=self.cfg.freq
        ).T
        # Pull etc
        data["n_temporal_out"] = var.temporal_mapping[1]
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    #
    cfg = Container().copy(hyp_var)
    cfg.enc_in = spatmp.original.train__n_spatial
    cfg.dec_in = spatmp.original.train__n_spatial
    cfg.c_out = spatmp.original.train__n_spatial
    cfg.seq_len = var.mapping.temporal_mapping[0]
    cfg.label_len = var.mapping.temporal_mapping[0] // 2
    cfg.pred_len = var.mapping.temporal_mapping[1]
    cfg.output_attention = False
    model = FEDformer(cfg)
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.version = "Fourier"
        self.mode_select = "random"
        self.modes = 64
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = [24]
        self.factor = 1
        self.distill = True
        self.dropout = 0.05
        self.embed = "timeF"
        self.activation = "gelu"
        self.freq = "h"


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 10
        self.patience = 3
        self.mbtach_size = 32
        self.lr = 2e-4 # is actually 1e-4 but doubling it to counteract lr decay at epoch=1 which the authors manually avoid
        self.loss = "MSELoss"
        self.optimizer = "Adam"
        self.lr_scheduler = "ExponentialLR"
        self.lr_scheduler_kwargs = {"gamma": 0.5}
