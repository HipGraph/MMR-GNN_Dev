import os
import sys
import time
import torch
import numpy as np
import datetime as dt

import util
from container import Container
from model import Model

from torch_geometric_temporal.nn.attention.gman import GMAN as _GMAN


class GMAN(Model):

    def __init__(
        self, 
        num_his, # num_his (int) - Number of history steps (P from paper).
        steps_per_day, # steps_per_day (int) - Number of steps in a day (T from paper).
        d, # d (int) - Dimension of each attention head outputs (dim of spatial features/embeddings).
        L, # L (int) - Number of STAtt blocks in the encoder/decoder.
        K, # K (int) - Number of attention heads.
        bn_decay, # bn_decay (float) - Batch normalization momentum.
        use_bias, # use_bias (bool) - Whether to use bias in Fully Connected layers.
        mask, # mask (bool) - Whether to mask attention score in temporal attention.
    ):
        super(GMAN, self).__init__()
        self.model = _GMAN(L, K, d, num_his, bn_decay, steps_per_day, use_bias, mask)
        # Save all vars
        self.num_his = num_his
        self.steps_per_day = steps_per_day
        self.d = d
        self.L = L
        self.K = K
        self.bn_decay = bn_decay
        self.use_bias = use_bias
        self.mask = mask
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["L", None], 
            ["K", None], 
            ["d", None], 
            ["bn_decay", None], 
            ["use_bias", None], 
            ["mask", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        x_s = inputs["x_s"]
        x_t, y_t = inputs["x_t"], inputs["y_t"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, 1)
        # Start forward
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        if self.debug:
            print("x_s =", x_s.shape, "=")
            if self.debug > 1:
                print(x_s)
        if self.debug:
            print("x_t =", x_t.shape, "=")
            if self.debug > 1:
                print(x_t)
        if self.debug:
            print("y_t =", y_t.shape, "=")
            if self.debug > 1:
                print(y_t)
        x = torch.squeeze(x, -1) # shape=(N, T, |V|)
        if self.debug:
            print("x reshaped =", x.shape, "=")
            if self.debug > 1:
                print(x)
        a = self.model(x, x_s, torch.cat((x_t, y_t), -2)) # shape=(N, T', |V|)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        a = torch.unsqueeze(a, -1) # shape=(N, T', |V|, 1)
        if self.debug:
            print("a reshaped =", a.shape, "=")
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
        # Acquired from class Conv2D in script @ https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/gman.html#GMAN
        for name, param in self.model.named_parameters():
            if "_conv2d.weight" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "_conv2d.bias" in name:
                torch.nn.init.zeros_(param)

    def pull_model_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = ["x", "y"]
        data["__outputs__"] = ["y"]
        # Pull spatiotemporal data
        data["x"] = dataset.spatiotemporal.transformed.original.get("predictor_features", partition)
        data["y"] = dataset.spatiotemporal.transformed.original.get("response_features", partition)
        # Pull spatial data
        if dataset.spatial.is_empty():
            data["x_s"] = np.zeros((data["x"].shape[-2], self.K * self.d))
        else:
            try:
                x_s = dataset.spatial.transformed.original.get("predictor_features", partition)
            except:
                x_s = dataset.spatial.transformed.original.get("predictor_features", partition)
            data["x_s"] = np.tile(x_s, self.K)
        # Pull temporal data
        sod_indices = util.temporal_labels_to_periodic_indices( # sod -> step-of-day
            dataset.spatiotemporal.original.get("temporal_labels", partition), 
            [1, "days"], 
            dataset.spatiotemporal.misc.temporal_resolution, 
            dataset.spatiotemporal.misc.temporal_label_format, 
        )
        dow_indices = util.temporal_labels_to_periodic_indices( # dow -> day-of-week
            dataset.spatiotemporal.original.get("temporal_labels", partition), 
            [1, "weeks"], 
            [1, "days"], 
            dataset.spatiotemporal.misc.temporal_label_format, 
        )
        data["x_t"] = np.concatenate((dow_indices[:,None], sod_indices[:,None]), -1)
        data["y_t"] = np.concatenate((dow_indices[:,None], sod_indices[:,None]), -1)
        data["__sampled__"] += ["x_t", "y_t"]
        data["__outputs__"].append("y_t")
        # Pull misc data
        data["n_temporal_out"] = var.temporal_mapping[1]
        return data

    def verify_pulled_data(self, data, var):
        if data["x"].shape[-1] > 1 or data["y"].shape[-1] > 1:
            raise ValueError(
                "GMAN only capable of univariate inference but \"x\" or \"y\" were multi-variate with x.shape=%s and y.shape=%s" % (str(data["x"].shape), str(data["y"].shape))
        )
        return data


def init(dataset, var):
    spa = dataset.spatial
    spatmp = dataset.spatiotemporal
    hyp_var = var.models.get(model_name()).hyperparameters
    model = GMAN(
        var.mapping.temporal_mapping[0], 
        dt.timedelta(days=1) // util.resolution_to_delta(spatmp.misc.temporal_resolution), 
        1 if spa.is_empty() else spa.misc.n_predictor, 
        hyp_var.L,
        hyp_var.K,
        hyp_var.bn_decay,
        hyp_var.use_bias,
        hyp_var.mask,
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/1911.08415.pdf under section "Experimental Settings", sub-section "Hyperparameters"
#       ::: optimizer, lr, L, K, d
#   2. The script METR/train.py from repository @ https://github.com/zhengchuanpan/GMAN
#       ::: n_epochs, patience, mbatch_size, lr_scheduler, lr_scheduler_kwargs, bn_decay
#   3. The documentation of BatchNorm2d @ https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
#       ::: bn_decay
#   4. Infered from common defaults in PyTorch, etc
#       ::: use_bias, mask
class HyperparameterVariables(Container):

    def __init__(self):
        self.L = 3
        self.K = 8
        self.d = 8
        self.bn_decay = 0.1
        self.use_bias = True
        self.mask = False


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 1000
        self.patience = 10
        self.mbatch_size = 16
        self.lr = 1e-3
        self.lr_scheduler = "StepLR"
        self.lr_scheduler_kwargs = {"step_size": 5, "gamma": 0.7}
        self.optimizer = "Adam"
        self.initializer = None
