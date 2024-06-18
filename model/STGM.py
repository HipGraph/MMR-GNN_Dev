import os
import sys
import torch
import warnings
import numpy as np

import util
from container import Container
from model import Model

from model.STGM_PyTorch.src.models.stgm_full import Model as _Model


class STGM(Model):

    debug = 0

    def __init__(
        self,
        in_channels, 
        hidden_channels, 
        out_channels, 
        timestep_max, 
        To, 
        embedding_dict, 
        nb_blocks=1, 
        channels_last=True, 
        use_super_node=True, 
        degrees=None, 
        name="STGM_FULL", 
        device="cpu", 
        show_scores=False, 
    ):
        super(STGM, self).__init__()
        print(
            in_channels,  
            hidden_channels,  
            out_channels,  
            timestep_max,  
            embedding_dict,  
            nb_blocks, 
            channels_last, 
            use_super_node, 
            degrees, 
            name, 
            device, 
            show_scores, 
        )
        # Due to line 68 of Models/STGM_PyTorch/src/models/stgm_full.py : 
        #    Conv2d is made with in_channels=Ti-7 from GraphAttention module with time_filter=7
        #    so timestep_max > 7 must be true and inputs must be padded when less than 7 time-steps
        pad_len = 0
        if timestep_max < 7: # doesn't meet Ti>=7 req
            pad_len = max(0, 7 - timestep_max) # pad_len=[0,6]
            warnings.warn("STGM: found input time-steps (Ti=%d) was less than 7 but Ti>=7 must hold. Will continue by feeding inputs zero-padded to 7 time-steps." % (timestep_max))
        if timestep_max + pad_len < To: # even padded, doesn't meet Ti>=To
            new_pad_len = To - timestep_max
            warnings.warn("STGM: found input time-steps (Ti=%d) was less than output time-steps (To=%d) but Ti>=To must hold. Will continue by feeding inputs zero-padded to %d time-steps." % (timestep_max + pad_len, To, To))
            pad_len = new_pad_len
        # Instantiate model layers
        self.model = _Model(
            in_channels, 
            hidden_channels, 
            out_channels, 
            timestep_max + pad_len, 
            embedding_dict, 
            nb_blocks, 
            channels_last, 
            use_super_node, 
            degrees, 
            name,
            device, 
            show_scores, 
        )
        # Save all vars
        self.pad_len = pad_len
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["hidden_channels", None],
            ["embedding_dict", None],
            ["nb_blocks", None],
            ["channels_last", None], 
            ["use_super_node", None], 
            ["degrees", None], 
            ["name", None], 
        ]

    def forward(self, inputs):
        x = inputs["x"]
        idx = inputs["idx"]
        adj = inputs["adj"]
        n_temporal_out = inputs["n_temporal_out"]
        B, T, N, C = x.shape
        #
        if self.pad_len:
            x = torch.cat((torch.zeros((B, self.pad_len, N, C), device=x.device), x), 1)
            idx = torch.cat((torch.zeros((B, self.pad_len, 2), dtype=idx.dtype, device=idx.device), idx), 1)
        if self.debug:
            print("STGM Inputs = ")
            print("x =", x.shape)
            print("idx =", idx.shape)
            print("adj =", adj.shape)
        #
        a = self.model(x, torch.transpose(idx, -2, -1), adj.expand((B, N, N)), None)[:,-n_temporal_out:,:,:]
        if self.debug:
            print("STGM Output =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def pull_model_data(self, dataset, partition, var):
        princ_data = dataset.get(var.principle_data_type)
        data = {}
        data["__sampled__"] = ["x", "y", "idx"]
        data["__outputs__"] = ["y"]
        # Pull spatiotemporal data
        data["x"] = princ_data.transformed.get(var.principle_data_form).get("predictor_features", partition)
        data["y"] = princ_data.transformed.get(var.principle_data_form).get("response_features", partition)
        # Pull temporal data
        tod_index = util.temporal_labels_to_periodic_indices( # sod -> step-of-day
            dataset.spatiotemporal.original.get("temporal_labels", partition), 
            [1, "days"], 
            dataset.spatiotemporal.misc.temporal_resolution, 
            dataset.spatiotemporal.misc.temporal_label_format, 
        )
        dow_index = util.temporal_labels_to_periodic_indices( # dow -> day-of-week
            dataset.spatiotemporal.original.get("temporal_labels", partition), 
            [1, "weeks"], 
            [1, "days"], 
            dataset.spatiotemporal.misc.temporal_label_format, 
        )
        data["idx"] = np.stack((tod_index, dow_index), -1)
        # Pull graph data
        data["adj"] = dataset.graph.original.get("W", partition)
        # Pull etc
        data["n_temporal_out"] = var.temporal_mapping[1]
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    hyp_var.embedding_dict={
        "time": 24 * 7 * var.mapping.temporal_mapping[0],
        "day": 7,
        "node": spatmp.original.get("n_spatial", "train"),
        "degree": max(int(_[1]) for _ in graph.original.get("nx_graph", "train").degree),
    }
    model = STGM(
        spatmp.misc.n_predictor,
        hyp_var.hidden_channels,
        spatmp.misc.n_response,
        var.mapping.temporal_mapping[0], 
        var.mapping.temporal_mapping[1], 
        hyp_var.embedding_dict, 
        hyp_var.nb_blocks, 
        hyp_var.channels_last, 
        hyp_var.use_super_node, 
        hyp_var.degrees, 
        hyp_var.name, 
        var.execution.device, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        # Settings acquired from "Datasets and Settings" section of https://arxiv.org/pdf/2211.14701.pdf
        self.hidden_channels = 64
        self.embedding_dict = {}
        self.nb_blocks = 2
        self.channels_last = True
        self.use_super_node = False
        self.degrees = None
        self.name = "STGM_FULL"


class TrainingVariables(Container):

    def __init__(self):
        # Settings acquired from "Datasets and Settings" section of https://arxiv.org/pdf/2211.14701.pdf
        self.n_epochs = 200
        self.patience = 20
        self.mbatch_size = 64
        self.optimizer = "Adam"
        self.lr = 0.01
        self.loss = "L1Loss"
