import os
import sys
import time
import torch
import numpy as np
import pandas as pd

import util
from container import Container
from data import transform
from data.spatiotemporal import cluster as spatmp_cluster
from model import Model
from model import RNN_HyperparameterVariables
from model import TemporalMapper_HyperparameterVariables
from model import Cluster_HyperparameterVariables

from model.MMRGNN_PyTorch.model import MMRGNN as _MMRGNN


class MMRGNN(Model):

    debug = 0

    def __init__(
        self,
        Fs, # Fs (int) - Number of spatial features
        Ft, # Ft (int) - Number of temporal features
        Fst, # Fst (int) - Number of spatiotemporal features
        N, # N (int) - Number of nodes in original graph G
        Fst_out=1, # Fst_out (int) - Number of forecast features (output dimension)
        embed_size=10, # embed_size (int) - Number of dimensions used in node embeddings
        M=8, # M (int) - Number of implicit modalities
        H=16, # H (int) - Number of hidden units (embedding dimension)
        augr_kwargs={}, 
        enc_kwargs={}, 
        mapper_kwargs={}, 
        dec_kwargs={}, 
        out_layer="mLinear", 
        #
        cluster_kwargs={}, 
        use_existing_edges=True, 
    ):
        super(MMRGNN, self).__init__()
        self.model = _MMRGNN(
            Fs, # Fs (int) - Number of spatial features
            Ft, # Ft (int) - Number of temporal features
            Fst, # Fst (int) - Number of spatiotemporal features
            N, # N (int) - Number of nodes in original graph G
            Fst_out, # Fst_out (int) - Number of forecast features (output dimension)
            embed_size, # embed_size (int) - Number of dimensions used in node embeddings
            M, # M (int) - Number of implicit modalities
            H, # H (int) - Number of hidden units (embedding dimension)
            augr_kwargs, 
            enc_kwargs, 
            mapper_kwargs, 
            dec_kwargs, 
            out_layer, 
        )
        # Save vars
        self.M = M
        self.cluster_kwargs = cluster_kwargs
        self.use_existing_edges = use_existing_edges
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["embed_size", None], 
            ["M", None],
            ["H", None],
            ["augr_kwargs", None],
            ["enc_kwargs", None],
            ["mapper_kwargs", None],
            ["dec_kwargs", None],
            ["use_existing_edges", None],
        ]

    def forward(self, inputs):
        self.model.debug = self.debug
        inputs["xst"] = inputs["x"]
        return {"yhat": self.model(**inputs)["yhat"]}

    def reset_parameters(self):
        self.model.reset_parameters()

    def pull_model_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = ["x", "y"]
        data["__outputs__"] = ["y"]
        # pull prepared data
        #    Spatiotemporal data
        data["x"] = dataset.spatiotemporal.transformed.original.get("predictor_features", partition)
        data["y"] = dataset.spatiotemporal.transformed.original.get("response_features", partition)
        #    Spatial data
        if not dataset.spatial.is_empty():
            try:
                data["xs"] = dataset.spatial.transformed.original.get("predictor_features", partition)
            except:
                pass
        #    Temporal data
        data["xt"] = transform.minmax_transform(
            dataset.spatiotemporal.original.get("periodic_indices", partition)[:,None], 
            0, dataset.spatiotemporal.original.period_size-1, a=-1, b=1
        )
        data["yt"] = transform.minmax_transform(
            dataset.spatiotemporal.original.get("periodic_indices", partition)[:,None], 
            0, dataset.spatiotemporal.original.period_size-1, a=-1, b=1
        )
        data["__outputs__"].append("yt")
        data["__sampled__"] += ["xt", "yt"]
        #    Graph data
        if not dataset.graph.is_empty() and self.use_existing_edges:
            data["edge_index"] = dataset.graph.original.get("coo_edge_index", partition)
            if not var.graph.edge_weight_feature is None:
                if var.graph.edge_weight_feature == "weight":
                    data["edge_weight"] = dataset.graph.original.get("weights", partition)
        else:
            data["edge_index"] = None
        # Misc data
        data["modality_index"] = get_clusters(
            dataset, 
            self.cluster_kwargs.get("cluster_method", "Agglomerative"), 
            self.cluster_kwargs.get("n_clusters", self.M), 
            self.cluster_kwargs.get("clustered_representation", "histogram"), 
        )
        data["T"] = var.temporal_mapping[1]
        return data


def get_clusters(dataset, alg="Agglomerative", n_clusters=3, rep="histogram"):
    x, clustering, cluster_index = spatmp_cluster(dataset.spatiotemporal, alg, n_clusters, rep, bins=12, lims=[-3,3], features="predictor")
    return cluster_index


def init(dataset, var):
    spatial = dataset.spatial
    temporal = dataset.temporal
    spatiotemporal = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    # Setup hyper-parameter variables
    hyp_var.enc_kwargs.rnn_kwargs["xs_size"] = hyp_var.embed_size
    # Init
    model = MMRGNN(
        # Data parameters
        0 if spatial.is_empty() else spatial.misc.n_predictor, 
        1, 
        spatiotemporal.misc.n_predictor,
        spatiotemporal.original.get("n_spatial", "train"),
        spatiotemporal.misc.n_response,
        # Model hyper-parameters
        hyp_var.embed_size, 
        hyp_var.M, 
        hyp_var.H,
        hyp_var.augr_kwargs.to_dict(),
        hyp_var.enc_kwargs.to_dict(),
        hyp_var.mapper_kwargs.to_dict(),
        hyp_var.dec_kwargs.to_dict(),
        hyp_var.out_layer, 
        # Model hyper-parameters introduced by pipeline
        hyp_var.clustering.to_dict(), 
        hyp_var.use_existing_edges, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.embed_size = 10
        self.M = 8
        self.H = 16
        # Set GraphAugr defaults
        self.augr_kwargs = Container()
        self.augr_kwargs.graph_construction_method = ["top-k", ["dot", "Softmax"], 1.0]
        # Set Encoder RNN defaults
        self.enc_kwargs = Container()
        self.enc_kwargs.rnn_layer = "stGRU"
        self.enc_kwargs.n_rnn_layers = 1
        self.enc_kwargs.rnn_kwargs = {
            "xs_size": self.embed_size, 
            "xt_size": 1, 
            "conv": "cheb", 
            "layer": "kmLinear", 
            "order": 2, 
            "n_hops": 2, 
            "M": self.M,
        }
        # Set Temporal Mapper defaults
        self.mapper_kwargs = Container()
        self.temporal_mapper = "last"
        # Set Decoder RNN defaults
        self.dec_kwargs = Container()
        self.dec_kwargs.rnn_layer = "stGRU"
        self.dec_kwargs.n_rnn_layers = 1
        self.dec_kwargs.rnn_kwargs = {
            "xs_size": 0, 
            "xt_size": 1, 
            "conv": "cheb", 
            "layer": "kmLinear", 
            "order": 2, 
            "n_hops": 2, 
            "M": self.M,
        }
        # Set Other Defaults
        self.out_layer = "mLinear"
        # Set Clustering defaults
        self.clustering = Container()
        self.clustering.cluster_method = "Agglomerative"
        self.clustering.n_clusters = self.M
        self.clustering.clustered_representation = "histogram"
        # Set self defaults
        self.use_existing_edges = False


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 100
        self.mbatch_size = 64
        self.patience = 15
        self.lr = 3e-3
        self.initializer = None
        self.loss = "L1Loss"
