import itertools
import gzip
import pandas as pd
import os
import sys
import time
import shutil
import numpy as np
import datetime as dt
import re
import torch
from inspect import currentframe
from sklearn.preprocessing import OneHotEncoder

import util
from data.integration import *
from data import transform
from data.selection import DataSelection
from container import Container
from variables import Variables
from data.dataset import Data
from model import GraphConstructor, GraphConstructor_HyperparameterVariables


def copy_missing_data(integrator, generator):
    if os.path.exists(integrator.spatial_labels_outpath()) and not os.path.exists(generator.spatial_labels_outpath()):
        shutil.copy(integrator.spatial_labels_outpath(), generator.spatial_labels_outpath())
    if os.path.exists(integrator.temporal_labels_outpath()) and not os.path.exists(generator.temporal_labels_outpath()):
        shutil.copy(integrator.temporal_labels_outpath(), generator.temporal_labels_outpath())
    if os.path.exists(integrator.spatiotemporal_features_outpath()) and not os.path.exists(generator.spatiotemporal_features_outpath()):
        shutil.copy(integrator.spatiotemporal_features_outpath(), generator.spatiotemporal_features_outpath())
    if os.path.exists(integrator.spatial_features_outpath()) and not os.path.exists(generator.spatial_features_outpath()):
        shutil.copy(integrator.spatial_features_outpath(), generator.spatial_features_outpath())
    if os.path.exists(integrator.temporal_features_outpath()) and not os.path.exists(generator.temporal_features_outpath()):
        shutil.copy(integrator.temporal_features_outpath(), generator.temporal_features_outpath())
    if os.path.exists(integrator.graph_features_outpath()) and not os.path.exists(generator.graph_features_outpath()):
            shutil.copy(integrator.graph_features_outpath(), generator.graph_features_outpath())


def generate_graph(dataset, inputs, var, args):
    # Handle imports
    # Handle arguments
    hyp_var = Container().set(
        [
            "graph_constructor_kwargs", 
        ], 
        [
            GraphConstructor_HyperparameterVariables(), 
        ], 
        multi_value=True
    )
    hyp_var = hyp_var.merge(args)
    # Prepare model and data
    model = GraphConstructor(**hyp_var.graph_constructor_kwargs.to_dict())
    model.debug = args.debug
    model, inputs = model.prepare(inputs, var.execution.device)
    # Generate graph data
    model.debug = 1
    outputs = model(**inputs)
    edge_index = util.to_ndarray(outputs["edge_index"])
    edge_weight = util.to_ndarray(outputs["edge_weight"])
    if args.debug:
        print("edge_index =", edge_index.shape)
        print(edge_index)
        print("edge_weight =", edge_weight.shape)
        print(edge_weight)
    if args.prune_weightless:
        edge_index = edge_index[:,edge_weight > 0]
        edge_weight = edge_weight[edge_weight > 0]
        if args.debug:
            print("edge_index =", edge_index.shape)
            print(edge_index)
            print("edge_weight =", edge_weight.shape)
            print(edge_weight)
    spatial_labels = dataset.get(
        "spatial_labels", 
        "train", 
        path=[var.execution.principle_data_type, var.execution.principle_data_form]
    )
    df = pd.DataFrame(
        {
            "source": spatial_labels[edge_index[0,:]], 
            "destination": spatial_labels[edge_index[1,:]], 
            "weight": edge_weight, 
        }, 
        dtype=str
    )
    return df


class Generator:

    debug = 2

    def __init__(self):
        os.makedirs(self.root_dir(), exist_ok=True)
        os.makedirs(self.cache_dir(), exist_ok=True)
        os.makedirs(self.generate_dir(), exist_ok=True)

    def name(self):
        return self.__class__.__name__

    def root_dir(self):
        return os.sep.join([data_dir(), self.name()]) 

    def cache_dir(self):
        return os.sep.join([self.root_dir(), "Generation", "Cache"])

    def generate_dir(self):
        return self.root_dir()

    def spatial_labels_fname(self):
        return "SpatialLabels.csv"

    def temporal_labels_fname(self):
        return "TemporalLabels.csv"

    def spatial_features_fname(self):
        return "Spatial.csv"

    def temporal_features_fname(self):
        return "Temporal.csv"

    def spatiotemporal_features_fname(self):
        return "Spatiotemporal.csv.gz"

    def graph_features_fname(self):
        return "Graph.csv"

    def spatial_labels_outpath(self):
        return os.sep.join([self.generate_dir(), self.spatial_labels_fname()])

    def temporal_labels_outpath(self):
        return os.sep.join([self.generate_dir(), self.temporal_labels_fname()])

    def spatial_features_outpath(self):
        return os.sep.join([self.generate_dir(), self.spatial_features_fname()])

    def temporal_features_outpath(self):
        return os.sep.join([self.generate_dir(), self.temporal_features_fname()])

    def spatiotemporal_features_outpath(self):
        return os.sep.join([self.generate_dir(), self.spatiotemporal_features_fname()])

    def graph_features_outpath(self):
        return os.sep.join([self.generate_dir(), self.graph_features_fname()])

    def generate(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))


class GraphGenerator(Generator):

    def name(self):
        raise NotImplementedError()

    def dataset(self):
        raise NotImplementedError()

    def default_args(self):
        return Container().set(
            [
                "debug", 
                "graph_construction_method", 
                "prune_weightless", 
            ], 
            [
                2, 
                ["k-nn", "Minkowski-2", 2], 
                True, 
            ], 
            multi_value=True
        )

    def pull_data(self, dataset, partition, var):
        raise NotImplementedError()

    def generate(self, args):
        # Handle imports
        # Handle arguments + variables
        args = self.default_args().merge(args)
        var = Variables().merge(args)
        var.meta.load_graph = False
        # Generate graph
        var.execution.set("dataset", self.dataset(), ["train", "valid", "test"])
        dataset = Data(var).get("dataset", "train")
        inputs = self.pull_data(dataset, "train", None)
        df = generate_graph(dataset, inputs, var, args)
        df.to_csv(self.graph_features_outpath(), index=False)


class NEW_METR_LA(GraphGenerator):

    def name(self):
        return "NEW-METR-LA"

    def dataset(self):
        return "new-metr-la"

    def pull_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = []
        P = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            np.array(util.get_dict_values(dataset.spatial.misc.feature_index_map, ["Longitude", "Latitude"]))
        ).astype(float)
        print(P.shape)
        print(np.min(P, 0), np.max(P, 0))
        P = transform.minmax_transform(P, np.min(P, 0), np.max(P, 0))
        print(np.min(P, 0), np.max(P, 0))
        input()
        F = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            dataset.spatial.misc.feature_index_map["Freeway"]
        )[:,None]
        B = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            dataset.spatial.misc.feature_index_map["Direction"]
        )[:,None]
        _F = OneHotEncoder(sparse_output=False).fit_transform(F)
        _B = OneHotEncoder(sparse_output=False).fit_transform(B)
        _R = np.dot(_F, _F.T) * np.dot(_B, _B.T)
        _D = np.zeros(_R.shape)
        for i, j in zip(*np.where(_R == 1)):
            if B[i] == "N":
                _D[i,j] = P[j][1] > P[i][1]
            if B[i] == "S":
                _D[i,j] = P[j][1] < P[i][1]
            if B[i] == "W":
                _D[i,j] = P[j][0] < P[i][0]
            if B[i] == "E":
                _D[i,j] = P[j][0] > P[i][0]
        data["x"] = P
        data["m"] = _R
        data["b"] = -np.eye(P.shape[0])# * sys.float_info.max
        data["ignore_self_loops"] = False
        for key, value in data.items():
            if key == "x" and isinstance(value, list):
                for i, X in enumerate(value):
                    print("x[%d] =" % (i), X.shape)
            elif isinstance(value, np.ndarray):
                print(key, "=", value.shape)
            else:
                print(key, "=", value)
        return data


class NEW_PEMS_BAY(NEW_METR_LA):

    def name(self):
        return "NEW-PEMS-BAY"

    def dataset(self):
        return "new-pems-bay"
