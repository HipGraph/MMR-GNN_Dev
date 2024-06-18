import os
import sys
import numpy as np
import pandas as pd

import experiment
import util


def module_dir():
    return os.path.dirname(__file__)


def module_name():
    return os.path.basename(__file__).replace(".py", "")


def get_models():
    return [
        # Pure temporal models
        "RNN", 
        "GRU", 
        "LSTM", 
        "TCN", 
        "FEDformer", 
        "LTSF_Linear", 
        "LTSF_NLinear", 
        "LTSF_DLinear", 
        # Pre-defined graph GCN models
        "GNN", 
        "TGCN", 
        "A3TGCN", 
        "STGCN", 
        "ASTGCN", 
        "STGM", 
        # Learned graph GCN models
        "GMAN", 
        "StemGNN", 
        "MTGNN", 
        "AGCRN", 
        "SCINet", 
        "MMRGNN", 
    ]


def get_datasets():
    return [
        # hydrology
        "littleriver", "wabashriver-swat", 
        # traffic
        "metr-la", "new-metr-la", "pems-bay", "new-pems-bay", 
        # energy
        "solar-energy", "electricity"
    ]


def get_init_seeds(self, n=10, rng=0):
    rng = np.random.default_rng(rng)
    return rng.integers(-sys.maxsize, sys.maxsize, size=n)


def add_dataset_args(args, dataset, exp):
    if dataset in ["littleriver", "wabashriver-swat"]: # Exps 1.x
        args.temporal_mapping = [7, 1]
        if dataset == "littleriver":
            args.predictor_features = ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"]
        if dataset == "wabashriver-swat":
            args.predictor_features = ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"]
    elif dataset in ["metr-la", "new-metr-la", "pems-bay", "new-pems-bay"]: # Exps 2.x
        args.temporal_mapping = [12, 12]
        if dataset in ["metr-la", "pems-bay"]:
            args.predictor_features = ["date", "speedmph"]
        elif dataset in ["new-metr-la", "new-pems-bay"]:
            args.predictor_features = [
                "Timestamp", "Samples", "Percent_Observed", "Total_Flow", "Avg_Occupancy", "Avg_Speed"
            ]
    elif dataset in ["traffic", "solar-energy", "electricity", "exchange-rate"]: # Exps 3.x
        if dataset == "solar-energy":
            args.temporal_mapping = [36, 1]
            args.predictor_features = ["date", "power_MW"] 
        if dataset == "electricity":
            args.temporal_mapping = [24, 1]
            args.predictor_features = ["date", "power_kWh"]
    else:
        raise NotImplementedError(dataset)
    args.response_features = args.predictor_features[-1:]
    args.set("dataset", dataset, ["train", "valid", "test"])
    return args


def add_model_args(args, model, dataset, exp):
    if model == "MMRGNN":
        if dataset == "littleriver":
            args.H = 16
            args.embed_size = 2
            args.M = 8
            args.loss = "L1Loss"
        elif dataset == "wabashriver-swat":
            args.H = 16
            args.M = 8
            args.loss = "L1Loss"
        elif dataset in ["metr-la", "new-metr-la", "pems-bay", "new-pems-bay"]:
            args.H = 16
            args.M = 2
            args.loss = "MSELoss"
        elif dataset == "solar-energy":
            args.H = 32
            args.M = 8
            args.loss = "L1Loss"
        elif dataset == "electricity":
            args.H = 64
            args.M = 3
            args.loss = "L1Loss"
        else:
            raise ValueError(model, dataset)
        args.predictor_features = util.list_subtract(args.predictor_features, ["date", "Timestamp"])
    elif model == "MTGNN":
        args.build_adj = True
        if dataset in ["littleriver", "exchange-rate"]:
            args.subgraph_size = 8
        if args.temporal_mapping[1] == 1: # single-step settings
            # hyper- and training parameters - already set by default settings in MTGNN module
            pass
        else: # multi-step settings
            # hyper-parameters
            args.layers = 3
            args.dilation_exponential = 1
            args.conv_channels = 32
            args.residual_channels = 32
            args.skip_channels = 64
            args.end_channels = 128
            # training parameters
            args.mbatch_size = 64
            args.n_epochs = 100
    elif model == "TCN":
        if args.temporal_mapping[1] > 1: # uni-variate when multi-step forcasting due to auto-regression
            args.predictor_features = args.response_features
    elif model == "STGCN":
        if args.temporal_mapping[0] < 12:
            args.Kt = 2
    elif model in ["GMAN", "GeoMAN"]: # uni-variate
        args.predictor_features = args.response_features
    elif model == "SCINet": # uni-variate
        if args.temporal_mapping[0] % 2 == 1: # input time-steps must be even
            args.temporal_mapping = [args.temporal_mapping[0] + 1] + args.temporal_mapping[1:]
        if dataset in ["littleriver"]: # 8 nodes - make hid_size=1/8>=1
            args.hid_size = 0.125
        args.predictor_features = args.response_features
    elif model == "StemGNN": # uni-variate
        args.predictor_features = args.response_features
    elif model == "FEDformer": # uni-variate
        args.predictor_features = args.response_features
    elif model in ["LTSF_Linear", "LTSF_NLinear", "LTSF_DLinear"]: # uni-variate
        args.predictor_features = args.response_features
    return args


class PAKDD2024_Experiment(experiment.Experiment):

    def add_first_args(self, args):
        if not "temporal_mapping" in args:
            args.temporal_mapping = [7, 1]
        args.default_feature_transform = "zscore"
        args.set("default_feature_transform", {"name": "minmax", "a": 0, "b": 1}, path=["processing", "spatial"])
        return args

    def add_last_args(self, args):
        args.feature_transform_map = {}
        if "date" in args.predictor_features:
            args.feature_transform_map = {"date": "minmax"}
        elif "Timestamp" in args.predictor_features:
            args.feature_transform_map = {"Timestamp": "minmax"}
        args.set("feature_transform_map", args.feature_transform_map, path=["processing", "spatiotemporal"])
        args.set(
            "default_feature_transform", args.default_feature_transform, path=["processing", "spatiotemporal"]
        )
        args.rem(["feature_transform_map", "default_feature_transform"])
        args.set("predictor_features", args.predictor_features, path=["mapping", "spatiotemporal"])
        args.set("response_features", args.response_features, path=["mapping", "spatiotemporal"])
        args.rem(["predictor_features", "response_features"])
        return args

    def root_dir(self):
        return os.sep.join([module_dir(), module_name().upper(), self.name()])


class MainExperiment(PAKDD2024_Experiment):

    def dataset(self):
        raise NotImplementedError()

    def models(self):
        return get_models()

    def init_seeds(self):
        return get_init_seeds(self.n_trials())

    def n_trials(self):
        return 10

    def grid_argvalue_map(self):
        return {
            "model": self.models(), 
            "initialization_seed": self.init_seeds(), 
        }

    def add_pre_subexp(self, args, subexp_id):
        args = add_dataset_args(args, self.dataset(), self)
        args = add_model_args(args, args.model, self.dataset(), self)
        return args


class Experiment__1_1(MainExperiment):

    def name(self):
        return "LittleRiver"

    def dataset(self):
        return "littleriver"


class Experiment__1_2(MainExperiment):

    def name(self):
        return "WabashRiver-SWAT"

    def dataset(self):
        return "wabashriver-swat"


class Experiment__2_1(MainExperiment):

    def name(self):
        return "METR-LA"

    def dataset(self):
        return "metr-la"


class Experiment__2_11(MainExperiment):

    def name(self):
        return "NEW-METR-LA"

    def dataset(self):
        return "new-metr-la"


class Experiment__2_2(MainExperiment):

    def name(self):
        return "PEMS-BAY"

    def dataset(self):
        return "pems-bay"


class Experiment__2_21(MainExperiment):

    def name(self):
        return "NEW-PEMS-BAY"

    def dataset(self):
        return "new-pems-bay"


class Experiment__3_1(MainExperiment):

    def name(self):
        return "Solar-Energy"

    def dataset(self):
        return "solar-energy"


class Experiment__3_2(MainExperiment):

    def name(self):
        return "Electricity"

    def dataset(self):
        return "electricity"



class AblationExperiment(experiment.ParameterSearch, MainExperiment):

    def dataset(self):
        return "wabashriver-swat"

    def models(self):
        return ["THSTM"]

    def n_trials(self):
        return 3


class Experiment__0_1(AblationExperiment):

    def name(self):
        return "Ablation(Decoder)"

    def subexp_ids(self):
        return range(2)

    def add_subexp(self, args, subexp_id):
        if subexp_id == 0: # without decoder
            args.kwargs = {
                "arch0": 2, 
                "arch": 0, 
                "arch2": 0, 
                "arch3": 3, 
                "embed_size": 10, 
                "bn": 0, 
                "out_layer": 3, 
                "ignore_self_loops": False, 
            }
        elif subexp_id == 1: # with decoder
            pass
        args.temporal_mapping[-1] = args.temporal_mapping[0]
        return args


class Experiment__0_2(AblationExperiment):

    def name(self):
        return "Ablation(MultiDomain)"

    def subexp_ids(self):
        return range(2)

    def add_subexp(self, args, subexp_id):
        if subexp_id == 0: # without gcstGRU
            args.gcn_kwargs = {
                "node_layer": "gcGRU", 
                "edge_act": "Identity", 
                "normalize": 0, 
                "nn_order": 2, 
                "n_hops": 0, 
                "bias": False
            }
            args.gcn_kwargs["rnn_kwargs"] = {
                "conv": "cheb", "layer": "kgLinear", "order": 2, "n_hops": 2, "embed_size": 10
            }
            args.rnn_layer = "gcGRU"
        elif subexp_id == 1: # with gcstGRU: default
            pass
        return args


class Experiment__0_3(AblationExperiment):

    def name(self):
        return "Ablation(ModalitySpace)"

    def subexp_ids(self):
        return range(5)

    def add_subexp(self, args, subexp_id):
        n_spatial = 100
        #
        if subexp_id in [0, 2, 3, 4]:
            args.model = "THSTM"
        else:
            args.model = "AGCRN"
        args.set("spatial_selection", ["random", n_spatial, 0], ["train", "valid", "test"])
        args.set("node_selection", ["random", n_spatial, 0], ["train", "valid", "test"])
        if subexp_id == 0:
            args.n_clusters = 1
        elif subexp_id == 1: # AGCRN
            pass
        elif subexp_id == 2: # THSTM using mixing (kvwLinear)
            args.gcn_kwargs = {
                "node_layer": "gcGRU", 
                "edge_act": "Identity", 
                "normalize": 0, 
                "nn_order": 2, 
                "n_hops": 0, 
                "bias": False
            }
            args.gcn_kwargs["rnn_kwargs"] = {
                "xs_size": 10, "xt_size": 1, 
                "conv": "cheb", "layer": "kvwLinear", "order": 2, "n_hops": 2, "embed_size": 10
            }
            args.rnn_layer = "gcGRU"
            args.rnn_kwargs = {
                "xs_size": 0, "xt_size": 1, 
                "conv": "cheb", "layer": "kvwLinear", "order": 2, "n_hops": 2, "embed_size": 10
            }
            args.kwargs = {
                "arch0": 2, 
                "arch": 0, 
                "arch2": 0, 
                "arch3": 3, 
                "embed_size": 10, 
                "bn": 0, 
                "out_layer": 2, 
                "ignore_self_loops": False, 
            }
        elif subexp_id == 3: # original THSTM
            args.n_clusters = 10
        else: # full-rank
            args.n_clusters = n_spatial
        if subexp_id in [0, 2, 3, 4]:
            args.hidden_size = 16
        return args


class Experiment__0_31(AblationExperiment):

    def name(self):
        return "Ablation(Modality)"

    def subexp_ids(self):
        self.v = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        return range(len(self.v))

    def add_subexp(self, args, subexp_id):
        args.n_clusters = self.v[subexp_id]
        args.hidden_size = 8
        return args


class Experiment__0_4(AblationExperiment):

    def name(self):
        return "Ablation(GraphAdaptation)"

    def subexp_ids(self):
        self.v = [0.25, 0.5, 0.75, round((1276**2-1275)/1276**2, 3)]
        return np.arange(2+len(self.v))

    def add_subexp(self, args, subexp_id):
        if subexp_id == 0:
            args.use_existing_edges = False
            args.graph_construction_method = None
        elif subexp_id == 1:
            args.use_existing_edges = True
            args.graph_construction_method = None
        elif subexp_id in range(2, 6):
            args.use_existing_edges = True
            args.graph_construction_method = ["top-k", ["dot", "Softmax"], self.v[subexp_id-2]]
        return args


class Experiment__0_5(AblationExperiment):

    def name(self):
        return "Ablation(HiddenSize)"

    def subexp_ids(self):
        self.v = 2**(np.arange(7))
        return range(len(self.v))

    def add_subexp(self, args, subexp_id):
        args.hidden_size = self.v[subexp_id]
        return args


class Experiment__0_6(AblationExperiment):

    def name(self):
        return "Ablation(Horizon)"

    def subexp_ids(self):
        self.horizons = {
            "littleriver": [2, 3, 5, 7], 
            "wabashriver-swat": [2, 3, 5, 7], 
            "metr-la": [3, 6, 9, 12], 
            "new-metr-la": [3, 6, 9, 12], 
            "pems-bay": [3, 6, 9, 12], 
            "new-pems-bay": [3, 6, 9, 12], 
            "solar-energy": [8, 16, 24, 36], 
            "electricity": [6, 12, 18, 24],
        }
        self.horizons = {
            "littleriver": [3, 5, 7], 
            "wabashriver-swat": [3, 5, 7], 
            "metr-la": [4, 8, 12], 
            "new-metr-la": [4, 8, 12], 
            "pems-bay": [4, 8, 12], 
            "new-pems-bay": [4, 8, 12], 
            "solar-energy": [12, 24, 36], 
            "electricity": [8, 16, 24],
        }
        self.horizons = {
            "littleriver": [3, 7], 
            "wabashriver-swat": [3, 7], 
            "metr-la": [6, 12], 
            "new-metr-la": [6, 12], 
            "pems-bay": [6, 12], 
            "new-pems-bay": [6, 12], 
            "solar-energy": [18, 36], 
            "electricity": [12, 24],
        }
        return range(2)

    def add_subexp(self, args, subexp_id):
        args.temporal_mapping[-1] = self.horizons[self.dataset()][subexp_id]
        return args


class Experiment__0_61(Experiment__0_6):

    def dataset(self):
        return "littleriver"


class Experiment__0_62(Experiment__0_6):

    def dataset(self):
        return "wabashriver-swat"


class Experiment__0_63(Experiment__0_6):

    def dataset(self):
        return "metr-la"


class Experiment__0_64(Experiment__0_6):

    def dataset(self):
        return "new-metr-la"


class Experiment__0_65(Experiment__0_6):

    def dataset(self):
        return "pems-bay"


class Experiment__0_66(Experiment__0_6):

    def dataset(self):
        return "new-pems-bay"


class Experiment__0_67(Experiment__0_6):

    def dataset(self):
        return "solar-energy"


class Experiment__0_68(Experiment__0_6):

    def dataset(self):
        return "electricity"


class Experiment__0_7(AblationExperiment):

    def name(self):
        return "Ablation(RNNCell)"

    def subexp_ids(self):
        self.v = ["gcstRNN", "gcstGRU", "gcstLSTM"]
        return range(len(self.v))

    def add_subexp(self, args, subexp_id):
        args.gcn_kwargs = {
            "node_layer": self.v[subexp_id], "edge_act": "Identity", "normalize": 0, "nn_order": 2, "n_hops": 0, 
            "bias": False
        }
        args.gcn_kwargs["rnn_kwargs"] = {
            "xs_size": 10, "xt_size": 1, "conv": "cheb", "layer": "kgLinear", "order": 2, "n_hops": 2
        }
        args.rnn_layer = self.v[subexp_id]
        args.rnn_kwargs = {
            "xs_size": 0, "xt_size": 1, "conv": "cheb", "layer": "kgLinear", "order": 2, "n_hops": 2
        }
        args.kwargs = {
            "arch0": 2, 
            "arch": 0, 
            "arch2": 0, 
            "arch3": 3, 
            "embed_size": 10, 
            "bn": 0, 
            "out_layer": 2, 
            "ignore_self_loops": False, 
        }
        return args


class Experiment__0_8(AblationExperiment):

    def name(self):
        return "Ablation(Fusion)"

    def subexp_ids(self):
        self.v = ["add", "attn"]
        return range(len(self.v))

    def add_subexp(self, args, subexp_id):
        args.gcn_kwargs = {
            "node_layer": "gcstGRU", "edge_act": "Identity", "normalize": 0, "nn_order": 2, "n_hops": 0, 
            "bias": False, 
        }
        args.gcn_kwargs["rnn_kwargs"] = {
            "xs_size": 10, "xt_size": 1, "conv": "cheb", "layer": "kgLinear", "order": 2, "n_hops": 2, 
            "shared": False, "fusion": self.v[subexp_id], 
        }
        args.rnn_layer = "gcstGRU"
        args.rnn_kwargs = {
            "xs_size": 0, "xt_size": 1, "conv": "cheb", "layer": "kgLinear", "order": 2, "n_hops": 2, 
            "shared": False, "fusion": self.v[subexp_id], 
        }
        args.kwargs = {
            "arch0": 2, 
            "arch": 0, 
            "arch2": 0, 
            "arch3": 3, 
            "embed_size": 10, 
            "bn": 0, 
            "out_layer": 2, 
            "ignore_self_loops": False, 
        }
        return args


class Experiment__0_9(AblationExperiment):

    def name(self):
        return "Ablation(Clustering)"

    def subexp_ids(self):
        self.v = ["Random", "KMeans", "Agglomerative"]
        return range(len(self.v))

    def add_subexp(self, args, subexp_id):
        args.cluster_method = self.v[subexp_id]
        return args


class Experiment__0_10(AblationExperiment):

    def name(self):
        return "Ablation(STLT)"

    def dataset(self):
        raise NotImplementedError()

    def models(self):
        return ["FEDformer", "LTSF_DLinear", "AGCRN", "THSTM"]

    def subexp_ids(self):
        self.i = {
            "littleriver": 28, 
            "wabashriver-swat": 28, 
            "metr-la": 48, 
            "new-metr-la": 48, 
            "pems-bay": 48, 
            "new-pems-bay": 48, 
            "solar-energy": 144, 
            "electricity": 96, 
        }
        self.o = {
            "littleriver": [1, 7, 14, 21, 28], 
            "wabashriver-swat": [1, 7, 14, 21, 28], 
            "metr-la": [1, 12, 24, 36, 48], 
            "new-metr-la": [1, 12, 24, 36, 48], 
            "pems-bay": [1, 12, 24, 36, 48], 
            "new-pems-bay": [1, 12, 24, 36, 48], 
            "solar-energy": [1, 36, 72, 108, 144], 
            "electricity": [1, 24, 48, 72, 96], 
        }
        return range(5)
        self.o = {
            "littleriver": [1, 14, 28], 
            "wabashriver-swat": [1, 14, 28], 
            "metr-la": [1, 24, 48], 
            "new-metr-la": [1, 24, 48], 
            "pems-bay": [1, 24, 48], 
            "new-pems-bay": [1, 24, 48], 
            "solar-energy": [1, 72, 144], 
            "electricity": [1, 48, 96], 
        }
        return range(3)

    def add_subexp(self, args, subexp_id):
        args.temporal_mapping = [self.i[self.dataset()], self.o[self.dataset()][subexp_id]]
        return args


class Experiment__0_101(Experiment__0_10):

    def dataset(self):
        return "littleriver"


class Experiment__0_102(Experiment__0_10):

    def dataset(self):
        return "wabashriver-swat"


class Experiment__0_103(Experiment__0_10):

    def dataset(self):
        return "metr-la"


class Experiment__0_104(Experiment__0_10):

    def dataset(self):
        return "new-metr-la"


class Experiment__0_105(Experiment__0_10):

    def dataset(self):
        return "pems-bay"


class Experiment__0_106(Experiment__0_10):

    def dataset(self):
        return "new-pems-bay"


class Experiment__0_107(Experiment__0_10):

    def dataset(self):
        return "solar-energy"


class Experiment__0_108(Experiment__0_10):

    def dataset(self):
        return "electricity"


class Experiment__0_211(AblationExperiment):

    def name(self):
        return "NEW-METR-LA[FI]"

    def subexp_ids(self):
        self.predictor_sets = [
            ["Avg_Speed"], 
            ["Avg_Occupancy", "Avg_Speed"], 
            ["Total_Flow", "Avg_Occupancy", "Avg_Speed"], 
            ["Percent_Observed", "Total_Flow", "Avg_Occupancy", "Avg_Speed"], 
            ["Samples", "Percent_Observed", "Total_Flow", "Avg_Occupancy", "Avg_Speed"], 
        ]
        return range(len(self.predictor_sets))

    def add_subexp(self, args, subexp_id):
        args.predictor_features = self.predictor_sets[subexp_id]
        return args


class Experiment__0_221(AblationExperiment):

    def name(self):
        return "NEW-PEMS-BAY[FI]"

    def subexp_ids(self):
        self.predictor_sets = [
            ["Avg_Speed"], 
            ["Avg_Occupancy", "Avg_Speed"], 
            ["Total_Flow", "Avg_Occupancy", "Avg_Speed"], 
            ["Percent_Observed", "Total_Flow", "Avg_Occupancy", "Avg_Speed"], 
            ["Samples", "Percent_Observed", "Total_Flow", "Avg_Occupancy", "Avg_Speed"], 
        ]
        return range(len(self.predictor_sets))

    def add_subexp(self, args, subexp_id):
        args.predictor_features = self.predictor_sets[subexp_id]
        return args
