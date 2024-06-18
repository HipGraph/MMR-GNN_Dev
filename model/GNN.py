import os
import sys
import torch

import util
from container import Container
from model import Model
from model import GNN as _GNN


class GNN(Model):

    def __init__(
        self,
        in_size,
        out_size,
        n_temporal_in,
        n_temporal_out,
        hidden_size=16,
        n_hops=1,
        gcn_layer="GCNConv",
        gcn_kwargs={},
        act_layer="ReLU",
        act_kwargs={},
        dropout=0.0,
        use_edge_weights=False,
    ):
        super(GNN, self).__init__()
        # Instantiate model layers
        self.gnn = _GNN(
            in_size,
            hidden_size,
            hidden_size,
            n_hops,
            gcn_layer,
            gcn_kwargs,
            act_layer,
            act_kwargs,
            dropout,
            use_edge_weights,
        )
        self.tmp_red = self.layer_fn_map["Linear"](n_temporal_in, n_temporal_out)
        self.tmp_red_act = self.layer_fn_map[act_layer](**act_kwargs)
        self.out_proj = self.layer_fn_map["Linear"](hidden_size, out_size)
        self.out_proj_act = self.layer_fn_map["Identity"]()
        # Save all vars
        self.in_size, self.out_size = in_size, out_size
        self.n_temporal_in, self.n_temporal_out = n_temporal_in, n_temporal_out
        self.hidden_size, self.n_hops = hidden_size, n_hops
        self.gcn_layer, self.gcn_kwargs = gcn_layer, gcn_kwargs
        self.act_layer, self.act_kwargs = act_layer, act_kwargs
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["hidden_size", None],
            ["n_hops", None],
            ["gcn_layer", None],
            ["gcn_kwargs", None],
            ["act_layer", None],
            ["act_kwargs", None],
            ["dropout", None],
            ["use_edge_weights", None],
        ]

    def forward(self, inputs):
#        self.debug = 1
        self.gnn.debug = self.debug
        x, edge_index = inputs["x"], inputs["edge_index"]
        n_sample, n_temporal_in, n_spatial, in_size = x.shape
        if self.debug:
            print(util.make_msg_block("GNN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
        if self.debug:
            print("edge_index =", edge_index.shape)
            print(util.memory_of(edge_index))
        # GNN layer(s) forward
        a = torch.reshape(x, (-1, n_spatial, in_size))
        a = self.gnn(x=a, edge_index=edge_index)["yhat"]
        a = torch.reshape(a, (n_sample, n_temporal_in, n_spatial, self.hidden_size))
        if self.debug:
            print("GNN Output =", a.shape)
            print(util.memory_of(a))
        # GNN temporal axis reduction forward
        a = torch.transpose(a, 1, -1)
        z = self.tmp_red(a)
        a = self.tmp_red_act(z)
        a = torch.transpose(a, 1, -1)
        if self.debug:
            print("Temporal Reduction =", a.shape)
            print(util.memory_of(a))
        # Output layer forward
        z = self.out_proj(a)
        a = self.out_proj_act(z)
        if self.debug:
            print("Output =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.tmp_red.reset_parameters()
        self.out_proj.reset_parameters()

    def pull_model_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = ["x", "y"]
        data["__outputs__"] = ["y"]
        # Pull spatiotemporal data
        data["x"] = dataset.spatiotemporal.transformed.original.get("predictor_features", partition)
        data["y"] = dataset.spatiotemporal.transformed.original.get("response_features", partition)
        # Pull graph data
        data["edge_index"] = dataset.graph.original.get("coo_edge_index", partition)
        if not var.graph.edge_weight_feature is None:
            if var.graph.edge_weight_feature == "weight":
                data["edge_weight"] = dataset.graph.original.get("weights", partition)
            elif self.gcn_layer == "FlowConv" and self.gcn_kwargs.get("normalize", 1) in [0, 1]: # edge_weight.shape=(?, |E|)
                data["edge_weight"] = dataset.spatiotemporal.filter_axis(
                    dataset.spatiotemporal.original.get("features", partition),
                    [-2, -1],
                    [
                        data["edge_index"][0,:],
                        dataset.spatiotemporal.misc.feature_index_map[var.graph.edge_weight_feature]
                    ]
                )
                data["__sampled__"].append("edge_weight")
            else: # edge_weight.shape=(?, |V|)
                data["edge_weight"] = dataset.spatiotemporal.filter_axis(
                    dataset.spatiotemporal.original.get("features", partition),
                    -1,
                    dataset.spatiotemporal.misc.feature_index_map[var.graph.edge_weight_feature]
                )
            data["__sampled__"].append("edge_weight")
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    hyp_var = var.models.get(model_name()).hyperparameters
    model = GNN(
        spatmp.misc.n_predictor,
        spatmp.misc.n_response,
        var.mapping.temporal_mapping[0],
        var.mapping.temporal_mapping[1],
        hyp_var.hidden_size,
        hyp_var.n_hops,
        hyp_var.gcn_layer,
        hyp_var.gcn_kwargs,
        hyp_var.act_layer,
        hyp_var.act_kwargs,
        hyp_var.dropout,
        hyp_var.use_edge_weights,
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.hidden_size = 16
        self.n_hops = 1
        self.gcn_layer = "GCNConv"
        self.gcn_kwargs = {}
        self.act_layer = "Identity"
        self.act_kwargs = {}
        self.dropout = 0.0
        self.use_edge_weights = True


class TrainingVariables(Container):

    def __init__(self):
        self.initialization = None
