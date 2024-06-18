import os
import sys
import time
import torch
import warnings

import util
from container import Container
from model import Model

from model.MTGNN_PyTorch.net import gtnet


# Notes on MTGNN:
#   Only works for graphs whose nodes have a single response variables
#   Takes inputs in the shape (n_samples, in_channels, n_temporal_in, n_sensors)
#   It operates using the adjacency matrix requiring O(N^2) memory. Specifically, the mixprop layers use torch.einsum("ncvl,nvwl->ncwl", (x, A)) which computes Ax and effectively sums features (x) over the neighborhood (A) of each node. Note: Einsum does not build any temporary product arrays.
class MTGNN(Model):

    def __init__(
        self, 
        in_dim, # in_dim (int) - Number of input features
        seq_length, # seq_length (int) - Length of input sequence
        out_dim, # out_dim (int) - Length of output sequence
        num_nodes, # num_nodes (int) - Number of nodes in the graph
        predefined_A=None, # predefined_A (array w/ shape=(|V|, |V|)) - Existing adjacency matrix
        static_feat=None, # static_feat (array w/ shape=(|V|, D) - Static features existing on each node
        gcn_true=True, # gcn_true (bool) - Whether to add graph convolution layer
        buildA_true=True, # buildA_true (bool) - Whether to learn an adjacency matrix
        gcn_depth=2, # gcn_depth (int) - Graph convolution depth
        dropout=0.3, # dropout (float) - Droupout rate
        subgraph_size=20, # subgraph_size (int) - Size of subgraph (k in k-nn)
        node_dim=40, # node_dim (int) - Dimension of node embeddings used for graph construction
        dilation_exponential=2, # dilation_exponential (int) - Dilation exponential
        conv_channels=16, # conv_channels (int) - Convolution channels
        residual_channels=16, # residual_channels (int) - Residual channels
        skip_channels=32, # skip_channels (int) - Skip channels
        end_channels=64, # end_channels (int) - End channels
        layers=5, # layers (int) - Number of layers
        propalpha=0.05, # propalpha (float) - Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1
        tanhalpha=3, # tanhalpha (float) - Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
        layer_norm_affline=True, # layer_norm_affline (bool) - Whether to do elementwise affine in Layer Normalization
        device="cpu", 
    ):
        super(MTGNN, self).__init__()
        if 1:
            print(
                in_dim, # in_dim (int) - Number of input features
                seq_length, # seq_length (int) - Length of input sequence
                out_dim, # out_dim (int) - Length of output sequence
                num_nodes, # num_nodes (int) - Number of nodes in the graph
                predefined_A, # predefined_A (array w/ shape=(|V|, |V|)) - Existing adjacency matrix
                static_feat, # static_feat (array w/ shape=(|V|, D) - Static features existing on each node
                gcn_true, # gcn_true (bool) - Whether to add graph convolution layer
                buildA_true, # buildA_true (bool) - Whether to learn an adjacency matrix
                gcn_depth, # gcn_depth (int) - Graph convolution depth
                dropout, # dropout (float) - Droupout rate
                subgraph_size, # subgraph_size (int) - Size of subgraph (k in k-nn)
                node_dim, # node_dim (int) - Dimension of node embeddings used for graph construction
                dilation_exponential, # dilation_exponential (int) - Dilation exponential
                conv_channels, # conv_channels (int) - Convolution channels
                residual_channels, # residual_channels (int) - Residual channels
                skip_channels, # skip_channels (int) - Skip channels
                end_channels, # end_channels (int) - End channels
                layers, # layers (int) - Number of layers
                propalpha, # propalpha (float) - Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1
                tanhalpha, # tanhalpha (float) - Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
                layer_norm_affline, # layer_norm_affline (bool) - Whether to do elementwise affine in Layer Normalization
                device, 
            )
        # Instantiate model layers
        if subgraph_size > num_nodes:
            raise ValueError("Sub-graph may NOT contain more nodes than original graph. Received subgraph_size=%d and num_nodes=%d" % (subgraph_size, num_nodes))
        pad_len = max(0, out_dim-seq_length)
        if pad_len:
            warnings.warn("MTGNN: found input time-steps (Ti=%d) are less than output time-steps (To=%d) but Ti>=To must hold. Will continue by feeding inputs zero-padded to %d time-steps" % (seq_length, out_dim, seq_length+pad_len))
        self.model = gtnet(
            gcn_true, # gcn_true (bool) - Whether to add graph convolution layer
            buildA_true, # buildA_true (bool) - Whether to learn an adjacency matrix
            gcn_depth, # gcn_depth (int) - Graph convolution depth
            num_nodes, # num_nodes (int) - Number of nodes in the graph
            device, 
            predefined_A, # predefined_A (array w/ shape=(|V|, |V|)) - Existing adjacency matrix
            static_feat, # static_feat (array w/ shape=(|V|, D) - Static features existing on each node
            dropout, # dropout (float) - Droupout rate
            subgraph_size, # subgraph_size (int) - Size of subgraph (k in k-nn)
            node_dim, # node_dim (int) - Dimension of node embeddings used for graph construction
            dilation_exponential, # dilation_exponential (int) - Dilation exponential
            conv_channels, # conv_channels (int) - Convolution channels
            residual_channels, # residual_channels (int) - Residual channels
            skip_channels, # skip_channels (int) - Skip channels
            end_channels, # end_channels (int) - End channels
            seq_length+pad_len, # seq_length (int) - Length of input sequence
            in_dim, # in_dim (int) - Number of input features
            out_dim, # out_dim (int) - Length of output sequence
            layers, # layers (int) - Number of layers
            propalpha, # propalpha (float) - Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1
            tanhalpha, # tanhalpha (float) - Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
            layer_norm_affline, # layer_norm_affline (bool) - Whether to do elementwise affine in Layer Normalization
        )
        # Save all vars
        self.in_dim = in_dim
        self.seq_length = seq_length
        self.out_dim = out_dim
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.pad_len = pad_len
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["gcn_true", None], 
            ["build_adj", None], 
            ["gcn_depth", None], 
            ["dropout", None], 
            ["subgraph_size", None], 
            ["node_dim", None], 
            ["dilation_exponential", None], 
            ["conv_channels", None], 
            ["residual_channels", None], 
            ["skip_channels", None], 
            ["end_channels", None], 
            ["layers", None], 
            ["propalpha", None], 
            ["tanhalpha", None], 
            ["layer_norm_affline", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        self.use_spatial_features = False
        x, adj, x_s = inputs["x"], inputs.get("adj", None), inputs.get("x_s", None)
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, F)
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
        if self.debug and not adj is None:
            print("adj =", adj.shape)
            print(util.memory_of(adj))
        if self.debug and not x_s is None:
            print("x_s =", x_s.shape)
            print(util.memory_of(x_s))
        # Start forward
        x = torch.transpose(x, 1, -1) # shape=(N, F, |V|, T)
        if self.pad_len:
            x = torch.cat((torch.zeros((n_sample, n_predictor, n_spatial, self.pad_len), device=x.device), x), -1)
        if self.debug:
            print("MTGNN Input =", x.shape)
            print(util.memory_of(x))
        if self.gcn_true: 
            if not self.buildA_true and adj is None:
                raise ValueError(
                    (
                        "MTGNN set to perform graph convolution without infering a graph topology but no"
                        "adjacency matrix \"adj\" was given as input."
                    )
                )
            if not self.buildA_true:
                self.model.predefined_A = adj
        if self.use_spatial_features and not x_s is None:
            self.model.gc.static_feat = x_s
        a = self.model.forward(x) # shape=(N, T', |V|, 1)
        if self.debug:
            print("MTGNN Output =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def pull_model_data(self, dataset, partition, var):
        princ_data = dataset.get(var.principle_data_type)
        data = {}
        data["__sampled__"] = ["x", "y"]
        data["__outputs__"] = ["y"]
        data["x"] = princ_data.transformed.get(var.principle_data_form).get("predictor_features", partition)
        data["y"] = princ_data.transformed.get(var.principle_data_form).get("response_features", partition)
        if not dataset.spatial.is_empty():
            try:
                data["x_s"] = dataset.spatial.transformed.original.get("predictor_features", partition)
            except:
                data["x_s"] = dataset.spatial.transformed.original.get("numerical_features", partition)
        if not dataset.graph.is_empty():
            if var.graph.edge_weight_feature is None:
                data["adj"] = dataset.graph.original.get("A", partition)
            elif var.graph.edge_weight_feature == "weight":
                data["adj"] = dataset.graph.original.get("W", partition)
            else:
                raise ValueError(
                    "MTGNN can only accept adjacency matrix A or weight matrix W denoted by edge_weight_feature=%s or edge_weight_feature=%s respectively" % (str(None), "weight")
                )
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    hyp_var = var.models.get(model_name()).hyperparameters
    model = MTGNN(
        spatmp.misc.n_predictor, 
        var.mapping.temporal_mapping[0], 
        var.mapping.temporal_mapping[1], 
        spatmp.original.get("n_spatial", "train"), 
        None, 
        None, 
        hyp_var.gcn_true, 
        hyp_var.build_adj, 
        hyp_var.gcn_depth, 
        hyp_var.dropout, 
        hyp_var.subgraph_size, 
        hyp_var.node_dim, 
        hyp_var.dilation_exponential, 
        hyp_var.conv_channels, 
        hyp_var.residual_channels, 
        hyp_var.skip_channels, 
        hyp_var.end_channels, 
        hyp_var.layers, 
        hyp_var.propalpha, 
        hyp_var.tanhalpha, 
        hyp_var.layer_norm_affline, 
        util.get_device(var.execution.device), 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        # Settings from the paper
        #   General settings (section A.3 and 4.4)
        self.dropout = 0.3
        self.gcn_true = True
        self.build_adj = True
        self.gcn_depth = 2 # Mix-hop propagation layer depth
        self.node_dim = 40 # dimension of node embeddings for graph construction
        self.propalpha = 0.05 # Mix-hop propagation layer retain ratio
        self.tanhalpha = 3 # Saturation rate for tanh() actvation on graph learning layer
        self.layer_norm_affline = True # Found in default arg-values of source code
        #   Settings for single-step forecasting (section A.3.1)
        self.layers = 5
        self.dilation_exponential = 2
        self.conv_channels = 16 # Output channels for graph and temporal convolution modules
        self.residual_channels = 16 # Output channels on starting 1x1 convolution layer
        self.skip_channels = 32 # Output channels for skip layers
        self.end_channels = 64 # Output channels for first layer of the output module
        self.subgraph_size = 20 # Neighbors for each node in learned graph (k in KNN())
        #   Settings for multi-step forecasting (section A.3.2)
        if 0:
            self.layers = 3
            self.dilation_exponential = 1
            self.conv_channels = 32 # Output channels for graph and temporal convolution modules
            self.residual_channels = 32 # Output channels on starting 1x1 convolution layer
            self.skip_channels = 64 # Output channels for skip layers
            self.end_channels = 128 # Output channels for first layer of the output module
        # Notes
        #   Traffic, Solar-Energy, and Electricity (subgraph_size=20) | Exchange-Rate (subgraph_size=8)


class TrainingVariables(Container):

    def __init__(self):
        # Settings from the paper
        #   General settings (section A.3)
        self.optimizer = "Adam"
        self.grad_clip = "norm"
        self.grad_clip_kwargs = {"max_norm": 5}
        self.lr = 1e-3
        self.l2_reg = 1e-4
        self.loss = "L1Loss"
        self.initializer = None
        #   Settings for single-step forecasting (section A.3.1)
        self.n_epochs = 30
        self.mbatch_size = 4
        #   Settings for multi-step forecasting (section A.3.2)
        if 0:
            self.n_epochs = 100
            self.mbatch_size = 64
            # Notes
            #   loss = masked_mae() : a function (in util.py) that computes mae (L1Loss) only on non-nan values
