import os
import sys
import time
import torch

import util
from container import Container
from model import Model

from torch_geometric_temporal.nn.recurrent.temporalgcn import TGCN2


class TGCN(Model):

    def __init__(
        self, 
        in_channels, # in_channels (int) - Number of input features.
        n_temporal_out, # n_temporal_out (int) - Number of output time-steps or forecast length..
        hidden_dim, # hidden_dim (int) - Number of dimensions for hidden state of GRU
        improved=False, # improved (bool) - Stronger self loops. Default is False.
        cached=False, # cached (bool) - Caching the message weights. Default is False.
        add_self_loops=True, # add_self_loops (bool) - Adding self-loops for smoothing. Default is True.
    ):
        super(TGCN, self).__init__()
        self.tgcn_cell = TGCN2(in_channels, hidden_dim, 1, improved, cached, add_self_loops)
        self.out_proj = torch.nn.Linear(hidden_dim, n_temporal_out)
        self.out_proj_act = torch.nn.Identity()
        # Save all vars
        self.in_channels = in_channels
        self.n_temporal_out = n_temporal_out
        self.hidden_dim = hidden_dim
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["hidden_dim", None], 
            ["improved", None], 
            ["add_self_loops", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        edge_index = inputs["edge_index"]
        edge_weight = inputs.get("edge_weight", None)
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, F)
        # Start forward
        if self.debug:
            print("x =", x.shape, "=")
        if self.debug:
            print("edge_index =", edge_index.shape, "=")
        if self.debug and not edge_weight is None:
            print("edge_weight =", edge_weight.shape, "=")
        hidden_state = None # shape=(N, |V|, H)
        for i in range(n_temporal_in):
            hidden_state = self.tgcn_cell(x[:,i,:,:], edge_index, edge_weight, hidden_state)
            if self.debug:
                print("hidden_state[%d] =" % (i), hidden_state.shape, "=")
                if self.debug > 1:
                    print(hidden_state)
        z = self.out_proj(hidden_state) # shape=(N, |V|, T')
        a = self.out_proj_act(z)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        a = torch.unsqueeze(torch.transpose(a, 1, 2), -1) # shape=(N, T', |V|, 1)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        self.tgcn_cell.conv_z.reset_parameters()
        self.tgcn_cell.linear_z.reset_parameters()
        self.tgcn_cell.conv_r.reset_parameters()
        self.tgcn_cell.linear_r.reset_parameters()
        self.tgcn_cell.conv_h.reset_parameters()
        self.tgcn_cell.linear_h.reset_parameters()
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
            else:
                data["edge_weight"] = dataset.spatiotemporal.filter_axis(
                dataset.spatiotemporal.original.get("features", partition),
                    [-2, -1],
                    [
                        data["edge_index"][0,:],
                        dataset.spatiotemporal.misc.feature_index_map[var.graph.edge_weight_feature]
                    ]
                )
                data["__sampled__"].append("edge_weight")
        return data


def init(dataset, var):
    spatiotemporal = dataset.spatiotemporal
    hyp_var = var.models.get(model_name()).hyperparameters
    model = TGCN(
        spatiotemporal.misc.n_predictor, 
        var.mapping.temporal_mapping[1], 
        hyp_var.hidden_dim, 
        hyp_var.improved, 
        hyp_var.cached, 
        hyp_var.add_self_loops, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/1811.05320.pdf from section 4.3 - "Model Parameters Designing"
#       ::: hidden_dim, lr, n_epoch, optimizer, mbatch_size, loss
#   2. The repository @ https://github.com/martinwhl/T-GCN-PyTorch from "Model Training"
#       ::: l2_reg
#   3. The library PyTorch Geometric Temporal @ https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/recurrent/temporalgcn.html#TGCN
#       ::: everything else
class HyperparameterVariables(Container):

    def __init__(self):
        self.hidden_dim = 100 # for SZ-Taxi
        self.hidden_dim = 64 # for Los-Loop
        self.improved = False
        self.cached = False
        self.add_self_loops = True


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 3000 # set as max_epochs in repo to be used with (I assume) early stopping
        self.n_epochs = 50
        self.mbatch_size = 64
        self.lr = 0.001
        self.l2_reg = 1.5e-3 # SZ-Taxi
        self.l2_reg = 0.0 # Los-Loop
        self.optimizer = "Adam"
        self.loss = "MSELoss"
