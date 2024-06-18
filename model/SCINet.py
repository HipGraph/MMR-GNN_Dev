import os
import sys
import time
import torch
import warnings

import util
from container import Container
from model import Model

from model.SCINet_PyTorch.models.SCINet import SCINet as _SCINet


class SCINet(Model):

    def __init__(
        self, 
        output_len, 
        input_len, 
        input_dim, 
        hid_size=1, 
        num_stacks=1,
        num_levels=2, 
        num_decoder_layer=1, 
        concat_len=0, 
        groups=1, 
        kernel=5, 
        dropout=0.5,
        single_step_output_One=0, 
        input_len_seg=0, 
        positionalE=False, 
        modified=True, 
        RIN=False, 
    ):
        super(SCINet, self).__init__()
        pad_len = 0
        if input_len % 2**num_levels:
            pad_len = (input_len // 2**num_levels + 1) * 2**num_levels - input_len
            warnings.warn("SCINet: found input time-steps (Ti) is not evenly divided by 2^num_levels (Ti %% 2^num_levels != 0). Specifically, %d %% %d != 0 for Ti=%d and num_levels=%d. Will continue by feeding inputs zero-padded to %d time-steps." % (input_len, 2**num_levels, input_len, num_levels, input_len+pad_len))
        self.model = _SCINet(
            output_len, 
            input_len+pad_len, 
            input_dim, 
            hid_size, 
            num_stacks, 
            num_levels, 
            num_decoder_layer, 
            concat_len, 
            groups, 
            kernel, 
            dropout, 
            single_step_output_One, 
            input_len_seg, 
            positionalE, 
            modified, 
            RIN, 
        )
        # Save all vars
        self.pad_len = pad_len
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["hid_size", None], 
            ["num_stacks", None], 
            ["num_levels", None], 
            ["num_decoder_layer", None], 
            ["concat_len", None], 
            ["groups", None], 
            ["kernel", None], 
            ["dropout", None], 
            ["single_step_output_One", None], 
            ["input_len_seg", None], 
            ["positionalE", None], 
            ["modified", None], 
            ["RIN", None], 
        ]

    def forward(self, inputs):
#        self.debug = 1
        # Handle args
        x = inputs["x"]
        n_sample, n_temporal_in, n_spatial, n_predictor = x.shape # shape=(N, T, |V|, 1)
        # Start forward
        if self.debug:
            print("x =", x.shape, "=")
            if self.debug > 1:
                print(x)
        x = torch.squeeze(x, -1) # shape=(N, T, |V|)
        if self.pad_len:
            x = torch.cat((torch.zeros((n_sample, self.pad_len, n_spatial), device=x.device), x), 1)
        if self.debug:
            print("x reshaped =", x.shape, "=")
            if self.debug > 1:
                print(x)
        a = self.model(x) # shape=(N, T', |V|)
        if self.debug:
            print("a =", a.shape, "=")
            if self.debug > 1:
                print(a)
        a = torch.unsqueeze(a, -1) # shape=(N, T, |V|)
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
        # Acquired from class SCINet in script models/SCINet.py in repository @ https://github.com/cure-lab/SCINet
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.bias.data.zero_()

    def verify_pulled_data(self, data, var):
        if data["x"].shape[-1] > 1 or data["y"].shape[-1] > 1:
            raise ValueError(
                "SCINet only capable of uni-variate inference but \"x\" or \"y\" were multi-variate with x.shape=%s and y.shape=%s" % (
                    str(data["x"].shape), str(data["y"].shape)
                )
            )
        return data


def init(dataset, var):
    spatmp = dataset.spatiotemporal
    graph = dataset.graph
    hyp_var = var.models.get(model_name()).hyperparameters
    model = SCINet(
        var.mapping.temporal_mapping[1], 
        var.mapping.temporal_mapping[0], 
        spatmp.original.get("n_spatial", "train"), 
        hyp_var.hid_size, 
        hyp_var.num_stacks, 
        hyp_var.num_levels, 
        hyp_var.num_decoder_layer, 
        hyp_var.concat_len, 
        hyp_var.groups, 
        hyp_var.kernel, 
        hyp_var.dropout, 
        hyp_var.single_step_output_One, 
        hyp_var.input_len_seg, 
        hyp_var.positionalE, 
        hyp_var.modified, 
        hyp_var.RIN, 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Settings found in:
#   1. The paper @ https://arxiv.org/pdf/2106.09305.pdf under APPENDIX C "Reproducibility" table 13 [PEMS04]
#       ::: mbatch_size, lr, hid_size, k, dropout, num_levels, num_stacks
#   2. The script run_pems.py in repository @ https://github.com/cure-lab/SCINet
#       ::: n_epochs, l2_reg, single_step_output_One, positionalE, RIN 
#   3. The script experiments/exp_pems.py in repository @ https://github.com/cure-lab/SCINet
#       ::: optimizer, lr_scheduler, lr_scheduler_kwargs, modified
#   4. The class SCINet from script models/SCINet.py in repository @ https://github.com/cure-lab/SCINet
#       ::: input_len_seg
class HyperparameterVariables(Container):

    def __init__(self):
        self.hid_size = 0.0625
        self.num_stacks = 1
        self.num_levels = 2
        self.num_decoder_layer = 1
        self.concat_len = 0
        self.groups = 1
        self.kernel = 5
        self.dropout = 0.0
        self.single_step_output_One = 0
        self.input_len_seg = 0
        self.positionalE = True
        self.modified = True
        self.RIN = False


class TrainingVariables(Container):

    def __init__(self):
        self.n_epochs = 80
        self.lr = 1e-3
        self.lr_scheduler = "StepLR"
        self.lr_scheduler_kwargs = {"step_size": 5, "gamma": 0.5}
        self.mbatch_size = 8
        self.optimizer = "Adam"
        self.l2_reg = 1e-5
