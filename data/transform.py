import numpy as np
import torch

import util


def minmax_transform(A, minimum, maximum, **kwargs):
    a = kwargs.get("a", -1)
    b = kwargs.get("b", 1)
    clip = kwargs.get("clip", False)
    revert = kwargs.get("revert", False)
    if revert:
        if clip:
            A = np.clip(A, a, b)
        A = ((A - a) / (b - a)) * (maximum - minimum) + minimum
    else:
        A = (b - a) * ((A - minimum) / (maximum - minimum)) + a
        if clip:
            A = np.clip(A, a, b)
    return A


def zscore_transform(A, mean, std, **kwargs):
    revert = kwargs.get("revert", False)
    if isinstance(std, np.ndarray) or isinstance(std, torch.Tensor):
        if std.shape == ():
            if std == 0:
                std = 1.0
        else:
            std[std == 0] = 1.0
    elif isinstance(std, float) or isinstance(std, int):
        if std == 0:
            std = 1
    if revert:
        return A * std + mean
    return (A - mean) / std


def log_transform(A, _1, _2, **kwargs):
    revert = kwargs.get("revert", False)
    if revert:
        return np.exp(A)
    A[A == 0] = 1 + 1e-5
    return np.log(A)


def root_transform(A, _1, _2, **kwargs):
    p = kwargs.get("p", 2)
    revert = kwargs.get("revert", False)
    if revert:
        return A**p
    return A**(1/p)


def index_transform(A, _1, _2, **kwargs):
    pass


def identity_transform(A, _1, _2, **kwargs):
    return A


def transform(args, transform, revert=False):
    transform_fn_map = {
        "minmax": minmax_transform, 
        "zscore": zscore_transform, 
        "log": log_transform, 
        "root": root_transform, 
        "identity": identity_transform, 
    }
    name, kwargs = transform, {"revert": revert} # case where transform is str and no kwargs
    if isinstance(transform, dict): # case where transform is dict and specifies kwargs
        _transform = util.copy_dict(transform)
        name = _transform.pop("name")
        kwargs = _transform
        kwargs["revert"] = revert
    return transform_fn_map[name](*args, **kwargs)


class LabelEncoder:

    def __init__(self):
        pass

    def fit(self, labels):
        self.classes_ = np.unique(labels)
        self.label_encoding_map = util.to_key_index_dict(self.classes_)
        return self

    def transform(self, labels, sparse=True):
        if isinstance(labels, str): # Single label
            labels = [labels]
        index = util.get_dict_values(self.label_encoding_map, labels)
        if not sparse:
            return np.eye(len(self.label_encoding_map))[index]
        return index
