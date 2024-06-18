import torch
import math
import numpy as np
import datetime as dt
import pickle
import pandas as pd
import time
import signal
import sys
import hashlib
import networkx as nx
import json
import re
import os
import inspect
import scipy


comparator_fn_map = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b, 
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b, 
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b, 
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b, 
    "in": lambda a, b: a in b, 
    "!in": lambda a, b: not a in b, 
}


operator_fn_map = {
    "+": lambda a, b: a + b, 
    "-": lambda a, b: a - b, 
    "*": lambda a, b: a * b, 
    "/": lambda a, b: a / b, 
    "**": lambda a, b: a ** b, 
    "//": lambda a, b: a // b, 
    "%": lambda a, b: a % b, 
}


month_name_map = {
    1: "January", 
    2: "February", 
    3: "March", 
    4: "April", 
    5: "May", 
    6: "June", 
    7: "July", 
    8: "August", 
    9: "September", 
    10: "October", 
    11: "November", 
    12: "December"
}


def get_next_fname(_dir, base_fname, ext=".csv"):
    n = 0
    fnames = os.listdir(_dir)
    for fname in fnames:
        if re.match("^%s\d+\%s$" % (base_fname, ext), fname):
            n += 1
    return "%s%d%s" % (base_fname, n, ext)


def module_of(obj):
    return sys.modules[obj.__module__]


def get_all_subsets(a):
    subsets = []
    for i in range(len(a)):
        subsets += list(list(_) for _ in itertools.combinations(a, i+1))
    return subsets


def subdivide(a, k):
    if k == 0:
        return a
    b = np.convolve(a, [1, 1], "valid") / 2
    c = []
    for i in range(len(b)):
        c += [a[i], b[i]]
    c.append(a[-1])
    return subdivide(c, k-1)


class TimeoutExpired(Exception):
    pass


def input_with_timeout(prompt, timeout, timer=time.monotonic):
    import msvcrt
    # Source : Alex Martelli @ https://stackoverflow.com/questions/2933399/how-to-set-time-limit-on-raw-input/2933423#2933423
    sys.stdout.write(prompt)
    sys.stdout.flush()
    endtime = timer() + timeout
    result = []
    while timer() < endtime:
        if msvcrt.kbhit():
            result.append(msvcrt.getwche()) #XXX can it block on multibyte characters?
            if result[-1] == '\r':
                return ''.join(result[:-1])
        time.sleep(0.04) # just to yield to other processes/threads
    raise TimeoutExpired


# Description:
#   Converts between naming convetions
# Arguments:
#   name - the name to convert
#   orig_conv - convention used for name
#   targ_conv - convention to convert name into
def convert_name_convention(name, orig_conv, targ_conv):
    name_fields, new_name = [], ""
    # parse into fields
    if orig_conv == "Pascal":
        name_fields = re.findall("[a-zA-Z][^A-Z]*", name)
    elif orig_conv == "camel":
        name_fields = re.findall("[a-zA-Z][^A-Z]*", name)
    elif orig_conv == "snake":
        name_fields = name.split("_")
    elif orig_conv == "kebab":
        name_fields = name.split("-")
    else:
        raise NotImplementedError("Unknown naming convertion \"%s\"" % (orig_conv))
    # join fields into new name
    if targ_conv == "Pascal":
        new_name = "".join([name_field[0].upper() + name_field[1:] for name_field in name_fields])
    elif targ_conv == "camel":
        new_name = "".join([name_field[0].upper() + name_field[1:] for name_field in name_fields])
        new_name = new_name[0].lower() + new_name[1:]
    elif targ_conv == "snake":
        new_name = "_".join(name_fields)
    elif targ_conv == "kebab":
        new_name = "-".join(name_fields)
    else:
        raise NotImplementedError("Unknown naming convertion \"%s\"" % (targ_conv))
    return new_name


def get_paths(search_dir, path_regex, recurse=False, files=True, debug=False):
    if not os.path.exists(search_dir):
        raise FileNotFoundError(search_dir)
    paths = []
    for root, dirnames, filenames in os.walk(search_dir):
        if files:
            for filename in filenames:
                if debug:
                    print(path_regex, root, filename)
                if re.match(path_regex, filename):
                    paths.append(os.path.join(root, filename))
        else:
            for dirname in dirnames:
                if debug:
                    print(path_regex, root, dirname)
                if re.match(path_regex, dirname):
                    paths.append(os.path.join(root, dirname))
        if not recurse:
            break
    return paths


def get_choice(items, sort=True):
    if len(items) == 0:
        return None
    elif len(items) == 1:
        return items[0]
    if sort:
        items = sorted(items)
    print("\n".join(["%d : %s" % (i, item) for i, item in enumerate(items)]))
    choice = input("Choice? : ")
    if "," in choice: # choice is list of int
        return [items[int(_)] for _ in choice.split(",")]
    try: # choice may be an int
        return items[int(choice)]
    except:
        pass
    # choice is regex
    _items = []
    for item in items:
        if re.match(choice, item):
            _items.append(item)
    return _items


class Types:

    def is_anything(item):
        return True

    def is_none(item):
        return item is None

    def is_int(item):
        return isinstance(item, int) or isinstance(item, np.integer)

    def is_float(item):
        return isinstance(item, float) or isinstance(item, np.floating)

    def is_numeric(item):
        return Types.is_int(item) or Types.is_float(item)

    def is_nan(item):
        return np.isnan(item)

    def is_string(item):
        return isinstance(item, str)

    def is_iter(item):
        return hasattr(item, "__iter__")

    def is_iter_of(item, types=()):
        if not isinstance(types, tuple):
            types = (types)
        if not Types.is_iter(item):
            return False
        if len(item) == 0: # empty iterable
            return len(types) == 0 # of nothing
        if len(types) == 0: # non-empty iterable cannot be of nothing
            return False
        _ = []
        for _item in item:
            __ = []
            for _type in types:
                try:
                    ___ = isinstance(_item, _type)
                except:
                    ___ = _type(_item)
                __.append(___)
            _.append(any(__))
        return all(_)

    def is_iter_of_numeric(item):
        return Types.is_iter_of(item, (Types.is_numeric,))

    def is_iter_of_str(item):
        return Types.is_iter_of(item, (str,))

    def is_collection(item):
        return not isinstance(item, str) and Types.is_iter(item)

    def is_collection_of(item, types):
        return Types.is_collection(item) and Types.is_iter_of(item, types)

    def is_collection_of_int(item):
        return Types.is_collection_of(item, (Types.is_int,))

    def is_collection_of_float(item):
        return Types.is_collection_of(item, (Types.is_float,))

    def is_collection_of_numeric(item):
        return Types.is_collection_of(item, (Types.is_numeric,))

    def is_collection_of_nan(item):
        return Types.is_collection_of(item, (Types.is_nan,))

    def is_collection_of_collection(item):
        return Types.is_collection_of(item, (Types.is_collection,))

    def is_list(item):
        return isinstance(item, list)

    def is_list_of_strings(item):
        return isinstance(item, list) and len(item) > 0 and all(isinstance(_, str) for _ in item)

    def is_list_of_list(item):
        return isinstance(item, list) and len(item) > 0 and all([isinstance(_, list) for _ in item])

    def is_list_of_dict(item):
        return isinstance(item, list) and len(item) > 0 and all([isinstance(_, dict) for _ in item])


def get_func_args(func):
    args, var_args, kw_args, def_vals = inspect.getargspec(func)
    if "self" in args:
        args.remove("self")
    req_args = args[:-len(def_vals)]
    def_args = dict(zip(list_subtract(list(args), req_args), def_vals))
    return req_args, var_args, kw_args, def_args


def compute_zscore_classes(vals, means, stddevs, z_intervals):
    classes = np.zeros(vals.shape)
    for i in range(len(z_intervals)):
        min_vals, max_vals = means + z_intervals[i][0] * stddevs, means + z_intervals[i][1] * stddevs
        if z_intervals[i][0] < -1:
            lower_mask, upper_mask = vals >= min_vals, vals < max_vals
        elif z_intervals[i][0] == -1 and z_intervals[i][1] == 1:
            lower_mask, upper_mask = vals >= min_vals, vals <= max_vals
        else:
            lower_mask, upper_mask = vals > min_vals, vals <= max_vals
        """
        print("Interval =", z_intervals[i])
        print("min/max =", min_vals[:3], max_vals[:3])
        print("vals =", vals[:3])
        print("lower/upper mask =", lower_mask[:3], upper_mask[:3])
        """
        indices = np.where(np.logical_and(lower_mask, upper_mask))[0]
        if indices.shape[0] > 0:
            classes[indices] = i
    return classes


def compute_zscore_confusion(preds, gts, means, stddevs, normalize=False):
    z_intervals = [[-8, -2], [-2, -1.5], [-1.5, -1], [-1, 1], [1, 1.5], [1.5, 2], [2, 8]]
    pred_classes = compute_zscore_classes(preds, means, stddevs, z_intervals)
    gt_classes = compute_zscore_classes(gts, means, stddevs, z_intervals)
    from sklearn.metrics import confusion_matrix
    normalization = ("true" if normalize else None)
    confusion = confusion_matrix(gt_classes, pred_classes, labels=np.arange(len(z_intervals)), normalize=normalization)
    return confusion


def compute_events(values, means, stddevs):
    intervals = [[-8, -2], [-2, -1.5], [-1.5, -1], [-1, 1], [1, 1.5], [1.5, 2], [2, 8]]
    interval_events_map = {}
    for interval in intervals:
        lower_bounds = means + interval[0] * stddevs
        upper_bounds = means + interval[1] * stddevs
        if interval[0] < 0 and interval[1] < 0:# Below mean
            if interval[0] == -8:# Extreme => (-8,-2)
                extremes_below = np.ma.masked_where(
                    np.logical_not(values < upper_bounds),
                    values
                )
                events = extremes_below
            elif interval[0] == -2:# Severe => [-2,-1.5)
                severes_below = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values >= lower_bounds,
                            values < upper_bounds
                        )
                    ),
                    values
                )
                events = severes_below
            elif interval[0] == -1.5:# Moderate
                moderates_below = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values >= lower_bounds,
                            values < upper_bounds
                        )
                    ),
                    values
                )
                events = moderates_below
        elif interval[0] > 0 and interval[1] > 0:# Above mean
            if interval[1] == 8:# Extreme
                extremes_above = np.ma.masked_where(
                    np.logical_not(values > lower_bounds),
                    values
                )
                events = extremes_above
            elif interval[1] == 2:# Severe
                severes_above = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values > lower_bounds,
                            values <= upper_bounds
                        )
                    ),
                    values
                )
                events = severes_above
            elif interval[1] == 1.5:# Moderate
                moderates_above = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values > lower_bounds,
                            values <= upper_bounds
                        )
                    ),
                    values
                )
                events = moderates_above
        else:
            normals = np.ma.masked_where(
                np.logical_not(
                    np.logical_and(
                        values >= lower_bounds,
                        values <= upper_bounds
                    )
                ),
                values
            )
            events = normals
        interval_events_map[",".join(map(str, interval))] = events
    return interval_events_map


def get_stats(a):
    percentiles = np.nanpercentile(a, [25, 50, 75])
    stat_value_map = {
        "count": np.count_nonzero(~np.isnan(a)),
        "mean": np.nanmean(a),
        "std": np.nanstd(a),
        "min": np.nanmin(a),
        "25%": percentiles[0],
        "50%": percentiles[1],
        "75%": percentiles[2],
        "max": np.nanmax(a)
    }
    stat_value_map["n"] = stat_value_map["count"]
    return stat_value_map


def format_memory(n_bytes):
    if n_bytes == 0:
        return "0B"
    mem_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(n_bytes, 1024)))
    p = math.pow(1024, i)
    s = int(round(n_bytes / p, 2))
    return "%s%s" % (s, mem_name[i])


def memory_of(item):
    if isinstance(item, torch.Tensor):
        n_bytes = item.element_size() * item.nelement()
    else:
        n_bytes = sys.getsizeof(item)
    return format_memory(n_bytes)


def n_sliding_windows(timesteps, length, stride=1, offset=0):
    return (timesteps - length + stride - offset) // stride


def sliding_window_indices(timesteps, length, stride=1, offset=0):
    """ Creates indices that index a set of windows

    Arguments
    ---------
    timesteps: int 
        total time-steps in source data
    length: int 
        time-steps per window
    stride: int
        time-steps between consecutive windows (typically 1)
    offset: int
        time-step where first window begins

    """
    n = n_sliding_windows(timesteps, length, stride, offset)
    return np.tile(np.arange(length), (n, 1)) + stride * np.reshape(np.arange(n), (-1, 1)) + offset


def input_output_window_indices(timesteps, in_length, out_length, horizon=1, stride=1, offset=0):
    """ Creates indices that index a set of input and output windows

    Arguments
    ---------
    timesteps: int
        number of time-steps in source
    in_length: int
        number of time-steps per input window
    out_length: int
        number of time-steps per output window
    horizon: int
        offset (in time-steps) of output windows relative to last time-step of each input window
        see notes below
    stride: int
        number of time-steps between window origins
    offset: int
        time-step where first window begins

    Notes
    -----
    horizon :
        at horizon=1  => input/output window indices are [0, 1, 2]/[3, 4, 5]
        at horizon=3  => input/output window indices are [0, 1, 2]/[5, 6, 7]
        at horizon=-2 => input/output window indices are [0, 1, 2]/[0, 1, 2]

    """
    indices = sliding_window_indices(timesteps, in_length + (horizon - 1) + out_length, stride, offset)
    return indices[:,:in_length], indices[:,-out_length:]


def contiguous_window_indices(n, length, stride=1, offset=0):
    assert length % stride == 0, "Windows cannot be contiguous if stride does not divide length (length % stride != 0)"
    return np.arange(offset, n, length // stride)


def pretty_temporal_labels(temporal_labels, curr_frmt, new_frmt="%m/%d/%Y, %H:%M:%S.%f", subs=[], keep_frmts=[]):
    # Definitions
    frmts = ["%Y", "%m", "%d", "%H", "%M", "%S", "%f"]
    attrs = ["year", "month", "day", "hour", "minute", "second", "microsecond"]
    def edit_new_frmt(new_frmt, redun_frmts):
        for frmt in redun_frmts:
            new_frmt = new_frmt.replace(frmt, "")
        new_frmt = re.sub("^\/+", "", new_frmt) # remove excess leading date seps (/)
        new_frmt = re.sub("\/+,", ",", new_frmt) # remove excess trailing date seps (/)
        new_frmt = re.sub("(:+.$|:+$|.$)", "", new_frmt) # remove excess time seps (:) and/or microsecond sep (.)
        new_frmt = re.sub("(^, |, $)", "", new_frmt) # missing date or time: remove date/time sep (", ")
        return new_frmt
    # Setup
    _type = type(temporal_labels)
    if _type != np.ndarray:
        temporal_labels = np.array(temporal_labels)
    shape = temporal_labels.shape
    temporal_labels = np.reshape(temporal_labels, -1)
    # Check for and remove redundant datetime fields
    _ = np.zeros((len(temporal_labels), len(attrs)))
    for i, label in enumerate(temporal_labels):
        date = dt.datetime.strptime(label, curr_frmt)
        for j, attr in enumerate(attrs):
            _[i,j] = getattr(date, attr)
    redun_index = [j for j in range(7) if np.all(_[:,j] == _[0,j])]
    redun_frmts = [frmts[j] for j in redun_index if not frmts[j] in keep_frmts]
    new_frmt = edit_new_frmt(new_frmt, redun_frmts)
    # Reformat the labels and perform extra substitutions
    _ = []
    for label in temporal_labels:
        _label = dt.datetime.strftime(dt.datetime.strptime(label, curr_frmt), new_frmt)
        for sub in subs:
            _label = re.sub(sub[0], sub[1], _label)
        _.append(_label)
    _ = np.reshape(_, shape)
    # Take down
    if _type == tuple:
        _ = to_tuple(_)
    elif _type == list:
        _ = to_list(_)
    return _


def to_tuple(a):
    return a if not hasattr(a, "__iter__") else tuple(to_tuple(_) for _ in a)


def to_list(a):
    return a if not hasattr(a, "__iter__") else list(to_list(_) for _ in a)


# Construct a - b
def list_subtract(a, b):
    if b is None:
        return []
    return [a_i for a_i in a if a_i not in b]


def list_indices(a, items):
    indices = []
    for item in items:
        indices.append(a.index(item))
    return indices


def dict_to_str(a):
    return json.dumps(a, sort_keys=True, indent=4)


def sort_dict(a, by="key", ascending=True):
    if by == "value":
        return dict(sorted(a.items(), key=lambda item: item[1], reverse=(not ascending)))
    if by == "key":
        b = {}
        keys = sorted(list(a.keys()))
        for key in keys:
            b[key] = a[key]
    return b


def merge_dicts(a, b, overwrite=True):
    if not (isinstance(a, dict) or isinstance(b, dict)):
        raise ValueError("Expected two dictionaries")
    if overwrite:
        return {**a, **b}
    c = copy_dict(a)
    for key, value in b.items():
        if not key in a:
            c[key] = value
    return c


def remap_dict(a, remap_dict, must_exist=False):
    b = copy_dict(a)
    for old_key, new_key in remap_dict.items():
        if old_key in b:
            b[new_key] = b.pop(old_key)
        elif must_exist:
            raise ValueError("Input dict a does not contain key %s" % (str(old_key)))
    return b


def copy_dict(a):
    return {key: value for key, value in a.items()}


def to_key_index_dict(keys, offset=0, stride=1):
    return {key: offset+i for key, i in zip(keys, range(0, len(keys), stride))}


def to_dict(keys, values, repeat=False):
    if repeat:
        return {key: values for key in keys}
    return {key: value for key, value in zip(keys, values)}


def invert_dict(a):
    return {value: key for key, value in a.items()}


def sort_dict(a, by="key"):
    if by == "key":
        return {key: a[key] for key in sorted(a.keys(), key=str.lower)}
    elif by == "value":
        return invert_dict(sort_dict(invert_dict(a)))
    raise ValueError("Unknown sorting option \"%s\"" % (by))


def get_dict_values(a, keys, must_exist=True):
    if must_exist:
        values = [a[_] for _ in keys]
    else:
        values = []
        for key in keys:
            if not key in a:
                continue
            values.append(a[key])
    return values


def filter_dict(a, keys, must_exist=False):
    b = {}
    for key in keys:
        if not key in a and not must_exist:
            continue
        b[key] = a[key]
    return b


def to_cache(data, path):
    if os.sep in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def from_cache(path, **kwargs):
    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def hash_str_to_int(string, n_digits):
    num = int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 10**n_digits
    cur_digits = len(str(num))
    return num * 10**(n_digits - cur_digits)


def make_msg_block(msg, block_char="#"):
    msg_line = 3*block_char + " " + msg + " " + 3*block_char
    msg_line_len = len(msg_line)
    msg_block = "%s\n%s\n%s" % (
        msg_line_len*block_char,
        msg_line,
        msg_line_len*block_char
    )
    return msg_block


def get_device(device="cuda"):
    if "cuda:-" in device and torch.cuda.is_available():
        device = "cuda:%d" % (torch.cuda.device_count() + int(device.split(":")[1]))
    return torch.device(device if torch.cuda.is_available() else "cpu")


def to_device(items, device):
    if isinstance(items, list): # Multiple items to put onto device
        device_items = []
        for item in items:
            if isinstance(item, torch.Tensor) or isinstance(item, torch.nn.Module):
                device_items += [item.to(device)]
            else:
                device_items += [item]
        return device_items
    else: # Single item to put onto device
        return to_device([items], device)[0]


def to_tensor(items, types):
    if not isinstance(items, list): # Single item
        return to_tensor([items], types)[0]
    # Convert each data in items to a tensor
    tensors = []
    for i in range(len(items)):
        item = items[i]
        if isinstance(types, list):
            _type = types[i]
        else:
            _type = types
        if _type is None: # ignore
            tensors.append(item)
            continue
        if isinstance(item, torch.Tensor): # No need to convert
            tensors.append(item)
        else:
            tensors.append(torch.tensor(item, dtype=_type))
    return tensors


def to_ndarray(items):
    if not isinstance(items, list): # Single item
        return to_ndarray([items])[0]
    # Convert each data in items to a tensor
    ndarrays = []
    for item in items:
        if isinstance(item, torch.Tensor):
            ndarrays += [item.detach().cpu().numpy()]
        elif isinstance(item, np.ndarray):
            ndarrays += [item]
        elif isinstance(item, list):
            ndarrays += [np.array(item)]
        else:
            raise NotImplementedError(
                "Do not know how to convert %s of type %s to a NumPy.ndarray" % (
                    item,
                    type(item)
                )
            )
    return ndarrays


def resolution_to_delta(res):
    return dt.timedelta(**{res[1]: res[0]})


def days_in_month(year, month):
    return (dt.datetime(year + month // 12, month % 12 + 1, 1) - dt.datetime(year, month, 1)).days


def datetime_range(start, end, delta):
    datetimes = []
    curr = start
    while curr <= end:
        datetimes.append(curr)
        curr += delta
    return datetimes


def generate_temporal_labels(start, end, delta, frmt="%Y-%m-%d_%H-%M-%S", incl=[True, False]):
    """ Generates a chronologically ordered set of string-formatted time-stamps for a given range

    Arguments
    ---------
    start : str or datetime
    end : str, int, or datetime
    delta : list of [int, str] or timedelta
    frmt : str
    incl : list of [bool, bool]

    Returns
    -------
    temporal_labels : list of str

    """
    # Check arguments
    if not (isinstance(start, str) or isinstance(start, dt.datetime)):
        raise ValueError("Argument \"start\" must be str or datetime. Received %s" % (type(start)))
    if not (isinstance(end, str) or isinstance(end, int) or isinstance(end, dt.datetime)):
        raise ValueError("Argument \"end\" must be str, int, or datetime. Received %s" % (type(end)))
    if not (isinstance(delta, list) or isinstance(delta, dt.timedelta)):
        raise ValueError("Argument \"delta\" must be list or timedelta. Received %s" % (type(delta)))
    elif isinstance(delta, list): # delta given as custom format [int, str] (e.g. [7, "days"])
        if not (isinstance(delta[0], int) and isinstance(delta[1], str)):
            raise ValueError("Argument \"delta\" as list must follow format [int, str]. Received %s" % (str(delta)))
        elif delta[0] < 1:
            raise ValueError("Number of time-steps in delta=[int str] must be positive. Received %d" % (delta[0]))
        elif delta[1] not in ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]:
            raise NotImplementedError("Unknown resolution=%s in argument \"delta\"" % (delta[1]))
    # Start
    temporal_labels = []
    if isinstance(start, str): # start given as date str
        start = dt.datetime.strptime(start, frmt)
    if isinstance(end, str): # end given as date str
        end = dt.datetime.strptime(end, frmt)
    if isinstance(delta, list): # delta given as custom format [int, str] (e.g. [7, "days"])
        time_steps, resolution = delta
        if resolution in ["months", "years"]:
            if isinstance(end, int): # end given as number of time-steps from start
                year, month, day = start.year, start.month, start.day
                hour, minute, second = start.hour, start.minute, start.second
                if resolution == "months":
#                    if day > 28:
#                        raise ValueError("Month denominated delta undefined for day > 28")
                    month += end * time_steps
                    year += (month - 1) // 12
                    month = (month - 1) % 12 + 1
                elif resolution == "years":
                    year += end * time_steps
                day = min(day, days_in_month(year, month))
                end = dt.datetime(year, month, day, hour, minute, second)
            # Start creating temporal labels at k*month/k*year resolution
            curr = start
            while curr <= end:
                temporal_labels.append(curr.strftime(frmt))
                year, month, day = curr.year, curr.month, start.day
                hour, minute, second = start.hour, start.minute, start.second
                if resolution == "months":
                    month += time_steps
                    year += (month - 1) // 12
                    month = (month - 1) % 12 + 1
                elif resolution == "years":
                    year += time_steps
                day = min(day, days_in_month(year, month))
                curr = dt.datetime(year, month, day, hour, minute, second)
        else: # resolution is one of ("seconds", "minutes", "hours", "days", "weeks")
            delta = dt.timedelta(**{resolution: time_steps})
            if isinstance(end, int): # end given as number of time-steps from start
                end = start + end * delta
            temporal_labels = [dt.strftime(frmt) for dt in datetime_range(start, end, delta)]
    elif isinstance(delta, dt.timedelta):
        if isinstance(end, int): # end given as number of time-steps from start
            end = start + end * delta
        temporal_labels = [dt.strftime(frmt) for dt in datetime_range(start, end, delta)]
    if not incl[0]:
        temporal_labels = temporal_labels[1:]
    if not incl[1]:
        temporal_labels = temporal_labels[:-1]
    return temporal_labels


def temporal_labels_to_periodic_indices(labels, period, resolution, frmt="%Y-%m-%d_%H-%M-%S"):
    """ Converts a chronologically ordered set of string-formatted time-stamps into a set of recurring index locations according to moments of a given period.

    Arguments
    ---------
    labels : tuple, list, or ndarray of str
    period : tuple or list 2-tuple of (int, str)
    resolution : tuple or list 2-tuple of (int, str)
    frmt : str

    Returns
    -------
    temporal_labels : tuple, list, or ndarray of str

    """
    # Check arguments
    supported_units = [
        "microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "weeks", "months", "years"
    ]
    if not period[1] in supported_units:
        raise NotImplementedError(period)
    if not resolution[1] in supported_units:
        raise NotImplementedError(resolution)
    if supported_units.index(resolution[1]) > supported_units.index(period[1]) or (resolution[1] == period[1] and resolution[0] > period[0]):
        raise ValueError("Argument \"period\" must be a time-span that encompasses the time-span of argument \"resolution\". Received period=%s and resolution=%s" % (str(period), str(resolution)))
    # Start conversion
    if isinstance(labels, np.ndarray):
        original_shape = labels.shape
        labels = np.reshape(labels, (-1,))
        indices = np.zeros(labels.shape, dtype=int)
    elif isinstance(labels, list):
        indices = list(0 for _ in labels)
    elif isinstance(labels, tuple):
        indices = tuple(0 for _ in labels)
    elif isinstance(labels, pd.Series):
        indices = pd.Series([0 for _ in labels])
    #   Handle special cases
    if resolution[1] == "days" and period[1] == "days": # day -> day-of-days zero-indexed to first Monday
        unit_delta = dt.timedelta(days=resolution[0])
        period_units = period[0]
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            units_elapsed = (date - dt.datetime.min) // unit_delta
            indices[i] = units_elapsed % period_units
    elif resolution[1] == "months":
        period_units = period[0] # month -> month-of-months
        if period[1] == "years": # month -> month-of-year(s)
            period_units *= 12
        period_units = period_units / resolution[0]
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            indices[i] = (12 * date.year + (date.month-1)) % period_units
    elif resolution[1] == "years":
        period_units = period[0] # year -> year-of-years
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            indices[i] = date.year % period_units
    elif period[1] == "weeks":
        if resolution[1] == "microseconds": # microsecond -> microsecond-of-the-week(s)
            unit_delta = dt.timedelta(microseconds=resolution[0])
            period_units = 1000 * 1000 * 60 * 60 * 24 * 7 * period[0] // resolution[0]
        elif resolution[1] == "milliseconds": # millisecond -> millisecond-of-the-week(s)
            unit_delta = dt.timedelta(milliseconds=resolution[0])
            period_units = 1000 * 60 * 60 * 24 * 7 * period[0] // resolution[0]
        elif resolution[1] == "seconds": # second -> second-of-the-week(s)
            unit_delta = dt.timedelta(seconds=resolution[0])
            period_units = 60 * 60 * 24 * 7 * period[0] // resolution[0]
        elif resolution[1] == "minutes": # minute -> minute-of-the-week(s)
            unit_delta = dt.timedelta(minutes=resolution[0])
            period_units = 60 * 24 * 7 * period[0] // resolution[0]
        elif resolution[1] == "hours": # hour -> hour-of-the-week(s)
            unit_delta = dt.timedelta(hours=resolution[0])
            period_units = 24 * 7 * period[0] // resolution[0]
        elif resolution[1] == "days": # day -> day-of-the-week(s)
            unit_delta = dt.timedelta(days=resolution[0])
            period_units = 7 * period[0] // resolution[0]
        elif resolution[1] == "weeks": # week -> week-of-the-weeks
            unit_delta = dt.timedelta(weeks=resolution[0])
            period_units = period[0]
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            units_elapsed = (date - dt.datetime.min) // unit_delta
            indices[i] = units_elapsed % period_units
    else:
        unit_delta = dt.timedelta(**{resolution[1]: resolution[0]})
        period_units = period[0]
        period_unit_name = period[1][:-1]
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            period_origin_kwargs = {
                "year": 1, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0
            }
            for unit_name in period_origin_kwargs.keys():
                unit_value = getattr(date, unit_name)
                period_origin_kwargs[unit_name] = unit_value
                if unit_name == period_unit_name:
                    unit_origin_value = (unit_value // period_units) * period_units
                    if unit_name in ["day", "month", "year"]: # handle one-based units
                        unit_origin_value = ((unit_value - 1) // period_units) * period_units + 1
                    period_origin_kwargs[unit_name] = unit_origin_value
                    break
            period_origin_date = dt.datetime(**period_origin_kwargs)
            indices[i] = (date - period_origin_date) // unit_delta
    if isinstance(labels, np.ndarray):
        labels = np.reshape(labels, original_shape)
        indices = np.reshape(indices, original_shape)
    return indices


def get_period_size(period, resolution):
    per_n, per_unit = period
    res_n, res_unit = resolution
    if per_unit == "years":
        if res_unit == "years":
            pass
        elif res_unit == "months":
            per_n *= 12
        elif res_unit == "weeks":
            per_n *= 53
        elif res_unit == "days":
            per_n *= 366
        elif res_unit == "hours":
            per_n *= 366 * 24
        elif res_unit == "minutes":
            per_n *= 366 * 24 * 60
        elif res_unit == "seconds":
            per_n *= 366 * 24 * 60 * 60
    elif per_unit == "months":
        if res_unit == "months":
            pass
        elif res_unit == "weeks":
            per_n *= 5
        elif res_unit == "days":
            per_n *= 31
        elif res_unit == "hours":
            per_n *= 31 * 24
        elif res_unit == "minutes":
            per_n *= 31 * 24 * 60
        elif res_unit == "seconds":
            per_n *= 31 * 24 * 60 * 60
    elif per_unit == "weeks":
        if res_unit == "weeks":
            pass
        elif res_unit == "days":
            per_n *= 7
        elif res_unit == "hours":
            per_n *= 7 * 24
        elif res_unit == "minutes":
            per_n *= 7 * 24 * 60
        elif res_unit == "seconds":
            per_n *= 7 * 24 * 60 * 60
    elif per_unit == "days":
        if res_unit == "days":
            pass
        elif res_unit == "hours":
            per_n *= 24
        elif res_unit == "minutes":
            per_n *= 24 * 60
        elif res_unit == "seconds":
            per_n *= 24 * 60 * 60
    elif per_unit == "hours":
        if res_unit == "hours":
            pass
        elif res_unit == "minutes":
            per_n *= 60
        elif res_unit == "seconds":
            per_n *= 60 * 60
    elif per_unit == "minutes":
        if res_unit == "minutes":
            pass
        elif res_unit == "seconds":
            per_n *= 60
    elif per_unit == "seconds":
        pass
    else:
        pass
    return per_n // res_n


def labels_to_ids(labels):
    orig_shape = labels.shape
    labels = np.reshape(labels, [-1])
    unique_labels = np.sort(np.unique(labels))
    label_index_map = to_key_index_dict(unique_labels)
    return np.reshape(np.array([label_index_map[label] for label in labels]), orig_shape)
