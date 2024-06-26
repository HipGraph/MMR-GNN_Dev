import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

import util
from plotting import Plotting
from container import Container
from arguments import ArgumentParser


def parse_config(contents, con=None):
    return parse_vars(contents, con)


def parse_errors(contents, con=None):
    if con is None: con = Container()
    partition, partition_error = None, None
    feature_label, feature_error = None, None
    spatial_label, spatial_error = None, None
    lines = contents.split("\n")
    for i, line in enumerate(lines):
        fields = line.split()
        if line.startswith("\t\t"): # error line for a spatial element
            if feature_label is None or feature_error is None:
                raise IOError("Failed to parse feature error line: \n%s" % (line))
            if fields[-1] in ["--"]:
                spatial_label, error_name, spatial_error = " ".join(fields[1:-3]), fields[-3], float('nan')
            else:
                spatial_label, error_name, spatial_error = " ".join(fields[1:-3]), fields[-3], float(fields[-1])
            con.get(error_name, partition).get(feature_label)[spatial_label] = spatial_error
        elif line.startswith("\t"): # error line for a feature
            if partition is None or partition_error is None:
                raise IOError("Failed to parse partition error line: \n%s" % (line))
            if not feature_label is None:
                con.get(error_name, partition).set(feature_label, feature_error)
            feature_label, error_name, feature_error = fields[0], fields[1], float(fields[-1])
            con.get(error_name, partition).set(feature_label, {})
            spatial_label, spatial_error = None, None
        elif len(line) > 0 and not line.startswith(("#", " ")): # error line for a partition
            if not partition is None: # previous partition line
                if feature_label is None: # no feature errors
                    con.set(error_name, partition_error, partition)
                elif spatial_label is None: # feature error lines but no spatial error lines
                    con.get(error_name, partition).set(feature_label, feature_error)
            partition, error_name, partition_error = fields[0], fields[1], float(fields[-1])
            con.set(error_name, Container(), partition)
            feature_label, feature_error = None, None
            spatial_label, spatial_error = None, None
        else:
            partition, partition_error = None, None
            feature_label, feature_error = None, None
            spatial_label, spatial_error = None, None
    if not partition is None: # previous partition line
        if feature_label is None: # no feature errors
            con.set(error_name, partition_error, partition)
        elif spatial_label is None: # feature error lines but no spatial error lines
            con.get(error_name, partition).set(feature_label, feature_error)
    return con


def parse_eval(contents, con=None):
    return contents


def parse_info(contents, con=None):
    return parse_vars(contents, con)


def parse_setting(contents, con=None):
    return parse_vars(contents, con)


def parse_vars(contents, con=None):
    if con is None: con = Container()
    lines = contents.split("\n")
    argv = []
    for i in range(len(lines)):
        if lines[i] == "": # empty lines can get added at end depending on OS
            continue
        argv += lines[i].split(" = ")
    ArgumentParser().parse_arguments(argv, con)
    return con


def get_config(path, con=None):
    return get_vars(path, con)


def get_errors(path, con=None):
    if con is None: con = Container()
    if os.path.exists(path):
        with open(path, "r") as f:
            parse_errors(f.read(), con)
    return con


def mp_get_errors(path, con=None, pid=0, ret_dict={}):
    if con is None: con = Container()
    if os.path.exists(path):
        with open(path, "r") as f:
            ret_dict[pid] = parse_errors(f.read(), con)


def get_eval(path, con=None):
    return parse_eval(util.from_cache(path), con)


def get_infos(path, con=None):
    return get_vars(path, con)


def get_settings(path, con=None):
    return get_vars(path, con)


def get_vars(path, con=None):
    if con is None: con = Container()
    if os.path.exists(path):
        with open(path, "r") as f:
            parse_vars(f.read(), con)
    return con


def get_all_errors(path, con=None, method="efficient", chkpt="Best"):
    if con is None: con = Container()
    if method == "efficient":
        paths = get_paths(path, "Performance_Checkpoint[%s].txt" % (chkpt))
    else:
        paths = util.get_paths(path, "^Performance_Checkpoint\[%s]\.txt$" % (chkpt), recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        if not model_name in con:
            con.set(model_name, Container())
        try:
            con.get(model_name).set(model_id, get_errors(path))
        except Exception as e:
            print(util.make_msg_block("Exception: \"%s\" at Model/ID: %s/%s" % (str(e), model_name, model_id)))
            raise e
    return con


def mp_get_all_errors(path, con=None, method="efficient", chkpt="Best", n_proc=100):
    if con is None: con = Container()
    if method == "efficient":
        paths = get_paths(path, "Performance_Checkpoint[%s].txt" % (chkpt))
    else:
        paths = util.get_paths(path, "^Performance_Checkpoint\[%s]\.txt$" % (chkpt), recurse=True)
    N = len(paths)
    manager = mp.Manager()
    ret_dict = manager.dict()
    n_passes = N // n_proc + 1 if N % n_proc else N // n_proc
    model_names, model_ids = [], []
    for i in range(n_passes):
        jobs = []
        for j in range(n_proc):
            pid = i * n_proc + j
            if pid >= N:
                break
            model_name, model_id = paths[pid].split(os.sep)[-3:-1]
            model_names.append(model_name)
            model_ids.append(model_id)
            if not model_name in con:
                con.set(model_name, Container())
            con.get(model_name).set(model_id, Container())
            p = mp.Process(
                target=mp_get_errors, 
                args=(paths[pid], con.get(model_name).get(model_id), pid, ret_dict), 
            )
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
    for pid in range(N):
        model_name, model_id = model_names[pid], model_ids[pid]
        con.get(model_name).set(model_id, ret_dict[pid])
    return con


def get_all_evals(path, con=None, method="efficient"):
    if con is None: con = Container()
    if method == "efficient":
        paths = get_paths(path, "Evaluation_Partition[train].pkl")
        paths += get_paths(path, "Evaluation_Partition[valid].pkl")
        paths += get_paths(path, "Evaluation_Partition[test].pkl")
    else:
        paths = util.get_paths(path, "^Evaluation_Partition\[(train|valid|test)]\.pkl", recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        fname = os.path.basename(path)
        if not model_name in con:
            con.set(model_name, Container())
        if not model_id in con.get(model_name):
            con.get(model_name).set(model_id, Container())
        partition = fname.replace("Evaluation_Partition[", "").replace("].pkl", "")
        eval_var = get_eval(path)
        con.set("var", eval_var.var, path=[model_name, model_id])
        con.set("yhat", eval_var.Yhat, partition, path=[model_name, model_id])
    return con


def get_all_infos(path, con=None, method="efficient"):
    if con is None: con = Container()
    if method == "efficient":
        paths = get_paths(path, "ModelInfo.txt")
        paths += get_paths(path, "OptimizationInfo.txt")
    else:
        paths = util.get_paths(path, "^.*Info\.txt$", recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        if not model_name in con:
            con.set(model_name, Container())
        if not model_id in con.get(model_name):
            con.get(model_name).set(model_id, Container())
        con.get(model_name).get(model_id).copy(get_infos(path))
    return con


def get_all_settings(path, con=None, method="efficient"):
    if con is None: con = Container()
    if method == "efficient":
        paths = []
        paths += get_paths(path, "DataSettings.txt")
        paths += get_paths(path, "HyperparameterSettings.txt")
        paths += get_paths(path, "OptimizationSettings.txt")
    else:
        paths = util.get_paths("^.*Settings\.txt", recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        if not model_name in con:
            con.set(model_name, Container())
        if not model_id in con.get(model_name):
            con.get(model_name).set(model_id, Container())
        con.get(model_name).get(model_id).copy(get_settings(path))
    return con


def get_cache(eval_dir, chkpt_dir, load_evals=False, con=None, method="efficient", n_proc=100, debug=0):
    if con is None: con = Container()
    s = time.time()
    if n_proc == 1:
        con.errors = get_all_errors(eval_dir, method=method)
    else:
        con.errors = mp_get_all_errors(eval_dir, method=method, n_proc=n_proc)
    if debug:
        print("%.2fs" % (time.time()-s))
    con.settings = get_all_settings(chkpt_dir, method=method)
    con.infos = get_all_infos(chkpt_dir, method=method)
    if load_evals:
        con.evals = get_all_evals(eval_dir, method=method)
    return con


def get_paths(root_dir, fname):
    paths = []
    model_names = os.listdir(root_dir)
    for model_name in model_names:
        model_dir = os.path.join(root_dir, model_name)
        model_ids = os.listdir(model_dir)
        for model_id in model_ids:
            model_id_dir = os.path.join(model_dir, model_id)
            path = os.path.join(model_id_dir, fname)
            if os.path.exists(path):
                paths.append(path)
    return paths


def find_model_id(cache, model, channel=None, where=None, on_multi="get-choice", return_channel_con=False, return_channel_name=False):
    if where is None or where == []:
        channel_name, channel_con, _ = cache[0]
        found = channel_con.get(model)
        if 0:
            if len(found) == 0:
                raise ValueError("No model instances found for model=\"%s\"" % (model))
            elif len(found) == 1:
                model_id, var, _ = found[0]
            elif on_multi == "get-choice":
                print("Found multiple model instances for model=\"%s\". Please choose one:" % (model))
                model_id = util.get_choice(found.get_names())
                var = found.get(model_id)
            elif on_multi == "get-first":
                model_id, var, _ = found[0]
            elif on_multi == "get-last":
                model_id, var, _ = found[-1]
            elif on_multi == "get-all":
                model_id = found.get_names()
            else:
                raise ValueError()
    else:
        if not (util.Types.is_list_of_list(where) or util.Types.is_list_of_dict(where)):
            if isinstance(where, dict):
                where = [where]
            elif isinstance(where, list):
                if isinstance(where[0], str):
                    where = [where]
                elif not isinstance(where[0], list):
                    raise ValueError("Input where may be list, dict, list of list, list of dict, or None. Received %s" % (type(where)))
            else:
                raise ValueError("Input where may be list, dict, list of list, list of dict, or None. Received %s" % (type(where)))
        if util.Types.is_list_of_dict(where):
            if channel is None:
                found = cache
                for _where in where:
                    found = found.find(**_where)
                channel_name, channel_con, _ = found[0]
                found = channel_con.get(model)
            else:
                found = cache.get(channel).get(model)
            for _where in where:
                found = found.find(**_where)
        else:
            if isinstance(where[0], str): # single condition - wrap to treat as multi
                where = [where]
            names = [_[0] for _ in where]
            comparators = [_[1] for _ in where]
            values = [_[2] for _ in where]
            if channel is None:
                found = cache.find(names[0], values[0], comparators[0])
                if len(found) == 0:
                    raise ValueError(
                        "Input where=%s did not result in any found cache channels" % (str(where))
                    )
                channel_name, channel_con, _ = found[0]
            else:
                channel_con = cache.get(channel)
            found = channel_con.find(names, values, comparators, path=[model])
    if len(found) == 0:
        raise ValueError("No model instances found for model=\"%s\" matching where=%s" % (model, str(where)))
    elif len(found) == 1:
        model_id, var, _ = found[0]
    elif on_multi == "get-choice":
        print("Found multiple model instances for model=\"%s\" matching where=%s. Please choose one:" % (model, str(where)))
        model_id = util.get_choice(found.get_names())
        var = found.get(model_id)
    elif on_multi == "get-first":
        model_id, var, _ = found[0]
    elif on_multi == "get-last":
        model_id, var, _ = found[-1]
    elif on_multi == "get-all":
        model_id = found.get_names()
    else:
        raise ValueError()
    if return_channel_con and return_channel_name:
        return model_id, channel_con, channel_name
    elif return_channel_con:
        return model_id, channel_con
    elif return_channel_name:
        return model_id, channel_name
    return model_id
