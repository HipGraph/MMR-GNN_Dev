import sys
import os
import numpy as np

import Utility as util
from Driver import Checkpointer
from Container import Container
from Arguments import ArgumentParser, ArgumentBuilder
from Variables import Variables


tab = 4*" "


def prepare_path(path):
    return path.replace("(", "\(").replace(")", "\)")


def double_check_rm_one(path):
    check = input(
        "Remove directory \"%s\"? : " % (path)
    )
    if check.lower() in ["y", "yes"]:
        check = input("Are you sure? : ")
    else:
        return
    if check.lower() in ["y", "yes"]:
        path = prepare_path(path)
        os.system("rm -r %s" % (path))


def double_check_rm_all(paths):
    check = input(
        "Remove directories (%s)? : " % ("\n    %s\n" % ("\n    ".join(paths)))
    )
    if check.lower() in ["y", "yes"]:
        check = input("Are you sure? : ")
    else:
        return
    if check.lower() in ["y", "yes"]:
        for path in paths:
            path = prepare_path(path)
            os.system("rm -r %s" % (path))


def double_check_rm(path):
    if isinstance(path, (tuple, list)):
        if len(path) < 1:
            print("No paths selected. Quitting.")
            return
        double_check_rm_all(path)
    else:
        double_check_rm_one(path)


def clean_dataset_cache(args):
    def get_cache_dir(dataset, var):
        dataset_var = var.datasets.get(dataset)
        if not dataset_var.spatial.is_empty():
            cache_dir = dataset_var.spatial.structure.cache_dir
        elif not dataset_var.temporal.is_empty():
            cache_dir = dataset_var.temporal.structure.cache_dir
        elif not dataset_var.spatiotemporal.is_empty():
            cache_dir = dataset_var.spatiotemporal.structure.cache_dir
        elif not dataset_var.graph.is_empty():
            cache_dir = dataset_var.graph.structure.cache_dir
        return cache_dir
    #
    var = Variables()
    dataset = util.get_choice(var.datasets.get_names())
    if isinstance(dataset, str):
        cache_dir = get_cache_dir(dataset, var)
    elif isinstance(dataset, (tuple, list)):
        cache_dir = [get_cache_dir(_, var) for _ in dataset]
    else:
        raise TypeError(datasets)
    double_check_rm(cache_dir)


def clean_experiment_cache(args):
    paths = util.get_paths("Experimentation", ".*", files=False)
    print("Here are the experiment modules I found:")
    module_dir = util.get_choice(paths)
    paths = util.get_paths(module_dir, ".*", files=False)
    print("Here are the experiments I found:")
    cache_dir = util.get_choice(paths)
    double_check_rm(cache_dir)

def clean_exp_cache(args):
    clean_experiment_cache(args)


def clean_analysis_cache(args):
    paths = util.get_paths("Analysis", ".*", files=False)
    print("Here are the analysis modules I found:")
    module_dir = util.get_choice(paths)
    paths = util.get_paths(module_dir, ".*", files=False)
    print("Here are the analysis' I found:")
    cache_dir = util.get_choice(paths)
    double_check_rm(cache_dir)

def clean_ana_cache(args):
    clean_analysis_cache(args)


def clean_model_cache(args):
    paths = util.get_paths("Experimentation", ".*", files=False)
    print("Here are the experiment modules I found:")
    module_dir = util.get_choice(paths)
    paths = util.get_paths(module_dir, ".*", files=False)
    print("Here are the experiments I found:")
    exp_dir = util.get_choice(paths)
    paths = util.get_paths(exp_dir, "\d+\.\d+.*", recurse=True, files=False)
    paths = np.unique([os.sep.join(_.split(os.sep)[:-1]) for _ in paths])
    print("Here are the models I found:")
    cache_dir = util.get_choice(paths)
    double_check_rm(cache_dir)

def clean_mod_cache(args):
    clean_model_cache(args)


def clean_modelid_cache(args):
    paths = util.get_paths("Experimentation", ".*", files=False)
    print("Here are the experiment modules I found:")
    module_dir = util.get_choice(paths)
    paths = util.get_paths(module_dir, ".*", files=False)
    print("Here are the experiments I found:")
    exp_dir = util.get_choice(paths)
    paths = util.get_paths(exp_dir, "\d+\.\d+.*", recurse=True, files=False)
    print("Here are the models IDs I found:")
    cache_dir = util.get_choice(paths)
    double_check_rm(cache_dir)

def clean_mid_cache(args):
    clean_modelid_cache(args)


def clean_slurm_cache(args):
    paths = util.get_paths("Slurm", ".*", files=False)
    print("Here are the slurm modules I found:")
    module_dir = util.get_choice(paths)
    paths = util.get_paths(module_dir, ".*", files=False)
    print("Here are the slurm caches I found:")
    cache_dir = util.get_choice(paths)
    double_check_rm(cache_dir)


def clean_chkpt_cache(args):
    chkptr = Checkpointer()
    paths = util.get_paths("Experimentation", ".*", files=False)
    print("Here are the experiment modules I found:")
    module_dir = util.get_choice(paths)
    paths = util.get_paths(module_dir, ".*", files=False)
    print("Here are the experiments I found:")
    exp_dir = util.get_choice(paths)
    path = os.sep.join([exp_dir, "Cache", "CompletedExperiments.txt"])
    with open(path, "r") as f:
        chkpts = f.read().split(chkptr.sep)
    jobs = [chkptr.job_from_checkpoint(_) for _ in chkpts if _ != ""]
    if len(jobs) == 0:
        print("No checkpoints found @ %s... Aborting." % (path))
        return
    if "where" in args:
        if not util.Types.is_collection_of_collection(args.where): # singleton where=[name,cond,value]
            args.where = [args.where]
        index = []
        for i, job in enumerate(jobs):
            if all(util.comparator_fn_map[cond](job.work.get(name), value) for name, cond, value in args.where):
                index.append(i)
        _jobs = [jobs[i] for i in index]
        if len(_jobs) == 0:
            print("No jobs matching criteria where=%s... Aborting." % (str(args.where).replace(" ", "")))
            return
        #
        print("Remove %d checkpoints (" % (len(_jobs)))
        for i, _job in zip(index, _jobs):
            print(tab+util.make_msg_block(50*"#"+" %d "%(i)+50*"#").replace("\n", "\n%s" % (tab)))
            print(tab+str(_job).replace("\n", "\n%s" % (tab)))
        check = input(")? : ")
        if check.lower() in ["y", "yes"]:
            check = input("Are you sure? : ")
        else:
            return
        if not check.lower() in ["y", "yes"]:
            return
        index = np.delete(np.arange(len(jobs)), index)
        jobs = [jobs[i] for i in index]
        chkpts = [chkptr.get_checkpoint_contents(_) for _ in jobs]
        with open(path, "w") as f:
            f.write(chkptr.sep.join(chkpts))
    else:
        raise NotImplementedError()
        opt = util.get_choice(["regex", "where"])
        if opt == "regex":
            raise NotImplementedError()
        elif opt == "where":
            print(args)
            input()
            where = input("where [name,cond,value]: ").split(",")
            where = ["\"%s\"" % (_) for _ in where]
            where = ArgumentParser().parse_arguments(
                ["--name", where[0], "--cond", where[1], "--value", where[2]]
            )
            print(where)
        print(jobs[0])
    


if __name__ == "__main__":
    args = ArgumentParser().parse_arguments(sys.argv[2:])
    globals()["clean_%s"%(sys.argv[1])](args)
