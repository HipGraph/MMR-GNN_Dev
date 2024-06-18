import os
import sys
import re
import signal
from subprocess import Popen
from importlib import import_module

import util
import gather
from container import Container
from arguments import ArgumentParser, ArgumentBuilder, arg_flag


def signal_handler(sig, frame):
    print("Exiting from interrupt Ctrl+C")
    sys.exit()


class Executor:

    def __init__(
        self, 
        interpreter="python", 
        program="main.py", 
        n_processes=1, 
        input_timeout=60, 
        input_default="n", 
        checkpoint=True, 
        pass_as_argfile=False, 
        print_all=False, 
        print_done=False, 
        print_todo=False, 
        debug=False, 
    ):
        self.interpreter = interpreter
        self.program = program
        self.n_processes = n_processes
        self.input_timeout = input_timeout
        self.input_default = input_default
        self.checkpoint = checkpoint
        self.pass_as_argfile = pass_as_argfile
        self.print_all = print_all
        self.print_done = print_done
        self.print_todo = print_todo
        self.debug = debug
        signal.signal(signal.SIGINT, self.on_interrupt)

    def invoke(self, jobs):
        arg_bldr, chkptr = ArgumentBuilder(), Checkpointer()
        i = -1
        while i < len(jobs) - 1:
            i += 1
            job = jobs[i]
            if self.print_all or \
                (self.print_done and chkptr.is_completed(job)) or \
                (self.print_todo and not chkptr.is_completed(job)):
                self.print_job(job, i)
            if self.print_all or self.print_done or self.print_todo:
                continue
            if self.checkpoint and job.chkpt:
                if chkptr.is_completed(job):
                    print("JOB ALREADY COMPLETED. MOVING ON...")
                    continue
            args = Container().copy(job.work)
            processes = []
            for j in range(self.n_processes):
                args.set("process_rank", j, None, ["distribution"])
                invocation = [self.interpreter, self.program]
                if self.pass_as_argfile:
                    path = os.sep.join([os.path.dirname(os.path.realpath(__file__)), "args[%d].pkl" % (j)])
                    util.to_cache(args, path)
                    invocation += ["--f", path]
                else:
                    invocation += arg_bldr.build(args)
                processes += [Popen(invocation)]
            self.interrupted, exited_correctly = False, True
            for process in processes:
                return_code = process.wait()
                exited_correctly = exited_correctly and not return_code
            if self.interrupted or not exited_correctly:
                _input = self.get_failure_input()
                if "r" in _input.lower():
                    i -= 1
                elif "q" in _input.lower():
                    break
                elif "n" in _input.lower():
                    pass
                else:
                    raise ValueError("Unknown option \"%s\". Quitting..." % (_input))
            elif self.checkpoint and job.chkpt:
                chkptr.checkpoint(job)

    def print_job(self, job, i):
        print(util.make_msg_block(47*"#" + " Arguments %2d " % (i) + 47*"#"))
        print(job.work)

    def get_failure_input(self):
        print(util.make_msg_block("Job failed and will not be checkpointed", "!"))
        if self.input_timeout <= 0:
            _input = self.input_default
        elif 0:
            try:
                _input = util.input_with_timeout("Repeat, next, or quit? [r,n,q]: ", self.input_timeout)
            except util.TimeoutExpired:
                print("No input received. Taking default option \"%s\"" % (self.input_default))
                _input = self.input_default
        else:
            _input = input("Repeat, next, or quit? [r,n,q]: ")
        return _input

    def on_interrupt(self, sig, frame):
        self.interrupted = True


class Checkpointer:

    sep = "\n\n"

    def __init__(self):
        pass

    def filter_for_checkpoint(self, work):
        work = Container().copy(work)
        work.rem(
            ["checkpoint_dir", "evaluation_dir", "train_timeout", "view_model_forward"], 
            must_exist=False
        )
        return work

    def get_checkpoint_contents(self, job):
        work = self.filter_for_checkpoint(job.work)
        return ArgumentBuilder().view(work)

    def checkpoint(self, job):
        path = job.checkpoint_path()
        chkpt_dir = job.checkpoint_dir()
        os.makedirs(chkpt_dir, exist_ok=True)
        contents = self.get_checkpoint_contents(job)
        with open(path, "a+") as f:
            f.write(contents + self.sep)

    def job_from_checkpoint(self, checkpoint):
        work = gather.parse_vars(checkpoint)
        work = self.filter_for_checkpoint(work)
        return Job(work)

    def get_completed_jobs(self, job):
        path = job.checkpoint_path()
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            checkpoints = f.read().split(self.sep)
        checkpoints = [chkpt for chkpt in checkpoints if chkpt != ""]
        completed_jobs = [self.job_from_checkpoint(chkpt) for chkpt in checkpoints]
        return completed_jobs

    def is_completed(self, job):
        return Job(self.filter_for_checkpoint(job.work)) in self.get_completed_jobs(job)


class Job:

    def __init__(self, work, chkpt=False):
        if isinstance(work, list):
            work = ArgumentParser().parse_arguments(work)
        elif not isinstance(work, Container):
            raise ValueError("Parameter \"work\" must be a list (sys.argv) or Container. Received %s" % (
                str(type(work))
            ))
        self.name = "Job"
        self.work = work
        self.chkpt = chkpt

    def root_dir(self):
        return os.sep.join(__file__.split(os.sep)[:-1])

    def checkpoint_dir(self):
        return os.sep.join([self.root_dir(), "Cache"])

    def checkpoint_path(self):
        return os.sep.join([self.checkpoint_dir(), "CompletedExperiments.txt"])

    def __eq__(self, job):
        return self.work == job.work

    def __str__(self):
        return self.work.to_string(extent=[-1, -1])


class Driver:

    interpreter = "python"
    program = "main.py"
    n_processes = 1
    input_timeout = sys.float_info.max
    input_default = "n"
    exec_index = None
    checkpoint = True
    pass_as_argfile = False
    print_all = False
    print_done = False
    print_todo = False
    debug = False

    def __init__(self, args):
        driver_args = None
        if "driver" in args:
            driver_args = args.driver
            args.rem("driver")
        if not driver_args is None:
            if "interpreter" in driver_args: self.interpreter = driver_args.interpreter
            if "program" in driver_args: self.program = driver_args.program
            if "n_processes" in driver_args: self.n_processes = driver_args.n_processes
            if "input_timeout" in driver_args: self.input_timeout = driver_args.input_timeout
            if "input_default" in driver_args: self.input_default = driver_args.input_default
            if "exec_index" in driver_args: self.exec_index = driver_args.exec_index
            if "checkpoint" in driver_args: self.checkpoint = driver_args.checkpoint
            if "pass_as_argfile" in driver_args: self.pass_as_argfile = driver_args.pass_as_argfile
            if "print_all" in driver_args: self.print_all = driver_args.print_all
            if "print_done" in driver_args: self.print_done = driver_args.print_done
            if "print_todo" in driver_args: self.print_todo = driver_args.print_todo
            if "debug" in driver_args: self.debug = driver_args.debug

    def run(self, args):
        if "E" in args: # Invoking a set of jobs defined by an experiment
            if not "e" in args:
                raise ValueError("Given %sE but missing %se" % (arg_flag, arg_flag))
            exp_module_name, exp_id = args.get(["E", "e"])
            args.rem(["E", "e"])
            exp_module = import_module("experiment.%s" % (exp_module_name))
            if isinstance(exp_id, list): # Multiple experiments: collect jobs from all
                jobs = []
                for _exp_id in exp_id:
                    exp = getattr(exp_module, "Experiment__%s" % (str(_exp_id).replace(".", "_")))()
                    jobs += self.filter_jobs(exp.jobs, self.exec_index)
            else:
                exp = getattr(exp_module, "Experiment__%s" % (str(exp_id).replace(".", "_")))()
                jobs = self.filter_jobs(exp.jobs, self.exec_index)
        elif "A" in args: # Invoking an anlysis
            if not "a" in args:
                raise ValueError("Given %sA but missing %sa" % (arg_flag, arg_flag))
            ana_module_name, ana_id = args.get(["A", "a"])
            args.rem(["A", "a"])
            ana_module = import_module("analysis.%s" % (ana_module_name))
            if not isinstance(ana_id, list):
                ana_id = [ana_id]
            for _ana_id in ana_id:
                if ana_module_name == "Analysis":
                    ana = getattr(ana_module, str(_ana_id).replace(".", "_"))
                else:
                    ana = getattr(ana_module, "Analysis__%s" % (str(_ana_id).replace(".", "_")))
                ana().run(args)
            sys.exit()
        elif "S" in args: # Invoking SLURM jobs
            if not "s" in args:
                raise ValueError("Given %sS but missing %ss" % (arg_flag, arg_flag))
            slurm_module_name, slurm_id = args.get(["S", "s"])
            args.rem(["S", "s"])
            slurm_module = import_module("slurm.%s" % (slurm_module_name))
            if not isinstance(slurm_id, list):
                slurm_id = [slurm_id]
            for _slurm_id in slurm_id:
                if slurm_module_name == "Slurm":
                    slurm = getattr(slurm_module, str(_slurm_id).replace(".", "_"))
                else:
                    slurm = getattr(slurm_module, "Slurm__%s" % (str(_slurm_id).replace(".", "_")))
                slurm().run(args)
            sys.exit()
        elif "I" in args: # Invoking data integration
            integration_module = import_module("data.integration")
            integrator_class_name = args.get("I")
            args.rem("I")
            acquire, convert = True, True
            if "acquire" in args:
                acquire = args.get("acquire")
                args.rem("acquire")
            if "convert" in args:
                convert = args.get("convert")
                args.rem("convert")
            if not isinstance(integrator_class_name, list):
                integrator_class_name = [integrator_class_name]
            for _integrator_class_name in integrator_class_name:
                integrator_class = getattr(integration_module, _integrator_class_name.replace(".", "_"))
                integrator = integrator_class()
            if acquire:
                integrator.acquire(args)
            if convert:
                integrator.convert(args)
            sys.exit()
        elif "G" in args: # Invoking data generation
            generation_module = import_module("data.generation")
            generator_class_name = args.get("G")
            args.rem("G")
            if not isinstance(generator_class_name, list):
                generator_class_name = [generator_class_name]
            for _generator_class_name in generator_class_name:
                generator_class = getattr(generation_module, _generator_class_name)
                generator = generator_class()
                generator.generate(args)
            sys.exit()
        else: # Invoking a single job with supplied args
            jobs = [Job(args)]
        # Edit jobs
        for job in jobs: # Add driver-level args to all jobs
            job.work.merge(args, coincident_only=False)
        # All jobs are ready - move to invocation
        executor = Executor(
            self.interpreter, 
            self.program, 
            self.n_processes, 
            self.input_timeout, 
            self.input_default, 
            self.checkpoint, 
            self.pass_as_argfile, 
            self.print_all, 
            self.print_done, 
            self.print_todo, 
            self.debug, 
        )
        executor.invoke(jobs)

    def filter_jobs(self, jobs, exec_index):
        if exec_index is None:
            return jobs
        def is_cond(_exec_index):
            return any([
                len(_exec_index) == 3 and _exec_index[1] in util.comparator_fn_map, 
                len(_exec_index) == 4 and _exec_index[0] == "~" and _exec_index[2] in util.comparator_fn_map, 
            ])
        def is_range(_exec_index):
            return isinstance(_exec_index, str) and bool(re.match("^~?range\(.*\)$", _exec_index))
        def get_range_index(_exec_index, jobs):
            N = len(jobs)
            invert = False
            if _exec_index[0] == "~":
                invert = True
                _exec_index = _exec_index[1:]
            _ = _exec_index.replace("range(", "").replace(")", "").split(",")
            c = 1
            if len(_) == 2:
                a, b = _
            elif len(_) == 3:
                a, b, c = _[0], _[1], int(_[2])
            else:
                raise ValueError(_exec_index)
            if a == "N":
                a = N
            else:
                a = int(a)
                if a < 0:
                    a += N
            if b == "N":
                b = N
            else:
                b = int(b)
                if b < 0:
                    b += N
            index = list(range(a, b, c))
            if invert:
                return util.list_subtract(list(range(N)), index)
            return index
        def get_cond_index(_exec_index, jobs):
            N = len(jobs)
            invert = False
            if len(_exec_index) == 4: # _exec_index=["~", name, comparator, value]
                invert = True
                _exec_index = _exec_index[1:]
            name, comparator, value = _exec_index
            index = []
            for i, job in enumerate(jobs):
                if util.comparator_fn_map[comparator](job.work.get(name), value):
                    index.append(i)
            if invert:
                return util.list_subtract(list(range(N)), index)
            return index
        def get_index(_exec_index, jobs):
            if util.Types.is_collection_of_int(_exec_index):
                return _exec_index
            elif is_range(_exec_index):
                index = get_range_index(_exec_index, len(jobs))
            elif is_cond(_exec_index):
                index = get_cond_index(_exec_index, jobs)
            else:
                raise TypeError(type(exec_index))
            return index
        # Setup
        if not util.Types.is_collection_of_collection(exec_index):
            exec_index = [exec_index]
        #
        invert = False
        if exec_index[0] == "~":
            invert = True
            exec_index = exec_index[1:]
        index = []
        for _exec_index in exec_index:
            index.append(get_index(_exec_index, jobs))
        index_set = set(index[0])
        for _index in index[1:]:
            index_set &= set(_index)
        index = sorted(list(index_set))
        if invert:
            index = util.list_subtract(list(range(len(jobs))), index)
        # Apply filter
        return [jobs[i] for i in index]


if __name__ == "__main__":
    args = ArgumentParser().parse_arguments(sys.argv[1:])
    driver = Driver(args)
    driver.run(args)
