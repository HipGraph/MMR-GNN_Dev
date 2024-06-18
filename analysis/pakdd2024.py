import os
import numpy as np
import pandas as pd

import util
import gather
import analysis
import experiment
from container import Container
from plotting import Plotting
from experiment import pakdd2024


def module_dir():
    return os.path.dirname(__file__)


def module_name():
    return os.path.basename(__file__).replace(".py", "")


def get_exp(exp_id=None):
    return experiment.get_exp(pakdd2024, exp_id)


class PAKDD2024_Analysis(analysis.Analysis):

    def name(self):
        return self.__class__.__name__.replace("Analysis__", "")

    def result_dir(self):
        return os.sep.join([module_dir(), module_name().upper(), self.name()])


class Analysis__ModelComparison(PAKDD2024_Analysis, analysis.ModelComparison):

    def __init__(self):
        self.exp = get_user_exp()

    def name(self):
        return "ModelComparison[%s]" % (self.exp.name())


class Analysis__ParameterSearch(PAKDD2024_Analysis, analysis.ParameterSearch):

    def __init__(self):
        self.exp = get_user_exp()

    def name(self):
        return "ParameterSearch[%s]" % (self.exp.name())


class Analysis__FeatureCorrelation(PAKDD2024_Analysis, analysis.FeatureCorrelation):
    pass


class Analysis__SpatiotemporalVisualization(PAKDD2024_Analysis, analysis.SpatiotemporalVisualization):
    pass


class Analysis__Clustering(PAKDD2024_Analysis, analysis.Clustering):
    pass


class Analysis__Hierarchy(PAKDD2024_Analysis, analysis.Hierarchy):
    pass



class MainAnalysis(PAKDD2024_Analysis, analysis.ModelComparison):
    
    def exp_id(self):
        return self.__class__.__name__.split("__")[-1]

    def exp(self):
        if not hasattr(self, "_exp"):
            self._exp = get_exp(self.exp_id())
        return self._exp

    def name(self):
        return self.exp().name()

    def values(self):
        return ["MAE", "MAPE", "RMSE"]

    def value_partitions(self):
        return "test"

    def on_multi(self):
        return "get-all"


class Analysis__1_1(MainAnalysis): pass


class Analysis__1_2(MainAnalysis): pass


class Analysis__2_1(MainAnalysis): pass


class Analysis__2_11(MainAnalysis): pass


class Analysis__2_2(MainAnalysis): pass


class Analysis__2_21(MainAnalysis): pass


class Analysis__3_1(MainAnalysis): pass


class Analysis__3_2(MainAnalysis): pass



class AblationAnalysis(PAKDD2024_Analysis, analysis.ParameterSearch):
    
    def exp_id(self):
        return self.__class__.__name__.split("__")[-1]

    def exp(self):
        if not hasattr(self, "_exp"):
            self._exp = get_exp(self.exp_id())
        return self._exp

    def name(self):
        return self.exp.name()

    def model(self):
        return "THSTM"

    def values(self):
        return ["MAE", "MAPE", "RMSE"]

    def value_partitions(self):
        return "test"

    def on_multi(self):
        return "get-all"


class Analysis__0_1(AblationAnalysis):

    def default_args(self):
        return Container().set(
            [
                "metrics", 
                "debug", 
            ], 
            [
                self.values(), 
                0, 
            ], 
            multi_value=True
        )

    def steps(self, args):
        args = self.get_args(args)
        #
        chkpt_dir, eval_dir = self.exp.jobs[0].work.get(["checkpoint_dir", "evaluation_dir"])
        cache = Gather.get_cache(eval_dir, chkpt_dir)
        model_ids = [
            ["THSTM", cache.settings.THSTM[1][0]], 
            ["THSTM", cache.settings.THSTM[0][0]], 
        ]
        exp_errors = [cache.errors.get(_id, path=[model]) for model, _id in model_ids]
        metrics = cache.errors.THSTM[0][1].get_names()
        data = {metric: [] for metric in metrics}
        for i, errors in enumerate(exp_errors):
            if args.debug:
                print(errors)
                print(model_ids[i])
                input()
            for metric in metrics:
                data[metric].append(
                    np.mean(list(errors.get(metric, "test")[0][1].values()))
                )
        df = pd.DataFrame(data) 
        df["Decoder"] = [
            "xmark", 
            "cmark", 
        ]
        df = df[["Decoder"]+args.metrics]
        df = df.round(3)
        print(df)
        table = df.to_latex(index=False, na_rep="N/A")
        table = table.replace("xmark", "\\xmark").replace("cmark", "\\cmark")
        path = os.sep.join([self.result_dir(), "Table_0.1.tex"])
        with open(path, "w") as f:
            f.write(table)


class Analysis__0_2(AblationAnalysis):

    def default_args(self):
        return Container().set(
            [
                "metrics", 
                "debug", 
            ], 
            [
                self.values(), 
                0, 
            ], 
            multi_value=True
        )

    def steps(self, args):
        args = self.get_args(args)
        #
        chkpt_dir, eval_dir = self.exp.jobs[0].work.get(["checkpoint_dir", "evaluation_dir"])
        cache = Gather.get_cache(eval_dir, chkpt_dir)
        if args.debug:
            print(cache)
        model_ids = [
            ["THSTM", cache.settings.THSTM.find("rnn_layer", "gcGRU")[0][0]], 
            ["THSTM", cache.settings.THSTM.find("rnn_layer", "gcstGRU")[0][0]], 
        ]
        exp_errors = [cache.errors.get(_id, path=[model]) for model, _id in model_ids]
        metrics = cache.errors.THSTM[0][1].get_names()
        data = {metric: [] for metric in metrics}
        for i, errors in enumerate(exp_errors):
            if args.debug:
                print(errors)
                print(model_ids[i])
                input()
            for metric in metrics:
                data[metric].append(
                    np.mean(list(errors.get(metric, "test")[0][1].values()))
                )
        df = pd.DataFrame(data) 
        df["Multi-domain"] = [
            "xmark", 
            "cmark", 
        ]
        df = df[["Multi-domain"]+args.metrics]
        df = df.round(3)
        print(df)
        table = df.to_latex(index=False, na_rep="N/A")
        table = table.replace("xmark", "\\xmark").replace("cmark", "\\cmark")
        path = os.sep.join([self.result_dir(), "Table_0.2.tex"])
        with open(path, "w") as f:
            f.write(table)


class Analysis__0_3(AblationAnalysis):

    def default_args(self):
        return Container().set(
            [
                "metrics", 
                "partition", 
                "debug", 
            ], 
            [
                self.values(), 
                "test", 
                0, 
            ], 
            multi_value=True
        )

    def steps(self, args):
        args = self.get_args(args)
        #
        chkpt_dir, eval_dir = self.exp.jobs[0].work.get(["checkpoint_dir", "evaluation_dir"])
        cache = Gather.get_cache(eval_dir, chkpt_dir)
        if args.debug:
            print(cache)
        model_ids = [
            ["THSTM", cache.settings.THSTM.find("n_clusters", 1)[0][0]], 
            ["AGCRN", cache.errors.AGCRN[0][0]], 
            ["THSTM", cache.settings.THSTM.find(["n_clusters", "rnn_layer"], [10, "gcGRU"])[0][0]], 
            ["THSTM", cache.settings.THSTM.find(["n_clusters", "rnn_layer"], [10, "gcstGRU"])[0][0]], 
            ["THSTM", cache.settings.THSTM.find("n_clusters", 100)[0][0]], 
        ]
        exp_errors = [cache.errors.get(_id, path=[model]) for model, _id in model_ids]
        metrics = cache.errors.THSTM[0][1].get_names()
        data = {metric: [] for metric in metrics}
        for i, errors in enumerate(exp_errors):
            if args.debug:
                print(errors)
                input()
            for metric in metrics:
                data[metric].append(
                    np.mean(list(errors.get(metric, args.partition)[0][1].values()))
                )
        df = pd.DataFrame(data) 
        df["Method"] = [
            "Shared (k=1)", 
            "AGCRN (k=10)", 
            "Approximated (k=10)", 
            "Clustered (k=10)", 
            "Full Rank (k=|V|)", 
        ]
        df = df[["Method"]+args.metrics]
        df = df.round(3)
        print(df)
        table = df.to_latex(index=False, na_rep="N/A")
        path = os.sep.join([self.result_dir(), "Table_0.31.tex"])
        with open(path, "w") as f:
            f.write(table)


class Analysis__0_31(AblationAnalysis):

    def parameter(self):
        return "n_clusters"


class Analysis__0_4(AblationAnalysis):

    def default_args(self):
        return Container().set(
            [
                "metrics", 
                "debug", 
            ], 
            [
                self.values(), 
                0, 
            ], 
            multi_value=True
        )

    def steps(self, args):
        args = self.get_args(args)
        #
        chkpt_dir, eval_dir = self.exp.jobs[0].work.get(["checkpoint_dir", "evaluation_dir"])
        exp_vals = [[job.work.use_existing_edges, job.work.graph_construction_method] for job in self.exp.jobs]
        cache = Gather.get_cache(eval_dir, chkpt_dir)
        model_ids = []
        for _exp_vals in exp_vals:
            if args.debug:
                print(_exp_vals)
            model_ids.append(
                [
                    "THSTM", 
                    cache.settings.THSTM.find(
                        ["use_existing_edges", "graph_construction_method"], 
                        _exp_vals
                    )[0][0]
                ]
            )
        exp_errors = [cache.errors.get(_id, path=[model]) for model, _id in model_ids]
        metrics = cache.errors.THSTM[0][1].get_names()
        data = {metric: [] for metric in metrics}
        for i, errors in enumerate(exp_errors):
            if args.debug:
                print(errors)
                input()
            for metric in metrics:
                data[metric].append(
                    np.mean(list(errors.get(metric, "test")[0][1].values()))
                )
        df = pd.DataFrame(data) 
        df["G"] = ["cmark" if _exp_vals[0] else "xmark" for _exp_vals in exp_vals]
        df["Density"] = [
            "%d%%" % (0 if _exp_vals[1] is None else 100*_exp_vals[1][-1]) for _exp_vals in exp_vals
        ]
        df = df[["G", "Density"]+args.metrics]
        df = df.round(3)
        print(df)
        table = df.to_latex(index=False, na_rep="N/A")
        table = table.replace("xmark", "\\xmark").replace("cmark", "\\cmark")
        table = table.replace("G", "$G$")
        path = os.sep.join([self.result_dir(), "Table_0.4.tex"])
        with open(path, "w") as f:
            f.write(table)


class Analysis__0_5(AblationAnalysis):

    def parameter(self):
        return "hidden_size"


class Analysis__0_6(AblationAnalysis):

    def parameter(self):
        return "temporal_mapping"


class Analysis__0_61(Analysis__0_6): pass


class Analysis__0_62(Analysis__0_6): pass


class Analysis__0_63(Analysis__0_6): pass


class Analysis__0_64(Analysis__0_6): pass


class Analysis__0_65(Analysis__0_6): pass


class Analysis__0_66(Analysis__0_6): pass


class Analysis__0_67(Analysis__0_6): pass


class Analysis__0_68(Analysis__0_6): pass


class Analysis__0_7(AblationAnalysis):

    def parameter(self):
        return "rnn_layer"


class Analysis__0_8(AblationAnalysis):

    def parameter(self):
        raise "rnn_kwargs"


class Analysis__0_9(AblationAnalysis):

    def parameter(self):
        return "cluster_method"


class Analysis__0_10(AblationAnalysis):

    def parameter(self):
        return "temporal_mapping"


class Analysis__0_101(Analysis__0_10): pass


class Analysis__0_102(Analysis__0_10): pass


class Analysis__0_103(Analysis__0_10): pass


class Analysis__0_104(Analysis__0_10): pass


class Analysis__0_105(Analysis__0_10): pass


class Analysis__0_106(Analysis__0_10): pass


class Analysis__0_107(Analysis__0_10): pass


class Analysis__0_108(Analysis__0_10): pass


class Analysis__0_211(AblationAnalysis):

    def parameter(self):
        return "predictor_features"


class Analysis__0_221(AblationAnalysis):

    def parameter(self):
        return "predictor_features"



class Analysis__Fig1(PAKDD2024_Analysis):

    def default_args(self):
        return Container().set(
            [
                "exp", 
                "exp_id", 
                "transform", 
                "rep", 
                "size", 
            ], 
            [
                1.1, 
                0, 
                "zscore", 
                "histogram", 
                (8, 4), 
            ], 
            multi_value=True
        )

    def steps(self, args):
        from variables import Variables
        from data.dataset import Data
        from data.spatiotemporal import SpatiotemporalData
        from plotting import Plotting
        # Setup
        args = self.get_args(args)
        self.exp = get_exp(args.exp)
        exp_args = self.exp.jobs[args.exp_id].work
        # Load dataset
        var = Variables().merge(exp_args)
        var.processing.default_feature_transform = args.transform
        var.processing.spatiotemporal.default_feature_transform = args.transform
        datasets = Data(var).train__dataset
        n_spatial = dataset.spatiotemporal.original.train__n_spatial
        n_temporal = dataset.spatiotemporal.original.train__n_temporal
        spatial_labels = dataset.spatiotemporal.original.train__spatial_labels
        temporal_labels = dataset.spatiotemporal.original.train__temporal_labels
        #
        k = 2
        x, clustering, cluster_index = SpatiotemporalData.cluster(
            dataset.spatiotemporal, "Agglomerative", k, args.rep
        )
        y = dataset.spatiotemporal.transformed.original.train__response_features
        indices = np.arange(n_temporal-365, n_temporal)
        if dataset.name == "wabashriver-swat":
            indices = dataset.spatiotemporal.indices_from_selection(
                dataset.spatiotemporal.original.train__temporal_labels, 
                ["interval", "1986-01-01", "1986-12-31"]
            )
        elif dataset.name == "solar-energy":
            indices = np.arange(n_temporal-144, n_temporal)
        elif dataset.name in ["metr-la", "new-metra-la", "pems-bay", "new-pems-bay"]:
            indices = np.arange(n_temporal-288, n_temporal)
        y = y[indices,:,0]
        temporal_labels = temporal_labels[indices]
        # Plot
        plt = Plotting()
        fig, ax = plt.subplots(size=args.size)
        colors = plt.default_colors[:k]
        if 1:
            print("Cluster 1 Nodes:", spatial_labels[cluster_index==0][:10])
            print("Cluster 2 Nodes:", spatial_labels[cluster_index==1][:10])
        font_size = 16
        line_width = 2
        for i in range(k): 
            lines = ["-", "--"]
            for j in range(2):
                index = cluster_index == i
                _spatial_labels = spatial_labels[index]
                _y = y[:,index][:,j]
                plt.plot_line(
                    None, 
                    _y, 
                    ax=ax, 
                    color=colors[i], 
                    label="Cluster %d - %s %s" % (
                        i + 1, 
                        dataset.spatiotemporal.misc.spatial_label_field.capitalize(), 
                        _spatial_labels[j]
                    ), 
                    linewidth=line_width, 
                    linestyle=lines[j]
                )
        plt.ylabel(
            plt.feature_ylabel_map[dataset.spatiotemporal.misc.response_features[0]], fontsize=font_size
        )
        indices = np.linspace(0, len(temporal_labels)-1, 6, dtype=int)
        xticks = temporal_labels[indices]
        if dataset.name in ["solar-energy", "metr-la", "new-metra-la", "pems-bay", "new-pems-bay"]:
            xticks = [xtick.split("_")[-1].replace("-", ":") for xtick in xticks]
        plt.xticks(indices, xticks, fontsize=3/4*font_size)
        plt.yticks([])
        plt.legend(style="standard", fontsize=font_size)
        plt.style("grid")
        # Save
        path = os.sep.join(
            [
                self.result_dir(), 
                "Fig1_dataset[%s]_transform[%s].png" % (dataset.name, args.transform)
            ]
        )
        plt.save_figure(path, fig)
