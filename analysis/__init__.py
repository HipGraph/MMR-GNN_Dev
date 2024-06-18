import re
import os
import sys
import warnings
import numpy as np
import pandas as pd

import util
import gather
from data import transform
from container import Container
from plotting import Plotting


class Analysis:

    def __init__(self):
        pass

    def name(self):
        return self.__class__.__name__

    def default_args(self):
        return Container()

    def get_args(self, args):
        return self.default_args().merge(args)

    def run(self, args):
        if self.result_dir():
            os.makedirs(self.result_dir(), exist_ok=True)
        self.steps(args)
        self.write(args)

    def steps(self, args):
        pass

    def write(self, args):
        pass

    def result_dir(self):
        return os.sep.join([__file__.replace(".py", ""), self.name()])


class ValuesOverParameterAnalysis(Analysis):

    def name(self):
        raise NotImplementedError()

    def exp_id(self):
        return None

    def exp(self):
        raise NotImplementedError("exp()")

    def values(self):
        return values_and_partitions(3)[0] + ["best_epoch"]

    def value_partitions(self):
        return values_and_partitions(3)[1] + [None]

    def value_containers(self):
        return None

    def parameter(self):
        return None

    def parameter_values(self):
        return None

    def parameter_partition(self):
        return None

    def parameter_container(self):
        return None

    def response_feature(self):
        return None

    def model(self):
        return None

    def where(self):
        return []

    def on_multi(self):
        return "get-choice"

    def round_to(self):
        return 3

    def fname_addr(self):
        return ""

    def debug(self):
        return 0

    def default_args(self):
        return Container().set(
            [
                "exp_id", 
                "values",
                "value_partitions",
                "value_containers",
                "parameter",
                "parameter_values",
                "parameter_partition", 
                "parameter_container",
                "response_feature",
                "model",
                "where", 
                "on_multi", 
                "round_to", 
                "plot", 
                "plot_types",
                "plot_kwargs",
                "normalize",
                "fname_adder", 
                "debug",
            ],
            [
                self.exp_id(), 
                self.values(), 
                self.value_partitions(), 
                self.value_containers(), 
                self.parameter(),
                self.parameter_values(), 
                self.parameter_partition(), 
                self.parameter_container(), 
                self.response_feature(), 
                self.model(), 
                self.where(), 
                self.on_multi(), 
                self.round_to(), 
                False, 
                "bar", 
                {"alpha": 0.5}, 
                False, 
                self.fname_addr(), 
                self.debug(), 
            ],
            multi_value=True
        )

    def steps(self, args):
        values_over_parameter_analysis(self.exp(), self.get_args(args), self.result_dir())


class ModelComparison(ValuesOverParameterAnalysis):

    def name(self):
        return "ModelComparison"

    def get_args(self, args):
        _args = self.default_args()
        _args.parameter = "model"
        _args.model = ""
        return _args.merge(args)


class ParameterSearch(ValuesOverParameterAnalysis):

    def name(self):
        return "ParameterSearch"


class FeatureCorrelation(Analysis):

    def steps(self, args):
        from Container import Container
        from Data.Data import Data
        from Data import Clustering, Decomposition
        from Variables import Variables
        from Plotting import Plotting
        import matplotlib.pyplot as plt
        import pandas as pd
        _args = Container().set(
            [
                "dataset", 
                "data_type", 
                "features", 
                "predictor_features", 
                "response_features", 
                "spatial_selection", 
                "transform", 
                "round_to", 
            ], 
            [
                "littleriver", 
                "spatiotemporal", 
                ["~", "date"], 
                None, 
                None, 
                ["all"], 
                "zscore", 
                3, 
            ], 
            multi_value=True, 
        )
        args = _args.merge(args)
        #
        var = Variables()
        var.execution.set("dataset", args.dataset, "*")
        if not args.features is None:
            if args.features[0] == "~":
                args.features = util.list_subtract(
                    var.datasets.get(args.dataset).get(args.data_type).loading.feature_fields, 
                    list(args.features[1:])
                )
        if args.predictor_features is None:
            args.predictor_features = args.features
        if args.response_features is None:
            args.response_features = args.features
        var.mapping.get(args.data_type).predictor_features = args.predictor_features
        var.mapping.get(args.data_type).response_features = args.response_features
        var.processing.get(args.data_type).default_feature_transform = args.transform
        var.meta.load_graph = False
        var.set(
            "spatial_selection", args.spatial_selection, ["train", "valid", "test"], 
            path=["datasets", args.dataset, "spatiotemporal", "partitioning"]
        )
        print(var.datasets.get(args.dataset))
        input()
        datasets = Data(var)
        dataset = datasets.train__dataset
        data = dataset.get(args.data_type)
        predictor_features = data.misc.predictor_features
        response_features = data.misc.response_features
        #
        result_dir = os.sep.join([self.result_dir(), args.dataset])
        os.makedirs(result_dir, exist_ok=1)
        #
        x = data.transformed.original.train__predictor_features
        for response_feature in response_features:
            if args.data_type == "spatial":
                print(x.shape)
                print(x)
                print(y.shape)
                print(y)
                corr = np.corrcoef(x.T)
                idx = data.misc.predictor_features.index(response_feature)
                print(corr.shape)
                print(corr[idx,:])
                corr = corr[idx,:]
                sort_index = np.flip(np.argsort(corr))
                features = np.array(dataset.spatial.misc.features)[sort_index]
                corr = corr[sort_index]
                df = {"Variable": features, "Correlation": corr}
            elif args.data_type == "spatiotemporal":
                x = data.transformed.original.train__predictor_features
                n_temporal, n_spatial, n_feature = x.shape
                spatial_labels = data.original.train__spatial_labels
                cols = [data.misc.spatial_label_field] + data.misc.predictor_features
                df = {col: [0.0] * n_spatial for col in cols}
                predictor_features = data.misc.predictor_features
                for i in range(n_spatial):
                    corr = np.corrcoef(x[:,i,:].T)
                    for j, feature in enumerate(predictor_features):
                        idx = predictor_features.index(response_feature)
                        df[feature][i] = corr[j,idx]
                df[data.misc.spatial_label_field] = spatial_labels
            else:
                raise NotImplementedError(args.data_type)
            df = pd.DataFrame(df)
            _ = df.iloc[:,1:].to_numpy()
            df.loc[len(df.index)] = ["$\\mu(x)$"] + list(np.mean(_, 0))
            df.loc[len(df.index)] = ["$cc(x)$"] + list(1 - (np.mean(np.abs(_), 0) - np.abs(np.mean(_, 0))))
            df = df.round(args.round_to)
            summary = df.describe().round(args.round_to)
            print(df)
            print(summary)
            #
            warnings.simplefilter(action='ignore', category=FutureWarning)
            path = os.sep.join(
                [
                    result_dir, 
                    "FC_Feature[%s].csv" % (response_feature)
                ]
            )
            sum_path = os.sep.join(
                [
                    result_dir, 
                    "FCS_Feature[%s].csv" % (response_feature)
                ]
            )
            df.to_csv(path, index=False)
            summary.to_csv(sum_path)
            path = path.replace(".csv", ".tex")
            sum_path = sum_path.replace(".csv", ".tex")
            table = self.clean_table(df.to_latex(index=False, na_rep="N/A"))
            with open(path, "w") as f:
                f.write(table)
            table = self.clean_table(summary.to_latex(index=False, na_rep="N/A"))
            with open(sum_path, "w") as f:
                f.write(table)

    def clean_table(self, table):
        return clean_tex(table)


class MissingValues(Analysis):

    def default_args(self):
        return Container().set(
            [
                "dataset", 
                "reduction", 
                "figsize", 
                "debug", 
            ],
            [
                "littleriver", 
                6*60, 
                (8, 8), 
                0, 
            ],
            multi_value=True
        )


    def steps(self, args):
        import matplotlib
        from Plotting import Plotting
        from Variables import Variables
        from Data.Data import Data
        from Data import Probability
        # Setup variables + args
        args = self.get_args(args)
        result_dir = os.sep.join([self.result_dir(), args.dataset])
        os.makedirs(result_dir, exist_ok=True)
        #   load dataset used in the target experiment
        var = Variables()
        var.execution.set("dataset", args.dataset, "*")
        datasets = Data(var)
        train_dataset = datasets.train__dataset
        spatmp = train_dataset.spatiotemporal
        spalabs = spatmp.original.spatial_labels
        temporal_labels = spatmp.original.temporal_labels
        features = spatmp.misc.features
        n_feature = len(features)
        #
        plt = Plotting()
        cmap = matplotlib.colors.ListedColormap(["green", "red"])
        T = temporal_labels
        M = spatmp.original.gtmask.astype(int)
        index = util.sliding_window_indices(M.shape[0], args.reduction, stride=args.reduction)
        T = T[index][:,0]
        xtick_index = np.linspace(0, len(T)-1, min(10, len(T)), dtype=int)
        T = T[xtick_index]
        M = np.sum(M[index,:,:], 1)
        percents = (1 - (np.sum(M, 0) / (M.shape[0] * args.reduction))) * 100
        _n = 50
        print(_n*"*"+" PERCENT MISSING "+_n*"*")
        print(spalabs)
        for i, feature in enumerate(features):
            print("%s: [%.1f%%] =" % (features[i], np.mean(percents[:,i])), percents[:,i])
        #
        M = np.swapaxes(M, 0, 1)
        path = os.sep.join([result_dir, "MVHM.png"])
        fig, axes = plt.subplots(n_feature, size=args.figsize)
        for i, feature in enumerate(features):
            ax = axes[i]
#                plt.plot_heatmap(M[...,i], xtick_locs=xtick_index, xtick_labels=xtick_index, cmap="inferno", ax=ax)
            plt.plot_heatmap(M[...,i], cmap="inferno", vmin=0, vmax=args.reduction, ax=ax)
            if n_feature < 11:
                plt.title(feature, ax=ax, fontsize=9)
        plt.save_figure(path, fig=fig)
        plt.close(fig)
        return
        #
        for i, feature in enumerate(features):
            path = os.sep.join(
                [result_dir, "MVHM_Feature[%s].png" % (feature)]
            )
            fig, ax = plt.subplots(size=(6, 6))
            plt.plot_heatmap(M[...,i], xlabel="Temporal", ylabel="Spatial", cmap="inferno", ax=ax)
            plt.title(feature, ax=ax, fontsize=9)
            plt.save_figure(path, fig=fig)
            plt.close(fig)


class SpatiotemporalVisualization(Analysis):

    def steps(self, args):
        from matplotlib import pyplot
        from Plotting import Plotting
        from Variables import Variables
        from Data.Data import Data
        from Data import Probability, Clustering
        # Setup variables + args
        _args = Container().set(
            [
                "dataset", 
                "data_form",
                "transform", 
                "partition", 
                "temporal_selection", 
                "spatial_selection", 
                "feature_selection", 
                "n_bins", 
                "bin_interval", 
                "xticks", 
                "yticks", 
                "xlabel", 
                "ylabel", 
                "plot_type", 
                "plot_args", 
                "debug", 
            ],
            [
                "littleriver", 
                "original", 
                "zscore", 
                "test", 
#                ["interval", "2022-04-20_18-30-00", "2022-04-21_16-08-00"], 
                None, 
                ["random", 3, 0], 
                ["~", "date"], 
                24+1, 
                None, 
                1, 
                1, 
                1, 
                1, 
                "time-series", 
                Container().set(
                    ["mean", "std", "cbar", "xticks", "lw", ], [1, 1, 1, 1, 1.0], multi_value=1
                ), 
                0, 
            ],
            multi_value=True
        )
        args = _args.merge(args)
        result_dir = os.sep.join([self.result_dir(), args.dataset])
        os.makedirs(result_dir, exist_ok=True)
        #   load dataset used in the target experiment
        var = Variables(1, 0)
        var.execution.set("dataset", args.dataset, "*")
        var.processing.spatiotemporal.default_feature_transform = args.transform
        if not args.spatial_selection is None:
            var.datasets.get(args.dataset).spatiotemporal.partitioning.set(
                "spatial_selection", args.spatial_selection, args.partition
            )
        if not args.temporal_selection is None:
            var.datasets.get(args.dataset).spatiotemporal.partitioning.set(
                "temporal_selection", args.temporal_selection, args.partition
            )
        if not args.feature_selection is None:
            if args.feature_selection[0] == "~":
                args.feature_selection = ["literal"] + util.list_subtract(
                    var.datasets.get(args.dataset).spatiotemporal.loading.feature_fields, 
                    list(args.feature_selection[1:])
                )
        datasets = Data(var)
        # Setup for call to plotting routine
        dataset = datasets.get("dataset", "train")
        spatmp = dataset.spatiotemporal
        features = spatmp.transformed.original.get("features", args.partition)
        gtmask = spatmp.original.get("gtmask", args.partition)
        means, stds = spatmp.statistics.means, spatmp.statistics.standard_deviations
        temporal_labels = spatmp.get(args.data_form).get("temporal_labels", args.partition)
        spatial_labels = spatmp.get(args.data_form).get("spatial_labels", args.partition)
        feature_labels = np.array(spatmp.misc.features)
        n_temporal, n_spatial, n_feature = features.shape
        spatial_indices, feature_indices = np.arange(n_spatial), np.arange(n_feature)
#        if not args.spatial_selection is None:
#            spatial_indices = spatmp.indices_from_selection(spatial_labels, args.spatial_selection)
        if not args.feature_selection is None:
            feature_indices = spatmp.indices_from_selection(feature_labels, args.feature_selection)
        features = spatmp.filter_axis(features, [-2, -1], [spatial_indices, feature_indices])
        masks = spatmp.filter_axis(~gtmask, [-2, -1], [spatial_indices, feature_indices])
        means, stds = spatmp.filter_axis(
            [means, stds], [-2, -1], [spatmp.original.get("spatial_indices", args.partition), feature_indices]
        )
        spatial_labels = spatial_labels[spatial_indices]
        feature_labels = feature_labels[feature_indices]
        n_temporal, n_spatial, n_feature = features.shape
        spatial_label_field = spatmp.misc.spatial_label_field
        tmplab_frmt = spatmp.misc.temporal_label_format
        print(features.shape, masks.shape)
        print(spatial_labels)
        print(temporal_labels)
        plt = Plotting()
        if args.plot_type == "time-series":
            for i in range(n_spatial):
                for j in range(n_feature):
                    fig, ax = plt.subplots(size=(8, 4))
                    xtick_labels = None
                    if args.plot_args.xticks:
                        xtick_labels = util.pretty_temporal_labels(temporal_labels, tmplab_frmt)
                    if args.plot_args.mean:
                        if args.transform == "identity":
                            plt.plot_axis(means[i,j], "x", ax=ax, color="g", alpha=1/2, label="$\mu$")
                        elif args.transform == "zscore":
                            plt.plot_axis(0, "x", ax=ax, color="g", alpha=1/2, label="$\mu$")
                        else:
                            raise ValueError()
                    if args.plot_args.std:
                        if args.transform == "identity":
                            plt.plot_axis(means[i,j]-stds[i,j], "x", ax=ax, color="r", alpha=1/2)
                            plt.plot_axis(means[i,j]+stds[i,j], "x", ax=ax, color="r", alpha=1/2, label="$\mu \pm \sigma$")
                        elif args.transform == "zscore":
                            plt.plot_axis(-1, "x", ax=ax, color="r", alpha=1/2)
                            plt.plot_axis(1, "x", ax=ax, color="r", alpha=1/2, label="$\mu \pm \sigma$")
                        else:
                            raise ValueError()
                    plt.plot_time_series(
                        None, features[:,i,j], xtick_labels, mask=masks[:,i,j], 
                        ax=ax, linewidth=args.plot_args.lw
                    )
                    if xtick_labels is None:
                        plt.xticks(None, [], ax=ax)
                    plt.labels("Time", feature_ylabel(feature_labels[j]), ax=ax)
                    plt.title("%s %s" % (spatial_label_field.capitalize(), spatial_labels[i]), ax=ax)
                    plt.legend(ax=ax)
                    path = os.sep.join(
                        [
                            result_dir, 
                            "TS_part[%s]_tran[%s]_feat[%s]_node[%s].png" % (
                                args.partition, 
                                args.transform, 
                                feature_labels[j], 
                                spatial_labels[i]
                            )
                        ]
                    )
                    plt.save_figure(path, fig=fig)
        elif args.plot_type == "time-series-ensemble":
            for j in range(n_feature):
                fig, ax = plt.subplots(size=(8, 4))
                for i in range(n_spatial):
                    xtick_labels = temporal_labels if args.plot_args.xticks else None
                    plt.plot_time_series(
                        None, features[:,i,j], xtick_labels, mask=masks[:,i,j], 
                        ax=ax, label=spatial_labels[i], linewidth=args.plot_args.lw
                    )
                plt.labels("Time", feature_labels[j], ax=ax)
                plt.legend(ax=ax)
                path = os.sep.join(
                    [
                        result_dir, 
                        "TSE_part[%s]_tran[%s]_feat[%s]_sel[%s].png" % (
                            args.partition, 
                            args.transform, 
                            feature_labels[j], 
                            args.spatial_selection, 
                        )
                    ]
                )
                plt.save_figure(path, fig=fig)
                plt.close(fig)
        elif args.plot_type == "mean-heatmap":
                plt.plot_heatmap(
                    means.T, 
                    xtick_labels=(spatial_labels if n_spatial < 25 else ""), 
                    ytick_labels=feature_labels, 
                    xlabel=spatial_label_field.capitalize(), 
                    ylabel="Features", 
                    plot_cbar=args.plot_args.cbar, 
                )
                path = os.sep.join(
                    [
                        result_dir, 
                        "StatHeatmap_Stat[%s]_Trans[%s]_Spat[%s].png" % (
                            "mean", args.transform, args.spatial_selection
                        )
                    ]
                )
                plt.save_figure(path)
                plt.close(fig)
        elif args.plot_type == "std-heatmap":
                plt.plot_heatmap(
                    stds.T, 
                    xtick_labels=(spatial_labels if n_spatial < 25 else ""), 
                    ytick_labels=feature_labels, 
                    xlabel=spatial_label_field.capitalize(), 
                    ylabel="Features", 
                    plot_cbar=args.plot_args.cbar, 
                )
                path = os.sep.join(
                    [
                        result_dir, 
                        "StatHeatmap_Stat[%s]_Trans[%s]_Spat[%s].png" % (
                            "std", args.transform, args.spatial_selection
                        )
                    ]
                )
                plt.save_figure(path)
                plt.close(fig)
        elif args.plot_type == "histogram":
            for i in range(n_feature):
                histograms, bins = Probability.compute_histograms(
                    features[:,:,i], args.n_bins, args.bin_interval, return_bins=1
                )
                for j in range(n_spatial):
                    if args.dataset in ["littleriver"]:
                        fig, ax = plt.subplots(size=(10, 10))
                    elif args.dataset in ["wabashriver-swat"]:
                        fig, ax = plt.subplots(size=(100, 100))
                    else:
                        fig, ax = plt.subplots(size=(8,4))
                    print(bins.shape)
                    print(bins)
                    print(histograms[j,:].shape)
                    print(histograms[j,:])
                    plt.plot_bar(
                        None, 
                        histograms[j,:], 
                        ax=ax, 
                    )
                path = os.sep.join(
                    [
                        result_dir, 
                        "H_part[%s]_tran[%s]_feat[%s]_spat[%s].png" % (
                            args.partition, args.transform, feature_labels[i], spatial_labels[j]
                        )
                    ]
                )
                plt.xticks(np.arange(args.n_bins+1)-0.5, np.round(bins, 2), ax=ax, rotation=60)
                plt.xlabel("Bins" if args.transform=="identity" else "Bins (%s)" % (args.transform), ax=ax)
                plt.ylabel("No. Samples", ax=ax)
                plt.title(feature_labels[i], ax=ax)
                plt.save_figure(path)
        elif args.plot_type == "partition-histograms":
            train_features = spatmp.filter_axis(
                spatmp.transformed.original.get("features", "train"), 
                [-2, -1], 
                [spatial_indices, feature_indices]
            )
            valid_features = spatmp.filter_axis(
                spatmp.transformed.original.get("features", "valid"), 
                [-2, -1], 
                [spatial_indices, feature_indices]
            )
            test_features = spatmp.filter_axis(
                spatmp.transformed.original.get("features", "test"), 
                [-2, -1], 
                [spatial_indices, feature_indices]
            )
            for i in range(n_feature):
                bin_interval = [np.min(features[:,:,i]), np.max(features[:,:,i])]
                train_histograms, train_bins = Probability.compute_histograms(
                    train_features[:,:,i], args.n_bins, bin_interval, return_bins=1
                )
                valid_histograms, valid_bins = Probability.compute_histograms(
                    valid_features[:,:,i], args.n_bins, bin_interval, return_bins=1
                )
                test_histograms, test_bins = Probability.compute_histograms(
                    test_features[:,:,i], args.n_bins, bin_interval, return_bins=1
                )
                train_histograms = train_histograms / np.sum(train_histograms, -1)[:,None]
                valid_histograms = valid_histograms / np.sum(valid_histograms, -1)[:,None]
                test_histograms = test_histograms / np.sum(test_histograms, -1)[:,None]
                for j in range(n_spatial):
                    fig, ax = plt.subplots(size=(8,4))
                    plt.plot_bar(
                        np.arange(args.n_bins)-1/5, 
                        train_histograms[j,:], 
                        ax=ax, 
                        label="train", 
                        alpha=0.5, 
                        width=0.4, 
                    )
                    plt.plot_bar(
                        np.arange(args.n_bins), 
                        valid_histograms[j,:], 
                        ax=ax, 
                        label="valid", 
                        alpha=0.5, 
                        width=0.4, 
                    )
                    plt.plot_bar(
                        np.arange(args.n_bins)+1/5, 
                        test_histograms[j,:], 
                        ax=ax, 
                        label="test", 
                        alpha=0.5, 
                        width=0.4, 
                    )
                    plt.xticks(np.arange(args.n_bins+1)-0.5, np.round(train_bins, 2), ax=ax, rotation=60)
                    plt.xlabel("Bins" if args.transform=="identity" else "Bins (%s)" % (args.transform), ax=ax)
                    plt.ylabel("Frequency", ax=ax)
                    plt.title(feature_labels[i], ax=ax)
                    plt.legend(ax=ax)
                    path = os.sep.join(
                        [
                            result_dir, 
                            "PH_tran[%s]_feat[%s]_spat[%s].png" % (
                                args.transform, feature_labels[i], spatial_labels[j]
                            )
                        ]
                    )
                    plt.save_figure(path)
                    plt.close(fig)
        elif args.plot_type == "histogram-heatmap":
            for i in range(n_feature):
                if args.dataset in ["littleriver"]:
                    fig, ax = plt.subplots(size=(10, 10))
                elif args.dataset in ["wabashriver-swat"]:
                    fig, ax = plt.subplots(size=(100, 100))
                yticks = []
                if n_spatial <= 25:
                    yticks = spatial_labels
                histograms, bins = Probability.compute_histograms(
                    features[:,:,i], args.n_bins, args.bin_interval, return_bins=1
                )
                print(histograms.shape)
                bins = bins.round(3)
                yticks = []
                if args.yticks and n_spatial <= 25:
                    yticks = spatial_labels[index]
                xticks = []
                if args.xticks and args.n_bins <= 25:
                    xticks = bins
                print(feature_labels[i], bins)
                if args.debug:
                    print(feature_labels[i])
                    print(spatial_labels)
                    print(histograms)
                plt.plot_heatmap(
                    histograms, 
                    xtick_locs=np.arange(len(bins)) - 0.5, 
                    xtick_labels=xticks, 
                    ytick_labels=yticks, 
                    xlabel=("Bins" if args.xlabel else ""), 
                    ylabel=("%ss" % (spatial_label_field.capitalize()) if args.ylabel else ""), 
                    plot_cbar=args.plot_args.cbar, 
                    cbar_label=("No. Samples" if args.plot_args.cbar else ""), 
                    cmap="inferno", 
                    ax=ax, 
                )
                path = os.sep.join(
                    [
                        result_dir, 
                        "HHM_Trans[%s]_Feat[%s]_Spat[%s].png" % (
                            args.transform, feature_labels[i], args.spatial_selection
                        )
                    ]
                )
                plt.save_figure(path)
        elif args.plot_type == "histogram-clusterheatmap":
            for i in range(n_feature):
                fig, ax = plt.subplots(size=(100, 100))
                print(feature_labels[i])
                histograms, bins = Probability.compute_histograms(
                    features[:,:,i], args.n_bins, args.bin_interval, return_bins=1
                )
                print(histograms.shape)
                print(np.min(bins), np.max(bins))
                bins = bins.round(3)
                cluster_index = Clustering.cluster(histograms, "Agglomerative", 8)
                index = np.concatenate([np.where(cluster_index==i)[0] for i in range(8)])
                M = histograms[index,:]
                if 1:
                    yticks = np.concatenate([["%d"%(i)]+[""]*(sum(cluster_index==i)-1) for i in range(8)])
                else:
                    yticks = []
                    if args.yticks and n_spatial <= 25:
                        yticks = spatial_labels[index]
                xticks = []
                if args.xticks and args.n_bins <= 25:
                    xticks = bins
                if 1:
                    plt.plot_heatmap(
                        M, 
                        xtick_locs=np.arange(len(bins)) - 0.5, 
                        xtick_labels=xticks, 
                        ytick_labels=yticks, 
                        xlabel=("Bins" if args.xlabel else ""), 
                        ylabel=("%ss" % (spatial_label_field.capitalize()) if args.ylabel else ""), 
                        plot_cbar=args.plot_args.cbar, 
                        cbar_label=("No. Samples" if args.plot_args.cbar else ""), 
                        cmap="inferno", 
                    )
                else:
                    plt.plot_cluster_heatmap(histograms, ytick_labels=yticks, rowcol_cluster=(1, 0))
                path = os.sep.join(
                    [
                        result_dir, 
                        "HCHM_Trans[%s]_Feat[%s]_Spat[%s].png" % (
                            args.transform, feature_labels[i], args.spatial_selection
                        )
                    ]
                )
                plt.save_figure(path)
        else:
            raise NotImplementedError(args.plot_type)


class GraphVisualization(Analysis):

    def default_args(self):
        return Container().set(
            [
                "dataset", 
                "node_position_features", 
                "node_size_feature", 
                "node_size_mult", 
                "node_size_root", 
                "shapes", 
                "partition", 
                "figsize", 
                "plot_args", 
            ],
            [
                "littleriver", 
                ["lon", "lat"], 
                None, 
                300, 
                4, 
                None, 
                [None, "train"], 
                [32, 18], 
                Container().set(
                    [
                        "edges", 
                        "labels", 
                    ], 
                    [
                        True, 
                        True, 
                    ], 
                    multi_value=True
                ), 
            ],
            multi_value=True
        )

    def steps(self, args):
        from Variables import Variables
        from Data.Data import Data
        args = self.get_args(args)
        #   load dataset used in the target experiment
        var = Variables()
        var.principle_data_type = "spatial"
        var.meta.load_spatiotemporal = False
        dataset = Data(var).get("dataset", "train")
        # Setup for call to plotting routine
        path = os.path.join(
            self.result_dir(),
            "Graph_Dataset[%s]_Args[%s].png" % (args.dataset, ",".join(map(str, args.plot_args.get_values())))
        )
        visualize_graph(dataset, args.partition, args, path)


class Predictions(Analysis):

    def cache_dir(self):
        return "Experimentation/DEV/LittleRiver"

    def model(self):
        return ["DEV", "THSTM"]

    def steps(self, args):
        from Plotting import Plotting
        from Data.Data import Data
        from Data.DataSelection import DataSelection
        from Variables import Variables
        from matplotlib import pyplot
        from matplotlib.collections import LineCollection
        #
        _args = Container().set(
            [
                "cache_dir", 
                "model", 
                "partition", 
                "spatial_element", 
                "xlim", 
                "y_kwargs", 
                "yhat_kwargs", 
                "plot_pred_intervals", 
                "plot_error_line", 
                "error_line_res", 
                "error_line_cbar", 
            ], 
            [
                self.cache_dir(), 
                self.model(), 
                "test", 
                0, 
                [0, 0.5], 
                {"color": "k"}, 
                [{"alpha": 0.8}, {"alpha": 0.4}], 
                True, 
                True, 
                "full", # or "window"
                False, 
            ], 
            multi_value=1, 
        )
        args = _args.merge(args)
        if isinstance(args.model, str):
            args.model = [args.model]
        if isinstance(args.yhat_kwargs, dict):
            args.yhat_kwargs = [args.yhat_kwargs for _ in args.model]
        print(args)
        #
        sel = DataSelection()
        chkpt_dir = os.sep.join([args.cache_dir, "Checkpoints"])
        eval_dir = os.sep.join([args.cache_dir, "Evaluations"])
        cache = gather.get_cache(eval_dir, chkpt_dir)
        #
        # Load model predictions
        model_ids = []
        chkpts = []
        for model in args.model:
            model_id = gather.find_model_id(cache, model)#, where=["response_features", "==", [feature]])
            path = os.sep.join(
                [eval_dir, model, model_id, "Evaluation_Partition[%s].pkl" % (args.partition)]
            )
            chkpt = util.from_cache(path)
            model_ids.append(model_id)
            chkpts.append(chkpt)
        # Load data
        var = Variables().merge(chkpt.var)
        datasets = Data(var)
        dataset = datasets.train__dataset
        dataset_name = dataset.name
        spatmp = dataset.spatiotemporal
        Ti, To = var.mapping.temporal_mapping
        horizon = var.mapping.horizon
        # Load groundtruth
        temporal_labels = spatmp.original.get("temporal_labels", args.partition)
        spatial_labels = spatmp.original.get("spatial_labels", args.partition)
        feature = spatmp.misc.response_features[0]
        if isinstance(args.spatial_element, str) and args.spatial_element == "*":
            args.spatial_element = spatial_labels
        elif not isinstance(args.spatial_element, list):
            args.spatial_element = [args.spatial_element]
        print(args)
        for spatial_element in args.spatial_element:
            if isinstance(spatial_element, int):
                spatial_element = spatial_labels[spatial_element]
            spatial_idx = sel.indices_from_selection(spatial_labels, ["literal", spatial_element])[0]
            feature_idx = spatmp.misc.feature_index_map[feature]
            y = spatmp.original.get("response_features", args.partition)[:,spatial_idx,0]
            gtmask = spatmp.original.get("gtmask", args.partition)
            y_mask = ~gtmask[:,spatial_idx,feature_idx]
            # Load model predictions
            yhats = []
            yhat_masks = []
            for i, chkpt in enumerate(chkpts):
                _, _gtmask = spatmp.original.to_windows(gtmask, Ti, To)
                yhat = chkpt.Yhat
                _ = util.contiguous_window_indices(yhat.shape[0], To)
                yhat = np.reshape(yhat[_,:,spatial_idx,0], -1)
                yhat_mask = np.reshape(~_gtmask[_,:,spatial_idx,feature_idx], -1)
                yhats.append(yhat)
                yhat_masks.append(yhat_mask)
            # Plot it all
            result_dir = os.sep.join([self.result_dir(), dataset.name])
            os.makedirs(result_dir, exist_ok=1)
            plt = Plotting()
            fig, ax = plt.subplots(size=(8, 4))
            if args.plot_pred_intervals:
                for loc in np.arange(Ti+horizon-1, len(y), To):
                    plt.plot_axis(loc, "y", ax=ax, linewidth=0.1, alpha=0.5)
            if args.plot_error_line:
                x = np.arange(Ti, len(y)) - 0.5
                _y = y[Ti:len(yhats[0])+Ti]
                errs = [np.abs(_y - yhat) for yhat in yhats]
                if args.error_line_res == "window":
                    errs = [np.repeat(np.convolve(err, np.ones(To)/To, "valid")[::To], To) for err in errs]
                _ = np.percentile(np.concatenate(errs), [90, 95, 99])
                norm = pyplot.Normalize(0, _[1])
                for i, err in enumerate(errs):
                    _y = np.ones(len(x)) * (i+1) * -.0125 * (np.max(y) - np.min(y))
                    points = np.array([x, _y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap="inferno", norm=norm)
                    lc.set_array(err)
                    lc.set_linewidth(2)
                    line = ax.add_collection(lc)
                if args.error_line_cbar:
                    fig.colorbar(line, ax=ax)
            plt.plot_time_series(
                None, y, temporal_labels=temporal_labels, mask=y_mask, 
                ax=ax, label=feature, **args.y_kwargs
            )
            locs = np.arange(len(yhats[0])) + Ti + horizon - 1
            for i, yhat in enumerate(yhats):
                plt.plot_time_series(
                    locs, yhat, mask=yhat_masks[i], 
                    ax=ax, label=args.model[i], **args.yhat_kwargs[i]
                )
            if not args.xlim is None:
                left, right = args.xlim
                if isinstance(left, float):
                    left = int(left * len(y))
                if isinstance(right, float):
                    right = int(right * len(y))
                plt.xlim(left, right, ax=ax)
            plt.legend(ax=ax)
            plt.title("$%d \\rightarrow %d$" % tuple(var.mapping.temporal_mapping), ax=ax)
            # Save
            path = os.sep.join(
                [
                    result_dir, 
                    "Pred_Mod[%s]_Mid[%s]_Part[%s]_Feat[%s]_Node[%s].png" % (
                        ",".join(args.model), ",".join(model_ids), args.partition, feature, spatial_element
                    )
                ]
            )
            plt.save_figure(path)


class Clustering(Analysis):

    def get_exp(self, exp_id):
        raise NotImplementedError()

    def steps(self, args):
        import matplotlib.pyplot as plt
        from Variables import Variables
        from Data.Data import Data
        from Data import SpatiotemporalData
        from Data import Decomposition
        from Plotting import Plotting
        #
        _args = Container().set(
            [
                "exp_id", 
                "subexp_id", 
                "dataset", 
                "data_type", 
                "features", 
                "predictor_features", 
                "response_features", 
                "rep", 
                "alg", 
                "n_clusters", 
                "vis", 
                "norm_vis", 
                "n_bins", 
                "bin_interval", 
                "color_index", 
                "plot_labels", 
                "plot_kwargs", 
                "debug", 
            ],
            [
                None, 
                0, 
                "littleriver", 
                "spatiotemporal", 
                ["~", "date"], 
                None, 
                None, 
#                ["FLOW_OUTcms"], 
                "histogram", 
                "Agglomerative", 
                3, 
                "TSNE", 
                False, 
                100, 
                None, 
                0, 
                False, 
                {"ticks": False}, 
                0, 
            ],
            multi_value=True
        )
        args = _args.merge(args)
        if args.predictor_features is None:
            args.predictor_features = args.features
        if args.response_features is None:
            args.response_features = args.features
        # Start
        #    Setup variables
        var = Variables()
        if not args.exp_id is None:
            self.exp = self.get_exp(args.exp_id)
            exp_args = self.exp.jobs[args.subexp_id].work
            var = var.merge(exp_args)
        else:
            var.execution.set("dataset", args.dataset, "*")
        if not args.predictor_features is None:
            if args.predictor_features[0] == "~":
                args.predictor_features = util.list_subtract(
                    var.datasets.get(args.dataset).get(args.data_type).loading.feature_fields, 
                    list(args.predictor_features[1:])
                )
        if not args.response_features is None:
            if args.response_features[0] == "~":
                args.response_features = util.list_subtract(
                    var.datasets.get(args.dataset).get(args.data_type).loading.feature_fields, 
                    list(args.response_features[1:])
                )
        var.mapping.spatiotemporal.predictor_features = args.predictor_features
        var.mapping.spatiotemporal.response_features = args.response_features
        #    Load data
        result_dir = os.sep.join([self.result_dir(), var.execution.get("dataset", "train")])
        data = Data(var)
        dataset = data.get("dataset", "train")
        spa = dataset.spatial
        spatmp = dataset.spatiotemporal
        spalabs = spatmp.original.spatial_labels
        os.makedirs(result_dir, exist_ok=True)
        #    Compute clusters
        x, clustering, cluster_index = SpatiotemporalData.cluster(
            spatmp, args.alg, args.n_clusters, args.rep, bins=args.n_bins, lims=args.bin_interval, debug=args.debug
        )
        print("x =", x.shape)
        #    Create xy points from representation x
        if args.vis == "coords":
            if args.dataset in ["littleriver", "wabashriver-swat"]:
                index = util.get_dict_values(spa.misc.feature_index_map, ["lon", "lat"])
            elif args.dataset in ["metr-la", "pems-bay"]:
                index = util.get_dict_values(spa.misc.feature_index_map, ["longitude", "latitude"])
            xy = spa.original.features[:,index]
        else:
            xy = x
            if x.shape[-1] < 2:
                raise ValueError(x.shape)
            elif x.shape[-1] == 2:
                args.vis = None
            else:
                vis_alg = args.vis
                if "-" in vis_alg:
                    vis_alg, vis_target = vis_alg.split("-")
                    if vis_target == "histogram":
                        x = SpatiotemporalData.compute_histograms(
                            spatmp.transformed.original.get("predictor_features", "train"), bins, lims
                        )
                    else:
                        raise NotImplementedError(args.vis)
                xy = Decomposition.reduce(x, 2, vis_alg)
        if args.norm_vis:
            xy = (xy - np.min(xy, 0)) / (np.max(xy, 0) - np.min(xy, 0))
        #    Visualize as a scatter-plot
        plt = Plotting()
        fig, ax = plt.subplots(size=(8, 8))
        colors = np.array(plt.default_colors)
        if isinstance(args.color_index, int):
            color_index = (np.arange(len(colors)) + args.color_index) % len(colors)
        elif isinstance(args.color_index, list):
            color_index = args.color_index
        else:
            raise TypeError(args.color_index)
        colors = colors[color_index]
        for k in range(args.n_clusters):
            mask = cluster_index == k
            labels = None if not args.plot_labels else spalabs[mask]
            labels = [str(k) for _ in range(len(mask))]
            plt.plot_scatter(
                xy[mask,0], xy[mask,1], labels=labels, 
                ax=ax, scatter_kwargs={"color": colors[k%len(colors)], "label": "$C_{%d}$" % (k)}
            )
        if not args.plot_kwargs.get("ticks", True):
            plt.xticks([])
            plt.yticks([])
        plt.legend(ax=ax)
        # Save everything
        features = spatmp.misc.predictor_features
        features_str = ",".join(features) if isinstance(features, list) else str(features)
        path = os.sep.join(
            [
                result_dir, 
                "Clusters_Feat[%s]_Rep[%s]_Alg[%s]_K[%d]_Vis[%s].png" % (
                    features_str, args.rep, args.alg, args.n_clusters, args.vis, 
                )
            ]
        )
        plt.save_figure(path, fig=fig)
        plt.close()
        df = pd.DataFrame(
            {
                spatmp.misc.spatial_label_field: spatmp.original.get("spatial_labels", "train"), 
                "cluster": cluster_index
            }
        )
        path = os.sep.join(
            [
                result_dir, 
                "Clusters_Feat[%s]_Rep[%s]_Alg[%s]_K[%d].csv" % (
                    features_str, args.rep, args.alg, args.n_clusters, 
                )
            ]
        )
        df.to_csv(path, index=False)
        ids, counts = np.unique(cluster_index, return_counts=True)
        df = pd.DataFrame({"Cluster_ID": ids, "Count": counts})
        path = os.sep.join(
            [
                result_dir, 
                "ClusterCounts_Feat[%s]_Rep[%s]_Alg[%s]_K[%d].csv" % (
                    features_str, args.rep, args.alg, args.n_clusters, 
                )
            ]
        )
        df.to_csv(path, index=False)
        print(df)


class Hierarchy(Analysis):

    def steps(self, args):
        import matplotlib.pyplot as plt
        from Variables import Variables
        from Data.Data import Data
        from Data import SpatiotemporalData
        from Data import Decomposition
        from Plotting import Plotting
        #
        _args = Container().set(
            [
                "exp", 
                "exp_id", 
                "dataset", 
                "features", 
                "predictor_features", 
                "response_features", 
                "rep", 
                "alg", 
                "vis", 
                "bins", 
                "lims", 
                "label_nodes", 
                "debug", 
            ],
            [
                None, 
                0, 
                "littleriver", 
                None, 
                None, 
                None, 
#                ["FLOW_OUTcms"], 
                "correlation", 
                "Agglomerative", 
                "dot", # "twopi"
                100, 
                None, 
                False, 
                0, 
            ],
            multi_value=True
        )
        args = _args.merge(args)
        if args.predictor_features is None:
            args.predictor_features = args.features
        if args.response_features is None:
            args.response_features = args.features
        # Start
        #    Setup variables
        var = Variables()
        if not args.exp is None:
            self.exp = globals()["Experiment__%s" % (str(args.exp).replace(".", "_"))]()
            exp_args = self.exp.jobs[args.exp_id].work
            var = var.merge(exp_args)
        else:
            var.execution.set("dataset", args.dataset, "*")
        if not args.predictor_features is None:
            if args.predictor_features[0] == "~":
                args.predictor_features = util.list_subtract(
                    var.datasets.get(args.dataset).get(args.data_type).loading.feature_fields, 
                    list(args.predictor_features[1:])
                )
        if not args.response_features is None:
            if args.response_features[0] == "~":
                args.response_features = util.list_subtract(
                    var.datasets.get(args.dataset).get(args.data_type).loading.feature_fields, 
                    list(args.response_features[1:])
                )
        var.mapping.spatiotemporal.predictor_features = args.predictor_features
        var.mapping.spatiotemporal.response_features = args.response_features
        #  Load data
        result_dir = os.sep.join([self.result_dir(), var.execution.get("dataset", "train")])
        data = Data(var)
        dataset = data.get("dataset", "train")
        spa = dataset.spatial
        spatmp = dataset.spatiotemporal
        n_spatial = spatmp.original.n_spatial
        spalabs = spatmp.original.get("spatial_labels", "train")
        os.makedirs(result_dir, exist_ok=True)
        #  Compute clusters
        x, clustering, tree_index = SpatiotemporalData.treeify(
            spatmp, args.alg, args.rep, bins=args.bins, lims=args.lims, features="predictor", debug=args.debug
        )
        #  Convert to Graph
        import networkx as nx
        if args.debug:
            for i in range(len(tree_index)):
                print(len(tree_index)+i+1, tree_index[i,:])
        G = nx.Graph()
        N, E = len(np.unique(tree_index)), len(tree_index)
        for i in range(N, -1, -1):
            nid = i
            if i < n_spatial:
                nid = spalabs[i]
            G.add_node(nid)
        if args.debug:
            print(G)
            print(N, E)
        for i in range(E):
            nid = i
            if i < n_spatial:
                nid = spalabs[i]
            u, v = tree_index[i,0], tree_index[i,1]
            u_nid, v_nid = u, v
            if u < n_spatial:
                u_nid = spalabs[u]
            if v < n_spatial:
                v_nid = spalabs[v]
            if args.debug:
                print("### %d ###" % (i))
                print("u=%s, v=%s" % (u_nid, v_nid))
            G.add_edge(u_nid, n_spatial+i)
            G.add_edge(v_nid, n_spatial+i)
        if args.debug:
            print(list(G))
            print(G)
        #  Plot the graph
        if args.vis == "coords":
            index = util.get_dict_values(spa.misc.feature_index_map, ["lon", "lat"])
            xy = spa.original.features[:,index]
            pos = {}
            for i in range(len(xy)):
                pos[i] = xy[i,:]
            print(pos)
            for i in range(E):
                u, v = tree_index[i,0], tree_index[i,1]
                print("u=%d, v=%d" % (u, v))
                pos[n_spatial+i] = (pos[u] + pos[v]) / 2
                print(pos[n_spatial+i])
        else:
            pos = nx.nx_agraph.graphviz_layout(G, prog=args.vis)
        # Save everything
        fig, ax = plt.subplots(1, figsize=(10,5))
        if args.dataset == "wabashriver-swat":
            fig, ax = plt.subplots(1, figsize=(100,50))
        if args.dataset in ["metr-la", "pems-bay"]:
            fig, ax = plt.subplots(1, figsize=(16,8))
        if args.dataset in ["electricity"]:
            fig, ax = plt.subplots(1, figsize=(20,10))
        if args.dataset in ["solar-energy"]:
            fig, ax = plt.subplots(1, figsize=(16,8))
        features = spatmp.misc.predictor_features
        features_str = ",".join(features) if isinstance(features, list) else str(features)
        path = os.sep.join(
            [
                result_dir, 
                "Hierarchy_Feat[%s]_Rep[%s]_Alg[%s]_Vis[%s].png" % (
                    features_str, args.rep, args.alg, args.vis, 
                )
            ]
        )
        size = 250
        node_size = np.concatenate((2*size*np.ones(n_spatial-1), size*np.ones(n_spatial)))
        if args.dataset in ["metr-la", "pems-bay"]:
            node_size *= 0.1
        if args.dataset in ["electricity", "solar-energy"]:
            node_size *= 0.1
        node_color = np.concatenate((["r" for _ in range(n_spatial-1)], ["b" for _ in range(n_spatial)]))
        nx.draw(G, pos, ax=ax, node_size=node_size, node_color=node_color, with_labels=args.label_nodes, font_color="w")
        plt.savefig(path)
        plt.close()


class SpatialMetrics(Analysis):

    def cache_dir_i(self):
        return "."

    def cache_dir_j(self):
        return "."

    def exp(self):
        return None

    def subexp_id(self):
        return 0

    def default_args(self):
        return Container().set(
            [
                "cache_dir_i",
                "cache_dir_j", 
                "model_i",
                "model_j",
                "where_i", 
                "where_j", 
                "metric",
                "partition",
                "response_feature",
                "plot_args",
            ],
            [
                self.cache_dir_i(), 
                self.cache_dir_j(), 
                "THSTM",
                "DEV",
                [],
                [], 
                ["RMSE"],
                "test",
                None, 
                Container().set(
                    ["x_axis", "y_axis", "z_axis", "plot_options"], 
                    ["metric", "%diff", "cov", ["trend"]], 
                    multi_value=True, 
                )
            ],
            multi_value=True
        )

    def steps(self, args):
        from Variables import Variables
        from Data.Data import Data
        # Setup variables + args
        args = self.get_args(args)
        if isinstance(args.where_i, str) and args.where_i == "where_j":
            args.where_i = args.where_j
        if isinstance(args.where_j, str) and args.where_j == "where_i":
            args.where_j = args.where_i
        # Load all cache data
        eval_dir_i = os.sep.join([args.cache_dir_i, "Evaluations"])
        chkpt_dir_i = os.sep.join([args.cache_dir_i, "Checkpoints"])
        cache_i = gather.get_cache(eval_dir_i, chkpt_dir_i)
        eval_dir_j = os.sep.join([args.cache_dir_i, "Evaluations"])
        chkpt_dir_j = os.sep.join([args.cache_dir_i, "Checkpoints"])
        if args.cache_dir_j == args.cache_dir_i:
            cache_j = cache_i
        else:
            cache_j = gather.get_cache(eval_dir_j, chkpt_dir_j)
        # Find model_i and model_j metrics according to their respective conditions (where_i & where_j)
        #   Find model_i errors
        model_i_id = gather.find_model_id(cache_i, args.model_i, args.where_i)
        model_i_metrics = cache_i.errors.get(model_i_id, path=[args.model_i])
        #   Find model_j errors
        model_j_id = gather.find_model_id(cache_j, args.model_j, args.where_j)
        model_j_metrics = cache_j.errors.get(model_j_id, path=[args.model_j])
        # Setup for call to plotting routine
        plt = Plotting()
        #   get response feature if not specified
        if args.response_feature is None:
            metric, feature_metrics, partition = model_i_metrics[0]
            feature_label, error_dict, partition = feature_metrics[0]
            args.response_feature = feature_label
        #   broadcast metric to list
        if isinstance(args.metric, str):
            if args.metric == "*":
                args.metric = cache.errors.get(args.model_i)[0][1].get_names()
            else:
                args.metric = [args.metric]
        elif not isinstance(args.metric, list):
            raise ValueError(args.metric)
        #   broadcast x_axis option
        if args.plot_args.x_axis == "*":
            args.plot_args.x_axis = ["in-degree", "metric", "stddev"]
        elif isinstance(args.plot_args.x_axis, str):
            args.plot_args.x_axis = [args.plot_args.x_axis]
        elif not isinstance(args.plot_args.x_axis, list):
            raise ValueError(args.plot_args.x_axis)
        #   load dataset used in the target experiment
        if self.exp() is None:
            datasets = Data(Variables())
        else:
            datasets = Data(Variables().merge(self.exp().jobs[self.subexp_id()].work))
        dataset_name = datasets.get("dataset", "train").name
        # Create all plots
        result_dir = os.sep.join([self.result_dir(), dataset_name])
        os.makedirs(result_dir, exist_ok=True)
        x_axes = list(args.plot_args.x_axis)
        for metric in list(args.metric):
            args.metric = metric
            for x_axis in x_axes:
                args.plot_args.x_axis = x_axis
                fig, ax = plt.subplots(1, size=(8, 4))
                path = os.path.join(
                    result_dir,
                    "MS_Models[%s]_Metric[%s]_Args[%s].png" % (
                        ",".join(["%s(%s)" % (args.model_i, model_i_id), "%s(%s)" % (args.model_j, model_j_id)]),
                        args.metric,
                        ",".join(map(str, args.plot_args.get_values())),
                    )
                )
                plt.title(args.response_feature)
                plt.plot_metric_scatter(model_i_metrics, model_j_metrics, datasets, args, ax=ax)
                plt.save_figure(path, fig=fig)


class TemporalMetrics(Analysis):

    def cache_dir_i(self):
        return "."

    def cache_dir_j(self):
        return "."

    def model_i(self):
        return "THSTM"

    def model_j(self):
        return "DEV"

    def where_i(self):
        return []

    def where_j(self):
        return []

    def features(self):
        _ = None
        if not self.exp() is None:
            _ = np.unique([_.work.get("response_features")[0] for _ in self.exp().jobs])
        return _

    def exp(self):
        return None

    def subexp_id(self):
        return 0

    def partition(self):
        return "test"

    def default_args(self):
        return Container().set(
            [
                "cache_dir_i",
                "cache_dir_j", 
                "model_i",
                "model_j",
                "where_i", 
                "where_j", 
                "features",
                "metric",
                "partition",
                "reduction", 
                "plot_type", 
            ],
            [
                self.cache_dir_i(), 
                self.cache_dir_j(), 
                self.model_i(),
                self.model_j(),
                self.where_i(),
                self.where_j(), 
                self.features(), 
                ["AE"],
                self.partition(),
                self.reduction(), 
                "bar-diff", 
            ],
            multi_value=True
        )

    def steps(self, args):
        from Variables import Variables
        from Data.Data import Data
        from Data import SpatiotemporalData
        # Setup variables + args
        args = self.get_args(args)
        if isinstance(args.where_i, str) and args.where_i == "where_j":
            args.where_i = args.where_j
        if isinstance(args.where_j, str) and args.where_j == "where_i":
            args.where_j = args.where_i
        #   broadcast metric to list
        if isinstance(args.metric, str):
            if args.metric == "*":
                args.metric = ["AE", "SE"]
            else:
                args.metric = [args.metric]
        elif not isinstance(args.metric, list):
            raise ValueError(args.metric)
        # Load all cache data
        plt = Plotting()
        eval_dir_i = os.sep.join([args.cache_dir_i, "Evaluations"])
        chkpt_dir_i = os.sep.join([args.cache_dir_i, "Checkpoints"])
        cache_i = gather.get_cache(eval_dir_i, chkpt_dir_i, load_evals=1)
        eval_dir_j = os.sep.join([args.cache_dir_i, "Evaluations"])
        chkpt_dir_j = os.sep.join([args.cache_dir_i, "Checkpoints"])
        if args.cache_dir_j == args.cache_dir_i:
            cache_j = cache_i
        else:
            cache_j = gather.get_cache(eval_dir_j, chkpt_dir_j, load_evals=1)
        # Find model_i and model_j metrics according to their respective conditions (where_i & where_j)
        for feature in args.features:
            print(40*"#", feature, 40*"#")
            #   Find model_i errors
            model_i_id = gather.find_model_id(
                cache_i, args.model_i, "settings", 
                where=args.where_i+[["response_features","==",[feature]]], 
            )
            model_i_metrics = cache_i.errors.get(model_i_id, path=[args.model_i])
            #   Find model_j errors
            model_j_id = gather.find_model_id(
                cache_j, args.model_j, "settings", 
                where=args.where_i+[["response_features","==",[feature]]], 
            )
            model_j_metrics = cache_j.errors.get(model_j_id, path=[args.model_j])
            # Setup for call to plotting routine
            #   load predictions for each model
            [Ti, To], horizon = cache_i.get(
                ["temporal_mapping", "horizon"], path=["settings", args.model_i, model_i_id, "mapping"]
            )
            yhat_i = cache_i.get("yhat", args.partition, path=["evals", args.model_i, model_i_id])
            yhat_i = SpatiotemporalData.contiguify(yhat_i, To, horizon)[:,:,0]
            yhat_j = cache_j.get("yhat", args.partition, path=["evals", args.model_j, model_j_id])
            yhat_j = SpatiotemporalData.contiguify(yhat_j, To, horizon)[:,:,0]
            #   load dataset used in the target experiment
            datasets = Data(cache_i.get("var", path=["evals", args.model_j, model_j_id]))
            dataset_name = datasets.get("dataset", "train").name
            spatmp = datasets.get("dataset", "train").spatiotemporal
            n_spatial = spatmp.original.get("n_spatial", args.partition)
            spalabs = spatmp.original.get("spatial_labels", args.partition)
            y = spatmp.original.get("response_features", args.partition)[:,:,0]
            #    compute temporal metrics
            tms = []
            for metric in args.metric:
                tm_i = y[Ti:,:][:len(yhat_i),:] - yhat_i
                tm_j = y[Ti:,:][:len(yhat_j),:] - yhat_j
                if metric == "AE":
                    tm_i = np.abs(tm_i)
                    tm_j = np.abs(tm_j)
                elif metric == "SE":
                    tm_i = np.square(tm_i)
                    tm_j = np.square(tm_j)
                else:
                    raise NotImplementedError(metric)
                if not args.reduction is None:
                    index = util.sliding_window_indices(tm_i.shape[0], args.reduction, stride=args.reduction)
                    tm_i = np.mean(np.take(tm_i, index, 0), 1)
                    tm_j = np.mean(np.take(tm_j, index, 0), 1)
                tms.append([tm_i, tm_j])
            # Create all plots
            result_dir = os.sep.join([self.result_dir(), dataset_name])
            os.makedirs(result_dir, exist_ok=True)
            for i, metric in enumerate(args.metric):
                if "heatmap" in args.plot_type:
                    if "diff" in args.plot_type:
                        fig, ax = plt.subplots(size=(8, 1.5))
                        M = np.swapaxes(np.abs(tms[i][1] - tms[i][0]), 0, 1)
                        M = np.swapaxes(tms[i][1] - tms[i][0], 0, 1)
                        plt.plot_heatmap(M, cmap="coolwarm", plot_cbar=1, ax=ax)
                        plt.title(
                            "%s (%s $\\rightarrow$ %s)" % (feature, args.model_j, args.model_i), ax=ax
                        )
                    else:
                        fig, axes = plt.subplots(2, 1, size=(8, 1.5))
                        plt.plot_heatmap(np.swapaxes(tms[i][0], 0, 1), plot_cbar=0, ax=axes[0])
                        plt.plot_heatmap(np.swapaxes(tms[i][1], 0, 1), plot_cbar=0, ax=axes[1])
                        plt.ylabel(args.model_i, ax=axes[0])
                        plt.ylabel(args.model_j, ax=axes[1])
                        plt.title("%s (%s)" % (feature, metric), ax=axes[0])
                elif "line" in args.plot_type:
                    fig, ax = plt.subplots(size=(8, 4))
                    plt.style("grid", ax=ax, zorder=1)
                    if "diff" in args.plot_type:
                        plt.plot_axis(0, "x", ax=ax, linewidth=2/3, zorder=2)
                        M = tms[i][1] - tms[i][0]
                        for j in range(n_spatial):
                            plt.plot_line(None, M[:,j], label=spalabs[j], ax=ax)
                        plt.ylabel(
                            "$\\Delta$ %s (%s $\\rightarrow$ %s)" % (
                                metric, args.model_i, args.model_j
                            ), 
                                ax=ax
                        )
                    else:
                        for j in range(n_spatial):
                            plt.plot_line(None, tms[i][0], label="%s(%s)" % (args.model_i, spalabs[j]), ax=ax)
                            plt.plot_line(None, tms[i][1], label="%s(%s)" % (args.model_j, spalabs[j]), ax=ax)
                        plt.ylabel(
                            "$\\Delta$ %s (%s $\\rightarrow$ %s)" % (
                                metric, args.model_j, args.model_i
                            ), 
                                ax=ax
                        )
                    locs = np.linspace(0, len(tms[i][0]), 25, endpoint=False, dtype=int)
                    labels = locs if args.reduction is None else args.reduction * locs
                    plt.xticks(locs, labels, ax=ax, rotation=90)
                    plt.xlim(locs[0]-1, locs[-1]+1, ax=ax)
                    plt.xlabel("Time-step", ax=ax)
                    plt.title(feature, ax=ax)
                    plt.legend(ax=ax)
                elif "bar" in args.plot_type:
                    fig, ax = plt.subplots(size=(8, 4))
                    plt.style("grid", ax=ax, zorder=1)
                    if "diff" in args.plot_type:
                        plt.plot_axis(0, "x", ax=ax, linewidth=2/3, zorder=2)
                        M = tms[i][1] - tms[i][0]
                        for j in range(n_spatial):
                            plt.plot_bar(
                                None, M[:,j], ax=ax, bottom=None, label=spalabs[j], alpha=1/n_spatial, zorder=3
                            )
                        plt.ylabel(
                            "$\\Delta$ %s (%s $\\rightarrow$ %s)" % (
                                metric, args.model_i, args.model_j
                            ), 
                            ax=ax
                        )
                    else:
                        for j in range(n_spatial):
                            plt.plot_bar(None, tms[i][0], bottom=None, label="%s(%s)" % (args.model_i, spalabs[j]), ax=ax)
                            plt.plot_bar(None, tms[i][1], bottom=None, label="%s(%s)" % (args.model_j, spalabs[j]), ax=ax)
                        plt.ylabel(
                            "$\\Delta$ %s (%s $\\rightarrow$ %s)" % (
                                metric, args.model_j, args.model_i
                            ), 
                                ax=ax
                        )
                    locs = np.linspace(0, len(tms[i][0]), 25, endpoint=False, dtype=int)
                    labels = locs if args.reduction is None else args.reduction * locs + Ti + 1
                    plt.xticks(locs, labels, ax=ax, rotation=90)
                    plt.xlim(locs[0]-1, locs[-1]+1, ax=ax)
                    plt.xlabel("Time-step", ax=ax)
                    plt.title(feature, ax=ax)
                    plt.legend(ax=ax)
                path = os.path.join(
                    result_dir,
                    "TM_Models[%s]_Metric[%s]_Plot[%s].png" % (
                        ",".join(["%s(%s)" % (args.model_i, model_i_id), "%s(%s)" % (args.model_j, model_j_id)]),
                        metric,
                        args.plot_type, 
                    )
                )
                plt.save_figure(path, fig=fig)


class AMI(Analysis):

    def default_args(self):
        return Container().set(
            [
                "dataset", 
                "spatial_selection", 
                "features", 
                "predictor_features", 
                "response_features", 
                "lags", 
                "max_lag", 
                "legend", 
            ], 
            [
                "littleriver", 
                0, 
                None, 
                None, 
                None, 
                list(range(1, 31)), 
                -1, 
                True, 
            ], 
            multi_value=1
        )

    def steps(self, args):
        import time
        from Variables import Variables
        from Data.Data import Data
        from Data import Probability as P
        from Plotting import Plotting
        #
        args = self.get_args(args)
        if args.predictor_features is None:
            args.predictor_features = args.features
        if args.response_features is None:
            args.response_features = args.features
        if args.max_lag > 0:
            args.lags = np.arange(args.max_lag) + 1
        if not isinstance(args.lags, np.ndarray):
            args.lags = np.array(args.lags)
        var = Variables()
        var.execution.set("dataset", args.dataset, "*")
        var.mapping.spatiotemporal.predictor_features = args.predictor_features
        var.mapping.spatiotemporal.response_features = args.response_features
        #  Load data
        result_dir = os.sep.join([self.result_dir(), var.execution.get("dataset", "train")])
        result_dir = self.result_dir()
        data = Data(var)
        dataset = data.get("dataset", "train")
        spa = dataset.spatial
        spatmp = dataset.spatiotemporal
        spalabs = spatmp.original.get("spatial_labels", "train")
        spalab_field = spatmp.misc.spatial_label_field
        os.makedirs(result_dir, exist_ok=True)
        #
        x_features = spatmp.misc.predictor_features
        y_features = spatmp.misc.response_features
        x = spatmp.original.get("predictor_features", "train")
        y = spatmp.original.get("response_features", "train")
        spatial_index = args.spatial_selection
        if isinstance(args.spatial_selection, int):
            spatial_index = [args.spatial_selection]
        elif isinstance(args.spatial_selection[0], str):
            spatial_index = spatmp.indices_from_selection(spalabs, args.spatial_selection)
        n_spatial = len(spatial_index)
        n_lags = len(args.lags)
        plt = Plotting()
        fig, ax = plt.subplots(size=(8, 4))
        plt.style("grid", ax=ax)
        colors = plt.default_colors
        if 0:
            amis = []
            for s in spatial_index:
                ami = P.lag_mutual_information(y[:,s,:], x[:,s,:], args.lags)
                amis.append(ami)
            amis = np.stack(amis)
        else:
            _x = np.swapaxes(x, 0, 1)[spatial_index,:,:]
            _y = np.swapaxes(y, 0, 1)[spatial_index,:,:]
            s = time.time()
            amis = P.mp_lag_mutual_information(_y, _x, args.lags, n_proc=128)
            print(time.time() - s)
            #
        fm_index = []
        if x.shape[-1] > 1:
            for i in range(x.shape[-1]):
                ami = amis[0][:,0,i]
                ami = ami / np.max(ami)
                fm_idx = -1
                if args.debug:
                    print("ami(%s;%s) =" % (y_features, x_features[i]), ami)
                for j in range(ami.shape[0]-1):
                    if ami[j+1] > ami[j]:
                        fm_idx = j
                        break
                if fm_idx > -1:
                    plt.plot_axis(fm_idx, "y", ax=ax, linestyle=":", color=colors[i%8])
                    if len(args.lags) > 30:
                        plt.plot_text(fm_idx, ami[fm_idx], str(fm_idx+1), ax=ax, color="k", fontsize=10)
                plt.plot_line(
                    None, ami, 
                    ax=ax, label="I(%s,%s)" % (y_features[0], x_features[i]), color=colors[i%8]
                )
                fm_idx.append(fm_idx)
        else:
            for i, s in enumerate(spatial_index):
                ami = amis[i,:,0,0]
                if n_spatial > 1:
                    ami = ami / np.max(ami)
                if args.debug:
                    print("ami(%s;%s) =" % (y_features, spalabs[s]), ami)
                fm_idx = -1
                for j in range(len(ami)-1):
                    if ami[j+1] > ami[j]:
                        fm_idx = j
                        break
                if fm_idx > -1:
                    plt.plot_axis(fm_idx, "y", ax=ax, linestyle=":", color=colors[i%8])
                    if len(args.lags) > 30:
                        plt.plot_text(fm_idx, ami[fm_idx], str(fm_idx+1), ax=ax, color="k", fontsize=10)
                plt.plot_line(
                    None, ami, 
                    ax=ax, label="%s %s" % (spalab_field, spalabs[s]), color=colors[i%8]
                )
                fm_index.append(fm_idx)
            print("AVERAGE FIRST MINIMUM =", np.mean(fm_index)+1)
        plt.xticks(np.arange(len(args.lags)), args.lags, ax=ax)
        if len(args.lags) > 30:
            index = np.linspace(0, len(args.lags)-1, 9, 1, dtype=int)
            plt.xticks(index, args.lags[index], ax=ax)
        if args.legend and n_spatial <= 10:
            plt.legend(ax=ax)
        #
        _ = "*" if args.predictor_features is None else ",".join(args.predictor_features)
        __ = "*" if args.response_features is None else ",".join(args.response_features)
        path = os.sep.join([result_dir, "AMI_[%s]_[%s|%s].png" % (
            args.dataset, _, __
        )])
        plt.save_figure(path, fig)
        #
        if x.shape[-1] > 1:
            pass
        else:
            data_dict = {
                spalab_field: np.repeat(spalabs[spatial_index], n_lags), 
                "lag": np.tile(args.lags, n_spatial)
            }
            for feature in x_features:
                data_dict["%s_ami"%(feature)] = np.reshape(amis[:,:,0,0], -1)
        df = pd.DataFrame(data_dict)
        print(df)
        path = os.sep.join([result_dir, "AMI_[%s]_[%s|%s].csv" % (
            args.dataset, _, __
        )])
        df.to_csv(path, index=False)
        #
        if x.shape[-1] > 1:
            pass
        else:
            data_dict = {
                spalab_field: spalabs[spatial_index], 
                "%s_amifm"%(x_features[0]): np.array(fm_index) + 1, 
            }
        df = pd.DataFrame(data_dict)
        print(df)
        path = os.sep.join([result_dir, "AMIFM_[%s]_[%s|%s].csv" % (
            args.dataset, _, __
        )])
        df.to_csv(path, index=False)


class WatershedVis(Analysis):
    
    def default_args(self):
        return Container().set(
            ["dataset", "subbasin_shapes_path", "stream_shapes_path"], 
            [
                "littleriver", 
                os.sep.join(["Data", "LittleRiver", "Integration", "Acquired", "Little River", "Geographic", "basins", "basins"]), 
                os.sep.join(["Data", "LittleRiver", "Integration", "Acquired", "Little River", "Geographic", "Hydrology", "gis_streams"]), 
            ], 
            multi_value=1
        )

    def steps(self, args):
        import shapefile
        from Plotting import Plotting
        args = self.get_args(args)
        subbasin_shapes = shapefile.Reader(args.subbasin_shapes_path)
        print(subbasin_shapes)
        stream_shapes = None
        if not args.stream_shapes_path is None:
            stream_shapes = shapefile.Reader(args.stream_shapes_path)
        print(stream_shapes)
        # Start
        kwargs = {}
        if args.dataset == "littleriver":
            figsize = (7, 8)
            kwargs["sr_label_idx"] = 8
        if args.dataset == "wabashriver":
            figsize = (20, 20)
            kwargs["subbasin_lw"] = 0.625
            kwargs["stream_lw"] = 0.5
            kwargs["label_fontsize"] = 6
            kwargs["sr_label_idx"] = 0
        plt = Plotting()
        fig, ax = plt.subplots(size=figsize)
        plt.plot_watershed(subbasin_shapes, stream_shapes, ax=ax, **kwargs)
        path = os.sep.join([self.result_dir(), "Watershed_[%s].png" % (args.dataset)])
        plt.save_figure(path, fig)


class TIL(Analysis): # Temporal Interval Locator

    def default_args(self):
        return Container().set(
            [
                "dataset", 
                "features", 
                "window_len", 
                "window_stride", 
                "criterion", 
            ], 
            [
                "littleriver", 
                ["~", "date", "Timestamp", "dv_dt"], 
                10*365, 
                365, 
                "missing-values", 
            ], 
            multi_value=1
        )

    def steps(self, args):
        from Variables import Variables
        from Data.Data import Data
        from Plotting import Plotting
        import matplotlib as mpl
        args = self.get_args(args)
        #
        var = Variables()
        if args.features[0] == "~":
            args.features = util.list_subtract(
                var.datasets.get(args.dataset).spatiotemporal.loading.feature_fields, args.features
            )
        var.execution.set("dataset", args.dataset, "*")
        var.mapping.spatiotemporal.response_features = args.features
        #  Load data
        result_dir = os.sep.join([self.result_dir(), var.execution.get("dataset", "train")])
        result_dir = self.result_dir()
        data = Data(var)
        dataset = data.get("dataset", "train")
        spatmp = dataset.spatiotemporal
        spalabs = spatmp.original.spatial_labels
        tmplabs = spatmp.original.temporal_labels
        spalab_field = spatmp.misc.spatial_label_field
        os.makedirs(result_dir, exist_ok=True)
        #
        if args.criterion == "missing-values":
            x = spatmp.filter_axis(spatmp.original.gtmask, -1, spatmp.misc.response_indices)
        else:
            raise NotImplementedError(args.criterion)
        T, V, F = x.shape
        index = util.sliding_window_indices(T, args.window_len, args.window_stride)
        x = np.take(x, index, 0)
        tmplab_wins = np.take(tmplabs, index, 0)
        intervals = tmplab_wins[:,[0,-1]]
        if args.criterion == "missing-values":
            n = args.window_len * F
            x = np.sum(x, (1, -1))
            if 0:
                x = n - x
            #
            top_index = np.argsort(np.sum(x, -1), 0)[-3:]
            top_x = x[top_index,:]
            print(top_x)
            top_intervals = intervals[top_index,:]
            for i in range(len(top_intervals)):
                __ = ["interval"] + list(top_intervals[i])
                print("%d/%d|%.2f%% :"%(np.mean(top_x[i]), n, 100*np.mean(top_x[i])/n), str(__).replace("'", "\""))
                _tmplabs = tmplab_wins[top_index[i]]
                _T = len(_tmplabs)
                train = _tmplabs[:int(0.7*_T)]
                valid = _tmplabs[int(0.7*_T):int(0.9*_T)]
                test = _tmplabs[int(0.9*_T):]
                print("train:", str(["interval"]+[train[0],train[-1]]).replace("'", "\""))
                print("valid:", str(["interval"]+[valid[0],valid[-1]]).replace("'", "\""))
                print("test:", str(["interval"]+[test[0],test[-1]]).replace("'", "\""))
            #
            print(x.shape)
            plt = Plotting()
            fig, ax = plt.subplots(size=(8, 4))
            index = np.linspace(0, x.shape[0]-1, 5, dtype=int)
            xt_labs = intervals[index,0]
            plt.plot_heatmap(
                np.swapaxes(x, 0, 1), vmin=0, vmax=n, xtick_locs=index, xtick_labels=xt_labs, 
                ax=ax, cmap="inferno", cbar_label="No. Groundtruth"
            )
#            ax.annotate("*", xy=(top_index[-1], 1.0), xytext=(top_index[-1], -.1), ha="center", va="center", arrowprops={"arrowstyle": "->"})
            ax.annotate("", xy=(top_index[-1], -.5), xytext=(top_index[-1], -1.5), ha="center", va="center", arrowprops={"arrowstyle": "->"})
            plt.title("Features=%s" % (args.features), ax=ax)
        #
        path = os.sep.join([result_dir, "TIL_[%s]_[%s].png" % (args.dataset, ",".join(args.features))])
        plt.save_figure(path, fig=fig)


def get_ana(module, ana_id=None):
    if ana_id is None:
        return get_user_ana(module)
    return getattr(module, "Analysis__%s" % (str(ana_id).replace(".", "_")))()


def get_user_ana(module):
    ana_id = input("Analysis: ")
    print()
    return get_ana(module, ana_id)


def values_and_partitions(mode=0):
    if mode == 0:
        values = ["n_parameters", "epoch_runtime", "UPR", "OPR", "MR", "RMSE"]
        partitions = [None, None, "test", "test", "test", "*"]
    elif mode == 1:
        values = ["n_parameters", "epoch_runtime", "MAE", "RMSE", "MAPE", "RRSE", "CORR"]
        partitions = [None, None, "test", "test", "test", "test", "test"]
    elif mode == 2:
        values = ["MAE", "RMSE", "MAPE", "RRSE", "CORR"]
        partitions = ["test", "test", "test", "test", "test"]
    elif mode == 3:
        values = ["MAE", "RMSE", "MAPE"]
        partitions = ["*", "*", "*"]
    else:
        raise ValueError(mode)
    return [values, partitions]


def add_improvement_row(df, model, ignore_cols=["Model"], debug=0):
    imps = []
    for i, col in enumerate(df.columns):
        if col in ignore_cols: continue
        target = df.loc[df["Model"]==model,col].values[0]
        smallest = df.nsmallest(2, col)
        model_a, model_b = smallest["Model"]
        metric_a, metric_b = smallest[col]
        if model_a == model: # target model is best - use as model_b
            model_a, metric_a = model_b, metric_b
            model_b, metric_b = model, target
        elif model_b == model: # target model is second best - keep as model_b
            pass
        else: # target model not in top-two - use as model_b
            model_b, metric_b = model, target
        if debug:
            print(smallest)
            print("+", model_a, metric_a, "+-+", model_b, metric_b, "+")
        imps.append((metric_a / metric_b) * 100 - 100)
    df.loc[len(df)] = ["Improvement"] + imps
    return df


def df_to_tex(df, round_to=3):
    def reduce_mean_stds(df, round_to=3):
        if isinstance(df.columns, pd.MultiIndex):
            # temporarily remove index to avoid "unhashable type" error
            index = df.index
            df.reset_index(drop=True, inplace=True)
            # 
            col_label_dict = {col0: [] for col0, col1 in df.columns if not col0 in ["__N__", "__mid__"]}
            for col0, col1 in df.columns:
                if col0 in ["__N__", "__mid__"] or col1 in ["n"]:
                    df.drop((col0,col1), axis=1, inplace=True)
                    continue
                col_label_dict[col0].append(col1)
            n_reduced_cols = sum(int(_ == ["mean", "std"]) for _ in col_label_dict.values())
            _df = pd.DataFrame(
                np.full((df.shape[0], df.shape[1]-n_reduced_cols), np.nan), 
                index=df.index, columns=col_label_dict.keys()
            )
            int_cols = set()
            for row in df.index:
                for col0 in col_label_dict.keys():
                    if col_label_dict[col0] == ["mean", "std"]:
                        mean, std = df.loc[row,(col0,"mean")], df.loc[row,(col0,"std")]
                        if not (np.isnan(mean) or np.isnan(std)):
                            if util.Types.is_int(mean) and util.Types.is_int(std):
                                result = "%d $\pm$ %d" % (mean, std)
                            else:
                                result = "%.*f $\pm$ %.*f" % (round_to, mean, round_to, std)
                        elif np.isnan(std):
                            if util.Types.is_int(mean):
                                result = "%d" % (mean)
                            else:
                                result = "%.*f" % (round_to, mean)
                        else:
                            result = "N/A"
                        _df.loc[row,col0] = result
                    else:
                        for col1 in col_label_dict[col0]:
                            value = df.loc[row,(col0,col1)]
                            _df.at[row,col0] = value
                            if util.Types.is_int(value):
                                int_cols.add(col0)
            for col in int_cols:
                _df.loc[:,col] = _df.loc[:,col].round().astype(int)
            _df.index = index
            df = _df
        return df
    df = reduce_mean_stds(df, round_to)
    tex = df.to_latex(na_rep="N/A")
    return tex


def clean_tex(tex, subs=[]):
    tex = re.sub("\\\\\\$", "$", tex) # sub "\$" with "$"
    tex = re.sub("\\\\textbackslash ", "\\\\", tex) # sub "\textbackslash " with "\"
    tex = re.sub(" +", " ", tex) # sub excess spaces
    for sub in subs:
        tex = re.sub(sub[0], sub[1], tex)
    return tex


def values_over_parameter_analysis(exp, var, result_dir):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # Gather all data from cache
    eval_dir, chkpt_dir = exp.jobs[0].work.get(["evaluation_dir", "checkpoint_dir"])
    cache = gather.get_cache(eval_dir, chkpt_dir)
    # Handle broadcasting for all variables
    #   handle values : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.values, str):
        if isinstance(var.value_partitions, list):
            var.values = [var.values for _ in var.value_partitions]
        else:
            var.values = [var.values]
    elif not isinstance(var.values, list):
        raise ValueError("Input values must be str or list of str, received %s" % (var.values))
    #   handle value_partitions : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.value_partitions, str): # single partition - broadcast to all values
        var.value_partitions = [var.value_partitions for _ in var.values]
    elif isinstance(var.value_partitions, list):
        if len(var.value_partitions) != len(var.values):
            var.value_partitions = [var.value_partitions for _ in var.values]
    elif not isinstance(var.value_partitions, list):
        raise ValueError("Input value_partitions must be str or list of str, received %s" % (var.value_partitions))
    for i, partition in enumerate(var.value_partitions):
        if partition == "*":
            var.value_partitions[i] = ["train", "valid", "test"]
        elif not isinstance(partition, list):
            var.value_partitions[i] = [var.value_partitions[i]]
    #   handle plot_types : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.plot_types, str):
        var.plot_types = [var.plot_types for _ in var.values]
    elif not isinstance(var.plot_types, list):
        raise ValueError("Input plot_types must be str or list of str, received %s" % (var.plot_types))
    if var.model is None:
        var.model = input("Please provide a model: ")
    if var.parameter is None:
        var.parameter = input("Please provide a parameter: ")
    if var.debug:
        print(var)
    # Collect values from cache
    if var.parameter_values is None:
        param_values = []
        for job in exp.jobs:
            param_value = job.work.get(var.parameter, var.parameter_partition)
            if param_value not in param_values:
                param_values.append(param_value)
    else:
        param_values = var.parameter_values
    vop_dicts = collect_values_over_parameter(
        var.values,
        var.value_partitions,
        var.parameter,
        param_values,
        var.parameter_partition, 
        cache,
        var,
    )
    if var.debug > 1:
        print(Container().from_dict(vop_dicts))
    # Join all data dicts into one DataFrame
    dfs = []
    for value_name, vop_dict in vop_dicts.items():
        df = pd.DataFrame(vop_dict).drop(var.parameter, axis=1)
        dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=var.values)
    # Count number of results
    _ = df.to_numpy().reshape(-1)
    n_results = 0
    for i in range(len(_)):
        if _[i] is None:
            pass
        elif isinstance(_[i], (tuple, list)):
            n_results += len(_[i])
        elif isinstance(_[i], str) and _[i] != "":
            n_results += 1
        elif not np.isnan(_[i]):
            n_results += 1
    print(util.make_msg_block(45*"*" + " No. Results (%d) " % (n_results) + 45*"*", "*"))
    #   Remove NaN instances (from None partition) from the columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_frame(
            df.columns.to_frame().fillna("")
        )
    else:
        df.columns = df.index.fillna("")
#    df.columns = pd.MultiIndex.from_frame(df.columns.to_frame().applymap(str.capitalize))
    #   Drop secondary columns if it simply repeats one element
    cols = np.array([list(col) for col in df.columns])
    if not len(np.unique(np.delete(cols[:,1], cols[:,1]==""))) > 1:
        df.columns = cols[:,0]
    # Check for tuple/list entries and replace with mean+std
    if any(util.Types.is_collection(_) for _ in df.to_numpy().reshape(-1)):
        stat_names = ["mean", "std", "n"]
        new_cols = []
        col_types = []
        stat_cols = []
        for col, values in df.items():
            if any(util.Types.is_collection_of_numeric(_) for _ in values): # col of multi-numeric -> stats
                new_cols += [(col, _) for _ in stat_names]
                col_types += [float, float, int]
                stat_cols.append(col)
            else:
                new_cols.append((col,""))
                if util.Types.is_collection_of_numeric(values): # col of numeric
                    if util.Types.is_int(values[0]):
                        col_types.append(int)
                    elif util.Types.is_float(values[0]):
                        col_types.append(float)
                    else:
                        raise TypeError(values)
                else: # col of non-numeric
                    col_types.append(object)
        mindex = pd.MultiIndex.from_tuples(new_cols, names=["Values", "Stat"])
        _df = pd.DataFrame(np.full((df.shape[0], len(mindex)), np.nan), index=df.index, columns=mindex)
        for col, values in df.items():
            if col in stat_cols: # maybe multiple values in this cell - place into mean/std mindex
                for idx, value in zip(df.index, values):
                    if util.Types.is_collection(value): # multiple values
                        if util.Types.is_collection_of_nan(value):
                            stats = {_: np.nan for _ in stat_names}
                            stats["n"] = 0
                        else:
                            stats = util.get_stats(value)
                        for _ in stat_names:
                            _df.loc[idx,(col,_)] = stats[_]
                        if util.Types.is_int(value[0]):
                            int_cols.add(col)
                    else: # singular value
                        n = 1
                        _df.loc[idx,(col,"mean")] = value
                        if np.isnan(value):
                            n = 0
                        _df.loc[idx,(col,"n")] = n
            else: # singular value in this cell
                _df.loc[:,(col,"")] = values
        for new_col, col_type in zip(new_cols, col_types):
            if col_type == int:
                _df.loc[:,new_col] = _df.loc[:,new_col].round().astype(col_type)
            else:
                _df.loc[:,new_col] = _df.loc[:,new_col].astype(col_type)
        df = _df
    #   Add parameter values as index
    df.index = pd.Index(param_values, name=var.parameter)
    #   Apply misc changes
    df = df.round(var.round_to)
    print(df)
    path = os.sep.join(
        [
            result_dir,
            "VOP_V[%s]_P[%s]%s.csv" % (",".join(vop_dicts.keys()), var.parameter, var.fname_adder),
        ]
    )
    df.to_csv(path)
    if var.plot:
        # Plot the value(s) as a function of the parameter
        path = os.sep.join(
            [
                result_dir,
                "VOP_V[%s]_P[%s].png" % (
                    ",".join(vop_dicts.keys()), var.parameter, var.fname_adder
                ),
            ]
        )
        plot_values_over_parameter(vop_dicts, var.value_partitions, var, path)
    # Save the value(s) as a Latex table
    path = path.replace(".csv", ".tex")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    table = df_to_tex(df)
    table = clean_tex(
        table, 
        subs=[
            ["model", "Model"], 
            ["n\\\\_parameters", "Parameters"], 
            ["runtime", "Runtime"], 
            ["epoch\\\\_runtime", "Epoch Runtime"], 
            ["best\\\\_epoch", "Best Epoch"], 
            ["\.0+ ", " "], 
        ]
    )
    if var.debug:
        print(table)
    with open(path, "w") as f:
        f.write(table)


def collect_values_over_parameter(value_names, value_partitions, param_name, param_values, param_partition, cache, var):
    def pull_error(cache, value_name, partition, channel_name, model, model_id):
        value = np.nan
        try:
            errors = cache.get(value_name, partition, path=[channel_name, model, model_id])
            if var.debug > 2:
                print("get(", value_name, partition, [model, model_id], ")")
                print(errors)
                input()
            if var.response_feature is None:
                response_feature, _errors, _ = errors[0]
            else:
                _errors = errors.get(var.response_feature)
            if isinstance(_errors, dict):
                value = np.nanmean(list(_errors.values()))
            elif isinstance(_errors, (int, float)):
                value = _errors
            else:
                raise ValueError(
                    "Unknown error value found: %s" % (str(_errors))
                )
        except:
            pass
        return value
    model_id_map = {}
    vop_dicts = {}
    for value_name, _value_partitions in zip(value_names, value_partitions):
        vop_dicts[value_name] = util.to_dict(
            [param_name] + _value_partitions, [[] for _ in range(1+len(_value_partitions))]
        )
    n_results = 0
    for value_name, _value_partitions in zip(value_names, value_partitions):
        vop_dicts[value_name][param_name] = param_values
        for partition in _value_partitions:
            for param_value in param_values:
                model, model_id = var.model, np.nan
                try:
                    if param_name in ["model"]: # model is the parameter
                        model = param_value
                        if model in model_id_map:
                            model_id = model_id_map[model]
                        else:
                            model_id, model_con = gather.find_model_id(
                                cache, 
                                model, 
                                where=var.where, 
                                on_multi=var.on_multi, 
                                return_channel_con=True
                            )
                            model_id_map[model] = model_id
                        if var.debug > 1:
                            print(util.make_msg_block(param_value + " " + model_id))
    #                        print(model_params)
                            input()
                    else:
                        where = [{"name": param_name, "value": param_value, "partition": param_partition}]
                        for _ in var.where: 
                            where.append({"name": _[0], "comparator": _[1], "value": _[2]})
                        model_id, model_con = gather.find_model_id(
                            cache, 
                            model, 
                            where=where, 
                            on_multi=var.on_multi, 
                            return_channel_con=True
                        )
                except Exception as err:
                    if var.debug > 1:
                        print(err)
                        print(model, model_id, partition)
                        input()
                if value_name == "__mid__":
                    vop_dicts[value_name][partition].append(model_id)
                    continue
                elif value_name == "__N__":
                    value = 0
                    if isinstance(model_id, (tuple, list)):
                        value = len(model_id)
                    elif isinstance(model_id, str):
                        value = 1
                    vop_dicts[value_name][partition].append(value)
                    continue
                # Get the cache channel that this value is in
                channel_name, channel_con = None, None
                for channel_name, channel_con, _ in cache.get_name_value_partitions(sort=False):
                    if channel_con.has(value_name, partition): break
                if channel_con is None:
                    raise LookupError(
                        "Failed to find cache channel containing variable by (name, partition)=(%s, %s)" % (
                            value_name, partition
                        )
                    )
                # Get the instance of this value
                value = np.nan
                try:
                    if channel_name in ["errors"]:
                        if isinstance(model_id, (tuple, list)):
                            value = [pull_error(cache, value_name, partition, channel_name, model, _) for _ in model_id]
                        else:
                            value = pull_error(cache, value_name, partition, channel_name, model, model_id)
                    else:
                        try:
                            if isinstance(model_id, (tuple, list)):
                                value = [cache.get(value_name, partition, path=[channel_name, model, _]) for _ in model_id]
                            else:
                                value = cache.get(value_name, partition, path=[channel_name, model, model_id])
                        except ValueError as err: # See if it exists in the None partition
                            if isinstance(model_id, (tuple, list)):
                                value = [cache.get(value_name, path=[channel_name, model, _]) for _ in model_id]
                            else:
                                value = cache.get(value_name, path=[channel_name, model, model_id])
                except ValueError as err:
                    if var.debug > 1:
                        print(err)
                        input()
                vop_dicts[value_name][partition].append(value)
    return vop_dicts


def plot_values_over_parameter(vop_dicts, partitions, var, path):
    plt = Plotting()
    normalize = var.normalize if "normalize" in var else True
    n_bars = 0
    for _partitions, plot_type in zip(partitions, var.plot_types):
        n_bars += len(_partitions) if plot_type == "bar" else 0
    i, bar_idx = 0, 0
    colors = plt.default_colors
    for (value_name, vop_dict), _partitions, plot_type in zip(vop_dicts.items(), partitions, var.plot_types):
        df = pd.DataFrame(vop_dict)
        if var.debug:
            print(util.make_msg_block(value_name))
            print(df)
        df.fillna(np.nan, inplace=True)
        for partition in _partitions:
            linestyle = "-"
            if partition in ["test"]:
                linestyle = "--"
            color = colors[i % len(colors)]
            label = value_name
            if not partition is None:
                label = "%s %s" % (partition, value_name)
            values = df[partition].to_numpy()
            if normalize and not all(np.logical_and(values >= 0, values <= 1)):
                values = Transform.minmax_transform(values, np.min(values), np.max(values), a=1/10, b=1)
            param_values = df[var.parameter]
            if not isinstance(param_values[0], float) and not isinstance(param_values[0], int):
                param_values = [str(param_value) for param_value in param_values]
            kwargs = {} if not "plot_kwargs" in var else var.plot_kwargs
            if plot_type == "line":
                kwargs = util.merge_dicts(kwargs, {"color": color, "label": label, "linestyle": linestyle})
                plt.plot_line(param_values, values, **kwargs)
            elif plot_type == "bar":
                if n_bars > 0:
                    bar_sep = 25 / len(param_values)
                    bar_extent = [-3/8 * bar_sep, 3/8 * bar_sep]
                    bar_width = plt.defaults.bars.width
                    if n_bars == 1:
                        bar_offsets = np.array([0])
                    else:
                        bar_width = plt.defaults.bars.width / n_bars
                        bar_width = (bar_extent[1] - bar_extent[0]) / (n_bars * 3/4)
                        bar_offsets = np.linspace(
                            bar_extent[0] + bar_width / 2,
                            bar_extent[1] - bar_width / 2,
                            n_bars,
                        )
                locs = param_values
                if isinstance(param_values[0], str):
                    locs = np.arange(len(param_values), dtype=float) * bar_sep
                locs += bar_offsets[bar_idx]
                kwargs = util.merge_dicts(
                    kwargs, 
                    {"width": bar_width, "color": color, "label": label, "linestyle": linestyle})
                plt.plot_bar(locs, values, **kwargs)
                bar_idx += 1
            elif plot_type == "scatter":
                kwargs = util.merge_dicts(kwargs, {"color": color, "label": label, "linestyle": linestyle})
                plt.plot_satter(param_values, values, **kwargs)
            else:
                raise NotImplementedError("Unknown plot_type \"%s\"" % (var.plot_type))
            if np.nan in values and value_name in ["MAE", "MSE", "MAPE", "RMSE", "NRMSE", "OPR", "UPR", "RRSE"]:
                _values = np.copy(values)
                _values[_values == 0] = int(sys.float_info.max)
                plt.plot_axis(min(values), color=color, linestyle=":")
            i += 1
    plt.style("grid")
    locs = param_values
    if isinstance(param_values[0], str):
        locs = np.arange(len(param_values), dtype=float) * bar_sep
        plt.lim([locs[0]-1, locs[-1]+1])
    fontsize = 11 * (25 / len(param_values))
    plt.xticks(locs, param_values, rotation=90, fontsize=fontsize)
    if normalize:
        plt.lim(None, [0, 1.025], ymargin=["1%", "2.5%"])
        plt.labels(var.parameter, "Normalized Value")
    else:
        plt.labels(var.parameter, None)
    plt.legend(prop={"size": 6.5}, ncol=len(partitions))
    plt.save_figure(path)
    plt.close()


def get_graph_data(dataset, partition, var):
    if dataset.graph.is_empty():
        raise ValueError("Cannot plot a graph without dataset.graph")
    graph = dataset.graph
    G = graph.original.get("nx_graph", partition)
    node_positions, node_sizes = None, None
    node_labels = graph.original.get("node_labels", partition)
    node_selection = ["literal"] + node_labels.tolist()
    n_nodes = len(node_labels)
    if not dataset.spatial.is_empty(): # Get node positions and sizes from spatial data
        spatial = dataset.spatial
        spatial_labels = spatial.original.spatial_labels
        spatial_indices = spatial.indices_from_selection(spatial_labels, node_selection)
        position_indices = np.array(util.get_dict_values(spatial.misc.feature_index_map, var.node_position_features))
        node_positions = spatial.original.filter_axis(
            spatial.original.features, 
            [-2, -1], 
            [spatial_indices, position_indices]
        )
        if var.node_size_feature in spatial.misc.feature_index_map:
            size_idx = spatial.misc.feature_index_map[var.node_size_feature]
            node_sizes = spatial.original.filter_axis(
                spatial.original.features, 
                [-2, -1], 
                [spatial_indices, size_idx]
            )
            node_sizes = node_sizes / np.max(node_sizes)
            node_sizes = node_sizes[spatial_indices]
    # Get node sizes from spatiotemporal data
    if not var.node_size_feature is None and not dataset.spatiotemporal.is_empty():
        spatiotemporal = dataset.spatiotemporal
        spatial_labels = spatiotemporal.original.spatial_labels
        spatial_indices = spatial.indices_from_selection(spatial_labels, node_selection)
        size_idx = spatiotemporal.misc.feature_index_map[var.node_size_feature]
        node_sizes = np.mean(
            spatiotemporal.filter_axis(
                spatiotemporal.statistics.means, 
                [-2, -1], 
                [spatial_indices, size_idx]
            ), 
            axis=0
        )
    else:
        node_sizes = np.ones((len(node_labels),))
    node_sizes = node_sizes**(1/var.node_size_root)
    node_sizes = node_sizes * var.node_size_mult
    return G, node_positions, node_sizes


def visualize_graph(dataset, partition, var, path):
    from Plotting import Plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=var.figsize)
    plot = Plotting()
    if "shapes" in var and not var.shapes is None:
        plot.plot_shapes(var.shapes, ax=ax, filled=True, color="k", alpha=1/8, linewidth=1)
    if partition is None or isinstance(partition, str):
        partition = [partition]
    elif not isinstance(partition, list):
        raise ValueError("Input \"partition\" may be None, str, or list of None/str. Received %s" % (type(partition)))
    for _partition in partition:
        alpha = 2/8 if _partition is None and len(partition) > 1 else 1
        G, node_positions, node_sizes = get_graph_data(dataset, _partition, var)
        plot.plot_networkx_graph(
            G, 
            node_positions, 
            node_sizes, 
            node_alpha=alpha, 
            edge_alpha=alpha, 
            plot_edges=var.plot_args.edges, 
            plot_labels=var.plot_args.labels, 
            ax=ax
        )
    xlim, ylim = plot.lim()
    plot.save_figure(path, dpi=200)
    plot.close()
    plt.close()


def feature_fullname(feature_label):
    return {
        "date": "Date", 
        "FLOW_OUTcms": "Streamflow", 
        "PRECIPmm": "Precipitation", 
        "SWmm": "Soil Water", 
        "tmax": "Maximum Temperature", 
        "tmin": "Minimum Temperature", 
        #
        "speedmph": "Speed", 
        #
        "power_kWh": "Power", 
        #
        "power_MW": "Power", 
        #
        "ApparentPowerVA": "Apparent Power", 
        "PowerFactor": "Power Factor", 
        "TruePowerW": "True Power", 
    }.get(feature_label, feature_label)


def feature_SIunit(feature_label):
    return {
        "date": "index", 
        "FLOW_OUTcms": "$m^3/s$", 
        "PRECIPmm": "mm", 
        "SWmm": "mm", 
        "tmax": "$^{\circ}C$", 
        "tmin": "$^{\circ}C$", 
        #
        "speedmph": "mph", 
        #
        "power_kWh": "kWH", 
        #
        "power_MW": "MW", 
        #
        "ApparentPowerVA": "volt-ampere", 
        "PowerFactor": "%", 
        "TruePowerW": "watt", 
    }.get(feature_label, None)


def feature_ylabel(feature_label):
    fullname = feature_fullname(feature_label)
    SIunit = feature_SIunit(feature_label)
    ylabel = fullname
    if not SIunit is None:
        ylabel = "%s (%s)" % (fullname, SIunit)
    return ylabel
