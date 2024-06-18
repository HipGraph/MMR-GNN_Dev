import scipy
import sklearn
import numpy as np
import multiprocessing as mp
from sklearn import cluster as sk_cluster


class Clustering:

    def cluster(x, alg="KMeans", n_clusters=3, return_obj=False, **kwargs):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables
        alg : str
        kwargs : dict

        Returns
        -------
        cluster_index : ndarray with shape=(N,)

        """
        debug = kwargs.get("debug", 0)
        if alg == "KMeans":
            clustering = sk_cluster.KMeans(n_clusters, random_state=0, **kwargs).fit(x)
            cluster_index = clustering.labels_
        elif alg == "Agglomerative":
            clustering = sk_cluster.AgglomerativeClustering(n_clusters, compute_distances=True, **kwargs).fit(x)
            if debug:
                print(clustering.children_)
                print(clustering.distances_)
            cluster_index = clustering.labels_
        elif alg == "DBSCAN":
            clustering = sk_cluster.DBSCAN(min_samples=5).fit(x)
            cluster_index = clustering.labels_
        elif alg == "Random":
            rng = np.random.default_rng(0)
            clustering = rng
            cluster_index = rng.integers(0, n_clusters, size=len(x))
        else:
            raise NotImplementedError(alg)
        if return_obj:
            return clustering, cluster_index
        return cluster_index
        

class Probability:

    def get_lims(x, lims=None):
        if lims is None:
            lims = np.stack((np.min(x, 0), np.max(x, 0)), -1)
        elif not isinstance(lims, np.ndarray):
            if isinstance(lims[0], str) and "%" in lims[0]: # percentile
                lims = [np.percentile(x, float(lims[0][:-1])), lims[1]]
            if isinstance(lims[1], str) and "%" in lims[1]: # percentile
                lims = [lims[0], np.percentile(x, float(lims[1][:-1]))]
        return lims

    def get_pv(pmf, bins, v, debug=0):
        if debug:
            print(pmf)
            print(bins)
        if v == bins[-1]:
            if debug:
                print(-1, v, bins[-2:], pmf[-1])
            return -1, pmf[-1]
        for i in range(len(bins)-1):
            if debug:
                print(i, v, bins[i:i+2], pmf[i])
            idx = i
            if v >= bins[i] and v < bins[i+1]:
                idx = i
                break
        return idx, pmf[idx]

    def normal_pdf(x, u=0, s=1):
        if hasattr(x, "__iter__"):
            return [Probability.normal_pdf(_, u, s) for _ in x]
        return (1.0 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.power((x - u) / s, 2))

    def mi(px, py, pxy):
        mi = pxy * np.log2(pxy / (px*py + np.finfo(float).eps) + np.finfo(float).eps)
        if mi < 0:
            mi = np.nan
#            print("mi=", mi, "px=", px, "py=", py, "pxy=", pxy, "...=", pxy / (px*py), "log(...)=", np.log2(pxy / (px*py)))
#            input()
        return mi

    def mp_lag_mutual_information(x, y, lag, bins=10, xlims=None, ylims=None, n_proc=64, **kwargs):
        debug = kwargs.get("debug", 0)
        Cx, Nx, Dx = x.shape
        Cy, Ny, Dy = y.shape
        manager = mp.Manager()
        ret_dict = manager.dict()
        n_passes = Cx // n_proc + 1 if Cx % n_proc else Cx // n_proc
        for i in range(n_passes):
            jobs = []
            for j in range(n_proc):
                pid = i * n_proc + j
                if pid >= Cx:
                    break
                p = mp.Process(
                    target=Probability._mp_lag_mutual_information, 
                    args=(x[pid,:,:], y[pid,:,:], lag, bins, xlims, ylims, pid, ret_dict, kwargs)
                )
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        if debug:
            for i in range(Cx):
                print(i, ret_dict[i].shape)
        lmis = np.stack([ret_dict[i] for i in range(Cx)])
        return lmis

    def _mp_lag_mutual_information(x, y, lag, bins=10, xlims=None, ylims=None, pid=0, ret_dict={}, kwargs={}):
        ret_dict[pid] = Probability.lag_mutual_information(x, y, lag, bins, xlims, ylims, **kwargs)

    def lag_mutual_information(x, y, lag, bins=10, xlims=None, ylims=None, **kwargs):
        debug = kwargs.get("debug", 0)
        reduction = kwargs.get("reduction", "sum")
        x_pmf = kwargs.get("x_pmf", None)
        y_pmf = kwargs.get("y_pmf", None)
        xy_pmf = kwargs.get("joint_pmfs", None)
        #
        Nx, Dx = x.shape
        Ny, Dy = y.shape
        if isinstance(lag, int):
            lag = np.array([lag])
        elif isinstance(lag, (range, tuple, list)):
            lag = np.array(lag)
        elif not isinstance(lag, np.ndarray):
            raise TypeError(type(lag), lag)
        xlims = Probability.get_lims(x, xlims)
        ylims = Probability.get_lims(y, ylims)
        if debug:
            print("x=", x.shape, "y=", y.shape)
            print("xlims=", xlims.shape)
            print(xlims)
            print("ylims=", ylims.shape)
            print(ylims)
            input()
        #
        if x_pmf is None:
            x_pmf, xbins = Probability.compute_pmfs(x, bins, xlims, return_bins=True)
        if y_pmf is None:
            y_pmf, ybins = Probability.compute_pmfs(y, bins, ylims, return_bins=True)
        if debug:
            print(x_pmf.shape)
            print(x_pmf)
            print(xbins.shape)
            print(xbins)
            print(y_pmf.shape)
            print(y_pmf)
            print(ybins.shape)
            print(ybins)
            input()
        if xy_pmf is None:
            xy_pmf = Probability.compute_joint_pmfs(x, y, bins, xlims, ylims)
        if debug:
            print(xy_pmf.shape)
            print(xy_pmf)
            input()
        px, px_idx = np.zeros((Nx, Dx)), np.zeros((Nx, Dx), dtype=int)
        for i in range(Nx):
            for j in range(Dx):
                px_idx[i,j], px[i,j] = Probability.get_pv(x_pmf[j,:], xbins[j,:], x[i,j], debug=0)
        py, py_idx = np.zeros((Ny, Dy)), np.zeros((Ny, Dy), dtype=int)
        for i in range(Ny):
            for j in range(Dy):
                py_idx[i,j], py[i,j] = Probability.get_pv(y_pmf[j,:], ybins[j,:], y[i,j], debug=0)
        if debug:
            print(px)
            print(px_idx)
            print(py)
            print(py_idx)
        if debug:
            input()
        #
        red_fn = {
            "mean": np.mean, 
            "sum": np.sum, 
            None: lambda a, axis: a
        }
        T = len(x)
        lmis = []
        for tau in lag:
            if reduction in ["sum"]: # remove sum bias by making # of mi samples constant across all lags
                T = len(x) - max(lag) + tau
            lmi = []
            for i in range(Dx):
                _ = []
                for j in range(Dy):
                    __ = []
                    for t in range(T-tau):
                        _px_idx, _px = px_idx[t,i], px[t,i]
                        _py_idx, _py = py_idx[t+tau,j], py[t+tau,j]
                        pxy = xy_pmf[i,j,_px_idx,_py_idx]
                        if debug:
                            print(_px_idx, _py_idx)
                            print(_px, _py, pxy)
                        mi = Probability.mi(_px, _py, pxy)
                        if 0 and np.isnan(mi):
                            print("px_idx=", _px_idx, "py_idx=", _py_idx)
                            print(xlims)
                            print(ylims)
                            print(xbins)
                            print(ybins)
                            print(x_pmf)
                            print(y_pmf)
                            print(xy_pmf[i,j])
                            input()
                        __.append(mi)
                    _.append(__)
                lmi.append(_)
            lmi = red_fn[reduction](lmi, -1)
            lmis.append(lmi)
        lmis = np.array(lmis)
        if lmis.shape[0] == 1:
            lmis = lmis[0,:]
        return lmis

    def mutual_information(x, y, bins=10, lims=None, **kwargs):
        #
        if lims is None:
            lims = np.stack((min(np.min(x), np.min(y)), max(np.max(x), np.max(y))), -1)
        debug = kwargs.get("debug", 0)
        x_pmf = kwargs.get("x_pmf", None)
        y_pmf = kwargs.get("y_pmf", None)
        xy_pmf = kwargs.get("joint_pmfs", None)
        if x_pmf is None:
            x_pmf, xbins = Probability.compute_pmfs(x, bins, lims, return_bins=True)
        if y_pmf is None:
            y_pmf, ybins = Probability.compute_pmfs(y, bins, lims, return_bins=True)
        if debug:
            print(x_pmf.shape, "=", x_pmf)
            print(y_pmf.shape, "=", y_pmf)
            print(xbin_edges.shape, "=", xbin_edges)
        if xy_pmf is None:
            xy_pmf = Probability.compute_joint_pmfs(x, y, bins, lims)
        if debug:
            print(xy_pmf.shape, "=", xy_pmf)
            input()
            print(x)
            print(y)
        i, j = kwargs.get("i", None), kwargs.get("j", None)
        if not (i is None or j is None):
            _, px = Probability.get_pv(x_pmf[0,:], xbins, x[i,0])
            _, py = Probability.get_pv(y_pmf[0,:], ybins, y[j,0])
            return Probability.mi(px, py, xy_pmf[0,i,j])
        px, px_idx = [], []
        for i, _x in enumerate(x):
            idx, _px = Probability.get_pv(x_pmf[0], xbins, _x[0])
            px.append(_px)
            px_idx.append(idx)
        py, py_idx = [], []
        for i, _y in enumerate(y):
            idx, _py = Probability.get_pv(y_pmf[0], ybins, _y[0])
            py.append(_py)
            py_idx.append(idx)
        if debug:
            input()
        #
        mi = 0
        for i, _x in enumerate(x):
            for j, _y in enumerate(y):
                _px_idx, _px = px_idx[i], px[i]
                _py_idx, _py = px_idx[j], py[j]
                pxy = xy_pmf[0,_px_idx,_py_idx]
                if debug:
                    print(_px_idx, _py_idx)
                    print(_px, _py, pxy)
                mi += Probability.mi(_px, _py, pxy)
        return mi / (len(x) * len(y))

    def compute_pmfs(x, bins=10, lims=None, **kwargs):
        return_bins = kwargs.get("return_bins", False)
        #
        if return_bins:
            hists, bins = Probability.compute_histograms(x, bins, lims, **kwargs)
        else:
            hists = Probability.compute_histograms(x, bins, lims, **kwargs)
        pmfs = hists / np.sum(hists, -1)[:,None]
        if return_bins:
            return pmfs, bins
        return pmfs

    def compute_joint_pmfs(x, y, bins=10, xlims=None, ylims=None, **kwargs):
        return_bins = kwargs.get("return_bins", False)
        #
        if return_bins:
            hists, xbins, ybins = Probability.compute_joint_histograms(x, y, bins, xlims, ylims, **kwargs)
        else:
            hists = Probability.compute_joint_histograms(x, y, bins, xlims, ylims, **kwargs)
        pmfs = hists / np.sum(hists, (-2, -1))[...,None,None]
        if return_bins:
            return pmfs, xbins, ybins
        return pmfs

    def compute_joint_histograms(x, y, bins=10, xlims=None, ylims=None, **kwargs):
        return_bins = kwargs.get("return_bins", False)
        debug = kwargs.get("debug", 0)
        Nx, Dx = x.shape
        Ny, Dy = y.shape
        n_bins = bins
        if not isinstance(bins, int): # bins defines bin edges
            n_bins = len(bins) - 1
        xlims = Probability.get_lims(x, xlims)
        ylims = Probability.get_lims(y, ylims)
        if debug:
            print("Data.__init__", "bins =", bins, "xlims =", xlims, "ylims =", ylims)
        hists = np.zeros((Dx, Dy, n_bins, n_bins))
        xbins, ybins = np.zeros((Dx, n_bins+1)), np.zeros((Dy, n_bins+1))
        for i in range(Dx):
            for j in range(Dy):
                hists[i,j,:,:], xbins[i,:], ybins[j,:], binnumber = scipy.stats.binned_statistic_2d(
                    x[:,i], y[:,j], y[:,j], "count", bins, np.stack((xlims[i,:], ylims[j,:]))
                )
        if return_bins:
            return hists, xbins, ybins
        return hists

    def compute_histograms(x, bins=10, lims=None, **kwargs):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables
        bins : int or ndarray with shape(B,)
        lims : None or ndarray with shape=(2,) or shape=(D, 2)

        Returns
        -------
        histograms : ndarray with shape=(D, B) for D variables

        """
        return_bins = kwargs.get("return_bins", False)
        debug = kwargs.get("debug", 0)
        N, D = x.shape
        n_bins = bins
        if not isinstance(bins, int): # bins defines bin edges
            n_bins = len(bins) - 1
        lims = Probability.get_lims(x, lims)
        if debug:
            print("Data.__init__", "bins =", bins, "lims =", lims)
        if not isinstance(lims, np.ndarray):
            lims = np.array(lims)
        if lims.ndim == 1:
            lims = np.tile(lims, (x.shape[-1], 1))
        elif not lims.ndim == 2 or not lims.shape == (D, 2):
            raise ValueError(lims.shape)
        histograms = np.zeros((D, n_bins))
        bin_edges = np.zeros((D, n_bins+1))
        for i in range(D):
            histograms[i,:], bin_edges[i,:], binnumber = scipy.stats.binned_statistic(
                x[:,i], x[:,i], "count", bins, lims[i,:]
            )
        if return_bins:
            return histograms, bin_edges
        return histograms

    def compute_correlations(x, **kwargs):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables

        Returns
        -------
        corrs : ndarray with shape=(D, B) for D variables

        """
        debug = kwargs.get("debug", 0)
        N, D = x.shape
        corrs = np.corrcoef(x, rowvar=False, **kwargs)
        return corrs

    def transform(x, rep="histogram", **kwargs):
        if rep == "histogram":
            bins = kwargs.get("bins", 12)
            lims = kwargs.get("lims", None)
            x = Probability.compute_histograms(x, bins, lims, **kwargs)
        elif rep == "correlation":
            x = Probability.compute_correlations(x, **kwargs)
        else:
            raise NotImplementedError(rep)
        return x


class Decomposition:

    def reduce(x, dim=2, alg="TSNE", **kwargs):
        debug = kwargs.get("debug", 0)
        if alg == "PCA":
            x = sklearn.decomposition.PCA(dim, random_state=0, **kwargs).fit_transform(x)
        elif alg == "TSNE":
            x = sklearn.manifold.TSNE(
                dim, perplexity=min(30.0, x.shape[0]-1), random_state=0, **kwargs
            ).fit_transform(x)
        else:
            raise NotImplementedError(alg)
        return x
