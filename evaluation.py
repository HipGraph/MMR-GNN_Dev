import numpy as np

import util
from data import transform
from container import Container


##############################
### Classification Metrics ###
##############################
# Source : https://en.wikipedia.org/wiki/Confusion_matrix


def div(numer, denom, on_div_0=1):
    numer = numer.astype(float)
    denom = denom.astype(float)
    out = on_div_0 * np.ones_like(numer)
    where = denom != 0
    return np.divide(numer, denom, out=out, where=where)


def confusion_matrix(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    classes = kwargs.get("classes", None)
    debug = kwargs.get("debug", 0)
    # Compute
    if classes is None:
        classes = np.unique(y)
    n_classes = len(classes)
    shape = (n_classes, n_classes)
    if not axis is None:
        shape += tuple(np.take(y.shape, util.list_subtract(list(range(y.ndim)), list(axis))))
    if debug:
        print("axis =", axis)
        print("shape =", shape)
        input()
    cm = np.zeros(shape, dtype=int)
    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes):
            if mask is None:
                cm[i,j] += np.sum(np.logical_and(y == class_i, yhat == class_j).astype(int), axis)
            else:
                cm[i,j] += np.sum(
                    np.logical_and(np.logical_and(y == class_i, yhat == class_j), mask).astype(int), 
                    axis
                )
    if debug:
        print("cm =", cm.shape)
        print(cm)
        input()
    return cm


def negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return np.reshape(np.sum(cm[0,:], 0), cm.shape[2:])


def positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    if 0:
        cm = confusion_matrix(y, yhat, **kwargs, debug=1)
        intr = confusion_matrix(y, yhat, **kwargs)[:,1]
        res = np.sum(confusion_matrix(y, yhat, **kwargs)[:,1], 0)
        print(y.shape, yhat.shape)
        print("cm =", cm.shape)
        print(cm)
        print("intr =", intr.shape)
        print(intr)
        print("res =", res.shape)
        print(res)
        input()
    return np.reshape(np.sum(cm[1,:], 0), cm.shape[2:])


def predicted_negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return np.reshape(np.sum(cm[:,0], 0), cm.shape[2:])


def predicted_positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    if 0:
        cm = confusion_matrix(y, yhat, **kwargs, debug=1)
        intr = confusion_matrix(y, yhat, **kwargs)[:,1]
        res = np.sum(confusion_matrix(y, yhat, **kwargs)[:,1], 0)
        print(y.shape, yhat.shape)
        print("cm =", cm.shape)
        print(cm)
        print("intr =", intr.shape)
        print(intr)
        print("res =", res.shape)
        print(res)
        input()
    return np.reshape(np.sum(cm[:,1], 0), cm.shape[2:])


def false_negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,0]


def false_positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,1]


def true_negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,0]


def true_positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,1]


def false_negative_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,0] / np.sum(cm[1,:], 0)


def false_positive_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,1] / np.sum(cm[0,:], 0)


def true_negative_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,0] / np.sum(cm[0,:], 0)


def true_positive_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,1] / np.sum(cm[1,:], 0)


def prevalence(y, yhat, **kwargs):
    n = negative(y, yhat, **kwargs)
    p = positive(y, yhat, **kwargs)
    return div(p, (p + n))


def informedness(y, yhat, **kwargs): # informedness, bookmaker (BM)
    tnr = true_negative_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return tpr + tnr - 1


def accuracy(y, yhat, **kwargs): # accuracy
    n = negative(y, yhat, **kwargs)
    p = positive(y, yhat, **kwargs)
    tn = true_negative(y, yhat, **kwargs)
    tp = true_positive(y, yhat, **kwargs)
    res = div(tp + tn, p + n)
    if 0:
        print("n =", n.shape)
        print(n)
        print("p =", p.shape)
        print(p)
        print("tn =", tn.shape)
        print(tn)
        print("tp =", tp.shape)
        print(tp)
        print("res =", res.shape)
        print(res)
        input()
    return div(tp + tn, p + n)


def prevalence_threshold(y, yhat, **kwargs): # prevalence threshold (PT)
    fpr = false_positive_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return div((tpr * fpr)**(1/2) - fpr, tpr - fpr)


def balanced_accuracy(y, yhat, **kwargs): # balanced accuracy
    tnr = true_negative_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return (tpr + tnr) / 2


def false_omission_rate(y, yhat, **kwargs): # FOR
    pn = predicted_negative(y, yhat, **kwargs)
    fn = false_negative(y, yhat, **kwargs)
    return div(fn, pn)


def false_discovery_rate(y, yhat, **kwargs): # FDR
    pp = predicted_positive(y, yhat, **kwargs)
    fp = false_positive(y, yhat, **kwargs)
    return div(fp, pp)


def negative_predictive_value(y, yhat, **kwargs): # NPV
    pn = predicted_negative(y, yhat, **kwargs)
    tn = true_negative(y, yhat, **kwargs)
    return div(tn, pn)


def positive_predictive_value(y, yhat, **kwargs): # PPV
    pp = predicted_positive(y, yhat, **kwargs)
    tp = true_positive(y, yhat, **kwargs)
    return div(tp, pp)


def f1_score(y, yhat, **kwargs): # F1
    ppv = positive_predictive_value(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return div(2 * ppv * tpr, ppv + tpr)


def fowlkes_mallows_index(y, yhat, **kwargs): # FM
    ppv = positive_predictive_value(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return (ppv * tpr)**(1/2)


def positive_likelihood_ratio(y, yhat, **kwargs): # PLR
    fpr = false_positive_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return div(tpr, fpr)


def negative_likelihood_ratio(y, yhat, **kwargs): # NLR
    tnr = true_negative_rate(y, yhat, **kwargs)
    fnr = false_negative_rate(y, yhat, **kwargs)
    return div(fnr, tnr)


def markedness(y, yhat, **kwargs): # MK
    ppv = positive_predictive_value(y, yhat, **kwargs)
    npv = negative_predictive_value(y, yhat, **kwargs)
    return ppv + npv - 1


def diagnostic_odds_ratio(y, yhat, **kwargs): # DOR
    nlr = negative_likelihood_ratio(y, yhat, **kwargs)
    plr = positive_likelihood_ratio(y, yhat, **kwargs)
    return div(plr, nlr)


def matthews_correlation_coefficient(y, yhat, **kwargs): # MCC
    tpr = true_positive_rate(y, yhat, **kwargs)
    tnr = true_negative_rate(y, yhat, **kwargs)
    ppv = positive_predictive_value(y, yhat, **kwargs)
    npv = negative_predictive_value(y, yhat, **kwargs)
    fnr = false_negative_rate(y, yhat, **kwargs)
    fpr = false_positive_rate(y, yhat, **kwargs)
    _for = false_omission_rate(y, yhat, **kwargs)
    fdr = false_discovery_rate(y, yhat, **kwargs)
    return (tpr * tnr * ppv * npv)**(1/2) - (fnr * fpr * _for * fdr)**(1/2)


def threat_score(y, yhat, **kwargs): # TS
    fn = false_negative(y, yhat, **kwargs)
    fp = false_positive(y, yhat, **kwargs)
    tp = true_positive(y, yhat, **kwargs)
    if 0:
        print("fn =", fn.shape)
        print("fp =", fp.shape)
        print("tp =", tp.shape)
        input()
    return div(tp, tp + fn + fp)


# Aliases


def precision(y, yhat, **kwargs):
    return positive_predictive_value(y, yhat, **kwargs)


def recall(y, yhat, **kwargs):
    return true_positive_rate(y, yhat, **kwargs)


classification_metricfn_dict = {
#    "CM": confusion_matrix, 
    "N": negative, 
    "P": positive, 
    "PN": predicted_negative, 
    "PP": predicted_positive, 
    "FN": false_negative, # type 2 error, miss, underestimation
    "FP": false_positive, # type 1 error, false alarm, overestimation
    "TN": true_negative, # correct rejection
    "TP": true_positive, # hit
    "FNR": false_negative_rate, # miss rate
    "FPR": false_positive_rate, # probability of false alarm, fall-out
    "TNR": true_negative_rate, # specificity (SPC), selectivity
    "TPR": true_positive_rate, # recall, sensitivity
    "PREV": prevalence, 
    "BM": informedness, # Informedness, bookmaker informedness (BM)
    "ACC": accuracy, 
    "PT": prevalence_threshold,
    "BA": balanced_accuracy, 
    "FOR": false_omission_rate, 
    "FDR": false_discovery_rate, 
    "NPV": negative_predictive_value, 
    "PPV": positive_predictive_value, # precision
    "F1": f1_score, 
    "FM": fowlkes_mallows_index, 
    "PLR": positive_likelihood_ratio, 
    "NLR": negative_likelihood_ratio, 
    "MK": markedness, # deltaP
    "DOR": diagnostic_odds_ratio, 
    "MCC": matthews_correlation_coefficient, 
    "TS": threat_score, # critical success index (CSI), Jaccard index
    "PREC": precision, 
    "RECA": recall, 
}



##########################
### Regression Metrics ###
##########################


def mean_absolute_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    if not mask is None:
        y = np.ma.masked_array(y, mask)
        yhat = np.ma.masked_array(yhat, mask)
        if debug:
            print("mask =", mask.shape)
            print(mask)
            print("y =", y.shape)
            print(y)
            print("yhat =", yhat.shape)
            print(yhat)
            input()
    # Compute
    if debug:
        a = y - yhat
        print("a =", a.shape)
        print(a)
        b = np.abs(a)
        print("b =", b.shape)
        print(b)
        c = np.mean(b, axis)
        print("c =", c.shape)
        print(c)
        input()
    return np.mean(np.abs(y - yhat), axis)


def mean_square_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    if not mask is None:
        y = np.ma.masked_array(y, mask)
        yhat = np.ma.masked_array(yhat, mask)
        if debug:
            print("mask =", mask.shape)
            print(mask)
            print("y =", y.shape)
            print(y)
            print("yhat =", yhat.shape)
            print(yhat)
    # Compute
    if debug:
        a = y - yhat
        print("a =", a.shape)
        print(a)
        b = a**2
        print("b =", b.shape)
        print(b)
        c = np.mean(b, axis)
        print("c =", c.shape)
        print(c)
    return np.mean(np.square(y - yhat), axis)


def mean_absolute_percentage_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    if not mask is None:
        y = np.ma.masked_array(y, mask)
        yhat = np.ma.masked_array(yhat, mask)
        if debug:
            print("mask =", mask.shape)
            print(mask)
            print("y =", y.shape)
            print(y)
            print("yhat =", yhat.shape)
            print(yhat)
    # Compute
    if debug:
        a = y - yhat
        print("a =", a.shape)
        print(a)
        b = a / np.maximum(np.abs(y), eps)
        print("b =", b.shape)
        print(b)
        c = np.abs(b)
        print("c =", c.shape)
        print(c)
        d = np.minimum(c, 1.0)
        print("d =", d.shape)
        print(d)
        e = 100 * d
        print("e =", e.shape)
        print(e)
        f = np.mean(e, axis)
        print("f =", f.shape)
        print(f)
    return np.mean(100.0 * np.minimum(np.abs(div(y - yhat, np.maximum(np.abs(y), eps))), 1.0), axis)


def root_mean_square_error(y, yhat, **kwargs):
    # Unpack args
    debug = kwargs.get("debug", 0)
    # Compute
    if debug:
        a = mean_square_error(y, yhat, **kwargs)
        print("a =", a.shape)
        print(a)
        b = np.sqrt(a)
        print("b =", b.shape)
        print(b)
        kwargs["debug"] = 0
    return np.sqrt(mean_square_error(y, yhat, **kwargs))


def normalized_root_mean_square_error(y, yhat, **kwargs):
    # Unpack args
    mins, maxes = kwargs["mins"], kwargs["maxes"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    debug = kwargs.get("debug", 0)
    # Compute
    if debug:
        a = root_mean_square_error(y, yhat, **kwargs)
        print("a =", a.shape)
        print(a)
        b = a / np.maximum(maxes - mins, eps)
        print("b =", b.shape)
        print(b)
        kwargs["debug"] = 0
    numer = root_mean_square_error(y, yhat, **kwargs)
    denom = np.maximum(maxes - mins, eps)
    return div(numer, denom)
    return numer / denom 


def under_prediction_rate(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 1)
    margin = kwargs.get("margin", 5/100)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    if not mask is None:
        y = np.ma.masked_array(y, mask)
        yhat = np.ma.masked_array(yhat, mask)
        if debug:
            print("mask =", mask.shape)
            print(mask)
            print("y =", y.shape)
            print(y)
            print("yhat =", yhat.shape)
            print(yhat)
    # Compute
    up_count = np.sum(yhat < ((1 - margin) * y), axis)
    if mask is None:
        N = np.prod(np.take(y.shape, axis))
    else:
        N = np.sum(~mask, axis)
    if debug:
        print(y.shape)
        print(y)
        print(mask)
        print(np.sum(mask, axis))
        print(op_count.shape)
        print(op_count)
        print(N.shape)
        print(N)
        input()
    return div(up_count, N)
    return up_count / N


def over_prediction_rate(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    margin = kwargs.get("margin", 5/100)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    if not mask is None:
        y = np.ma.masked_array(y, mask)
        yhat = np.ma.masked_array(yhat, mask)
        if debug:
            print("mask =", mask.shape)
            print(mask)
            print("y =", y.shape)
            print(y)
            print("yhat =", yhat.shape)
            print(yhat)
    # Compute
    op_count = np.sum(yhat > ((1 + margin) * y), axis)
    if mask is None:
        N = np.prod(np.take(y.shape, axis))
    else:
        N = np.sum(~mask, axis)
    if debug:
        print(y.shape)
        print(y)
        print(mask)
        print(np.sum(mask, axis))
        print(op_count.shape)
        print(op_count)
        print(N.shape)
        print(N)
        input()
    return div(op_count, N)
    return op_count / N


def miss_rate(y, yhat, **kwargs):
    # Compute
    return under_prediction_rate(y, yhat, **kwargs) + over_prediction_rate(y, yhat, **kwargs)


def root_relative_square_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mean = kwargs["means"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    if not mask is None:
        y = np.ma.masked_array(y, mask)
        yhat = np.ma.masked_array(yhat, mask)
        if debug:
            print("mask =", mask.shape)
            print(mask)
            print("y =", y.shape)
            print(y)
            print("yhat =", yhat.shape)
            print(yhat)
    # Compute
    numer = np.sum(np.square(y - yhat), axis)
    denom = np.sum(np.square(y - mean), axis)
    return np.sqrt(div(numer, denom, on_div_0=numer))


def correlation(y, yhat, **kwargs):
    axis = kwargs.get("axis", 0)
    mean = kwargs["means"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    if not mask is None:
        y = np.ma.masked_array(y, mask)
        yhat = np.ma.masked_array(yhat, mask)
        if debug:
            print("mask =", mask.shape)
            print(mask)
            print("y =", y.shape)
            print(y)
            print("yhat =", yhat.shape)
            print(yhat)
    # Compute
    numer = np.sum((yhat - mean) * (y - mean), axis)
    denom = np.sqrt(np.sum(np.square(yhat - mean), axis)) * np.sqrt(np.sum(np.square(y - mean), axis))
    return div(numer, denom, on_div_0=0)


def z01_metric(y, yhat, **kwargs):
    mean = kwargs["means"]
    std = kwargs["stds"]
    metric = kwargs.get("metric", "MAE")
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    # Compute
    y_z = np.abs(transform.zscore_transform(y, mean, std))
    _mask = y_z > 1 # use only y in [-1,1]
    if not mask is None:
        _mask = np.logical_or(_mask, mask)
    _kwargs = util.merge_dicts(
        kwargs, 
        {"mask": _mask}, 
    )
    return mean_absolute_error(y, yhat, **_kwargs)


def z12_metric(y, yhat, **kwargs):
    mean = kwargs["means"]
    std = kwargs["stds"]
    metric = kwargs.get("metric", "MAE")
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    # Compute
    y_z = np.abs(transform.zscore_transform(y, mean, std))
    _mask = np.logical_or(y_z <= 1, y_z > 2) # use only y in [-2,-1) & (1,2]
    if not mask is None:
        _mask = np.logical_or(_mask, mask)
    _kwargs = util.merge_dicts(
        kwargs, 
        {"mask": _mask}, 
    )
    return mean_absolute_error(y, yhat, **_kwargs)


def z23_metric(y, yhat, **kwargs):
    mean = kwargs["means"]
    std = kwargs["stds"]
    metric = kwargs.get("metric", "MAE")
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    # Compute
    y_z = np.abs(transform.zscore_transform(y, mean, std))
    _mask = np.logical_or(y_z <= 2, y_z > 3) # use only y in [-3,-2) & (2,3]
    if not mask is None:
        _mask = np.logical_or(_mask, mask)
    _kwargs = util.merge_dicts(
        kwargs, 
        {"mask": _mask}, 
    )
    return mean_absolute_error(y, yhat, **_kwargs)


def z3inf_metric(y, yhat, **kwargs):
    mean = kwargs["means"]
    std = kwargs["stds"]
    metric = kwargs.get("metric", "MAE")
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    debug = kwargs.get("debug", 0)
    # Setup
    # Compute
    y_z = np.abs(transform.zscore_transform(y, mean, std))
    _mask = y_z <= 3 # use only y in (-inf,-3) & (3,inf)
    if not mask is None:
        _mask = np.logical_or(_mask, mask)
    _kwargs = util.merge_dicts(
        kwargs, 
        {"mask": _mask}, 
    )
    return mean_absolute_error(y, yhat, **_kwargs)


regression_metricfn_dict = {
    "MAE": mean_absolute_error,
    "MSE": mean_square_error,
    "MAPE": mean_absolute_percentage_error,
    "RMSE": root_mean_square_error,
    "NRMSE": normalized_root_mean_square_error,
    "UPR": under_prediction_rate,
    "OPR": over_prediction_rate,
    "MR": miss_rate, 
    "RRSE": root_relative_square_error, 
    "CORR": correlation, 
    "Z01": z01_metric, 
    "Z12": z12_metric, 
    "Z23": z23_metric, 
    "Z3INF": z3inf_metric, 
}


#####################
### Other Methods ###
#####################


metric_fn_dict = util.merge_dicts(classification_metricfn_dict, regression_metricfn_dict)


def get_metrics(metrics="regression"):
    if not (isinstance(metrics, str) or (isinstance(metrics, (tuple, list)) and isinstance(metrics[0], str))):
        type_str = type(metrics)
        if isinstance(metrics, (tuple, list)):
            type_str = "%s of %s" % (type(metrics), type(metrics[0]))
        raise TypeError("Input metrics may be str or tuple/list of str. Received %s" % (type_str))
    if metrics == "*":
        metrics = metric_fn_dict.keys()
    elif metrics == "classification":
        metrics = classification_metricfn_dict.keys()
    elif metrics == "regression":
        metrics = regression_metricfn_dict.keys()
    else:
        raise ValueError("Input metrics may be one of \"*\", \"classification\", or \"regression\" when str.")
    return metrics


def evaluate(y, yhat, datatype, metrics="regression", **kwargs):
    # Handle arguments
    metrics = get_metrics(metrics)
    # Start
    con = Container()
    if 0:
        print(y.shape)
        print(y)
        print(yhat.shape)
        print(yhat)
        input()
    if datatype == "spatial":
        kwargs = util.merge_dicts({"axis": tuple(_ for _ in range(y.ndim-1))}, kwargs)
        for metric in metrics:
            scores = metric_fn_dict[metric](y, yhat, **kwargs)
            con.set(metric, scores)
            if 0:
                print(metric_fn_dict[metric])
                print(metric, "=", scores, con.get(metric))
                input()
    elif datatype == "temporal": 
        kwargs = util.merge_dicts({"axis": tuple(_ for _ in range(y.ndim-1))}, kwargs)
        for metric in metrics:
            con.set(metric, metric_fn_dict[metric](y, yhat, **kwargs))
    elif datatype == "spatiotemporal":
        kwargs = util.merge_dicts({"axis": tuple(_ for _ in range(y.ndim-2))}, kwargs)
        for metric in metrics:
            con.set(metric, metric_fn_dict[metric](y, yhat, **kwargs))
    elif datatype == "graph":
        raise NotImplementedError()
    return con


def evaluate_datasets(ys, yhats, datasets, datatypes, partitions, metrics="regression", **kwargs):
    """ Computes a variety of scoring metrics to evaluate the performance of a model given groundtruth and predicted values.

    Arguments
    ---------
    ys : ndarray or tuplei/list of ndarray
    yhats : ndarray or tuple/list of ndarray
    datasets: Data.Data object or tuple/list of Data.Data objects
    datatypes : str or tuple/list of str
    partitions : str or tuple/list of str
    metrics : str or tuple/list of str
    kwargs : ...

    Returns
    -------

    """
    # Handle args
    if not isinstance(ys, (tuple, list)):
        ys = [ys]
    if not isinstance(yhats, (tuple, list)):
        yhats = [yhats for _ in ys]
    if not isinstance(datasets, (tuple, list)):
        datasets = [datasets for _ in ys]
    if isinstance(datatypes, str):
        datatypes = [datatypes for _ in ys]
    if isinstance(partitions, str):
        partitions = [partitions for _ in ys]
    # Start
    con = Container()
    for i in range(len(ys)):
        stats = datasets[i].get(datatypes[i]).statistics.to_dict()
        axis = (-1,)
        indices = datasets[i].get(datatypes[i]).misc.response_indices
        if datatypes[i] in ("spatial", "temporal"):
            pass
        elif datatypes[i] == "spatiotemporal":
            axis = (-2, -1)
            indices = (
                datasets[i].get(datatypes[i]).original.get("spatial_indices", partitions[i]), 
                datasets[i].get(datatypes[i]).misc.response_indices
            )
        elif datatypes[i] == "graph":
            raise NotImplementedError()
        else:
            raise ValueError()
        stats = datasets[i].get(datatypes[i]).filter_axis(stats, axis, indices)
        _kwargs = util.merge_dicts(
            util.remap_dict(
                stats, 
                {"minimums": "mins", "maximums": "maxes", "medians": "meds", "standard_deviations": "stds"}
            ), 
            kwargs
        )
        _kwargs = util.remap_dict(_kwargs, {"%s_mask"%(partitions[i]): "mask"})
        eval_con = evaluate(ys[i], yhats[i], datatypes[i], metrics, **_kwargs)
        names = eval_con.get_names()
        values = [eval_con.get(_) for _ in names]
        con.set(names, values, partitions[i], multi_value=1)
    return con


def evaluation_to_report(eval_con, datasets, datatypes, partitions, n_decimal=8, debug=0):
    if isinstance(partitions, str):
        partitions = [partitions]
    if not isinstance(datasets, (tuple, list)):
        datasets = [datasets for _ in partitions]
    if not isinstance(datatypes, (tuple, list)):
        datatypes = [datatypes for _ in datatypes]
    # make sure metrics are in same order as in metric_fn_dict
    metrics = []
    _ = {_: None for _ in eval_con.get_names()}
    for metric in metric_fn_dict.keys():
        if metric in _:
            metrics.append(metric)
    #
    lines = []
    for metric in metrics:
        for dataset, datatype, partition in zip(datasets, datatypes, partitions):
#            if partition == "test": debug = 1
            feature_labels = dataset.get(datatype).misc.response_features
            try:
                scores = eval_con.get(metric, partition)
            except:
                scores = eval_con.get(metric, partition).copy() # copying due to: "ValueError: output array is read-only"
            # Add partition-level metric score
            if debug:
                print(eval_con)
                print(metric)
                print(scores.shape)
                print(scores)
                input()
            score = np.round(np.nanmean(scores), n_decimal)
            if int(score) == score:
                score = int(score)
            lines.append("%s %s = %s" % (partition, metric, score))
            for j, feature_label in enumerate(feature_labels):
                # Add feature-level metric score for this partition
                score = np.round(np.nanmean(scores[...,j]), n_decimal)
                if int(score) == score:
                    score = int(score)
                lines.append("\t%s %s = %s" % (feature_label, metric, score))
                if datatype == "spatiotemporal":
                    spatial_label_field = dataset.get(datatype).misc.spatial_label_field
                    spatial_labels = dataset.get(datatype).original.get("spatial_labels", partition)
                    for k, spatial_label in enumerate(spatial_labels):
                        # Add spatial-level metric score for this feature and partition
                        score = np.round(scores[k,j], n_decimal)
                        if debug:
                            print(scores[k,j])
                            print(score)
                        if np.isnan(score) or isinstance(score, np.ma.core.MaskedConstant):
                            pass
                        elif int(score) == score:
                            score = int(score)
                        lines.append("\t\t%s %8s %s = %s" % (spatial_label_field, spatial_label, metric, score))
                        if debug:
                            print(lines[-1])
                            input()
    return "\n".join(lines)


def curate_evaluation_report(ys, yhats, datasets, datatypes, partitions, metrics, **kwargs):
    return evaluation_to_report(
        evaluate_datasets(ys, yhats, datasets, datatypes, partitions, metrics, **kwargs), 
        datasets, datatypes, partitions
    )
