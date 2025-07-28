""" Performance evaluation helpers """

__author__ = "Thomas Kaplan"

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import warnings
from scipy.stats import spearmanr, pearsonr, skew, skewtest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, mean_absolute_error
from confidenceinterval import (
    f1_score,
    tpr_score,
    tnr_score,
    roc_auc_score,
    classification_report_with_ci
)

def opt_youden_j_binary(y_test, y_pred_proba, avg_method="macro"):
    fpr, tpr, thresh = roc_curve(y_test, y_pred_proba)
    idx = np.argmax(tpr - fpr)
    dec_thresh = thresh[idx]
    sens = tpr_score(y_test, y_pred_proba >= dec_thresh, average=avg_method)
    spec = tnr_score(y_test, y_pred_proba >= dec_thresh, average=avg_method)
    f1 = f1_score(y_test, y_pred_proba >= dec_thresh, average=avg_method)
    auc = roc_auc_score(y_test, y_pred_proba, average=avg_method)
    return dec_thresh, sens, spec, f1, auc

def opt_roc_curve(y_test, y_pred_proba, n_p_thresh=100):
    p_thresh = np.linspace(0, 1, n_p_thresh)
    senss = [tpr_score(y_test, y_pred_proba >= t)[0] for t in p_thresh]
    specs = [tnr_score(y_test, y_pred_proba >= t)[0] for t in p_thresh]
    return senss, specs

def print_classification_performance_summary(
    y_true, y_pred, y_pred_proba, multi_class=False, labels=None, n_p_thresh=50,
):
    print("Classification (CI):")
    with warnings.catch_warnings(action="ignore"):
        # Lots of statsmodels RuntimeWarning error in divide by zero
        report = classification_report_with_ci(y_true, y_pred)
    print(report)
    if not multi_class:
        print('AUC (pred):', roc_auc_score(y_true.astype(int), y_pred))
        print('AUC (proba):', roc_auc_score(y_true.astype(int), y_pred_proba))
        print("> Youden-J optimised:")
        thresh, sens, spec, f1, auc = opt_youden_j_binary(y_true.astype(np.int32), y_pred_proba)
        print('Thresh:', thresh)
        print('Sens:', sens)
        print('Spec:', spec)
        print('F1:', f1)
        print('AUC:', auc)
        senss, specs = opt_roc_curve(y_true.astype(np.int32), y_pred_proba, n_p_thresh=n_p_thresh)
        return sens, spec, senss, specs
    else:
        print("> Per-class Youden-J optimised:")
        classes = np.unique(y_true)
        senss_specs = []
        opt_senss_specs = []
        macro_averages = []
        p_thresh = np.linspace(0, 1, n_p_thresh)
        for i, class_ in enumerate(classes):
            y_true_binary = [1 if y == class_ else 0 for y in y_true]
            y_pred_binary = [1 if y == class_ else 0 for y in y_pred]
            thresh, sens, spec, f1, auc = opt_youden_j_binary(
                y_true_binary, y_pred_proba[:, i]
            )
            print(class_, thresh, sens, spec, f1, auc)
    print()


def print_regression_performance_summary(y_true, y_pred, n_bootstr=5000):
    print("Regression: ")
    spr = spearmanr(y_pred, y_true)
    print("> SpearmanR:", spr)
    pr = pearsonr(y_pred, y_true)
    print("> PearsonR:", spr)
    model = sm.OLS(y_true, sm.add_constant(y_pred))
    results = model.fit()
    print("> OLS:", results.summary())
    print("> MAE:")
    errors = np.abs(y_true - y_pred)
    bootstrap_mae = np.array(
        [
            np.mean(np.random.choice(errors, size=len(errors), replace=True))
            for _ in range(n_bootstr)
        ]
    )
    lower, upper = np.percentile(bootstrap_mae, [2.5, 97.5])
    print(mean_absolute_error(y_pred, y_true), lower, upper)
    print("> Skew (pred - real):")
    errors = y_pred - y_true
    bootstrap_me = np.array(
        [
            np.mean(np.random.choice(errors, size=len(errors), replace=True))
            for _ in range(n_bootstr)
        ]
    )
    lower, upper = np.percentile(bootstrap_me, [2.5, 97.5])
    print("Mean, ", np.mean(errors), lower, upper)
    print("Skew/Skewtest, ", skew(y_pred - y_true), skewtest(y_true - y_pred))
    print()


def _predictions(model, loader):
    out = []
    indices = []
    for X_batch, X_meta_batch, y_batch, inds in loader:
        ys = model(X_batch, X_meta_batch)
        preds = ys.flatten()
        out.append(torch.vstack([preds, y_batch]))
        indices.append(inds.detach().numpy())
    indices = list(itertools.chain(*indices))
    predictions = torch.hstack(out).detach().numpy().T
    return indices, predictions[:, 0]


def lvh_threshold_learning(model, val_loader):
    indices_val, y_pred_val = _predictions(model, val_loader)
    X, y = [], []
    for i, pred in zip(indices_val, y_pred_val):
        df_rec = val_loader.dataset.get_record(i)
        X.append([df_rec["Sex"] == 1, pred])
        y.append(df_rec["LVM.group"])
    y = np.array(y)
    X = np.array(X)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=25000, random_state=20240302
    )
    clf.fit(X, y)
    return clf

def lvm_recalibration(model, val_loader, col='LVM'):
    indices_val, y_pred_val = _predictions(model, val_loader)
    X, y = [], []
    for i, pred in zip(indices_val, y_pred_val):
        df_rec = val_loader.dataset.get_record(i)
        X.append([df_rec["Sex"] == 1, pred])
        y.append(df_rec[col])
    y = np.array(y)
    X = np.array(X)
    clf = LinearRegression()
    clf.fit(X, y)
    return clf

