#!/usr/bin/env python3
# coding : utf-8

import random
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def gen_params(random_state=-1, max_depth=None, n_estimators=None, max_samples=None,
               n_feats=None, n_train_leg=None):
    # nb training noise
    n_train_leg = [150] if n_train_leg is None else n_train_leg
    for a in sorted(n_train_leg, reverse=True):
        # profondeur
        max_depth = [50] if max_depth is None else max_depth
        for b in max_depth:
            # nb estimateurs
            n_estimators = [300] if n_estimators is None else n_estimators
            for c in n_estimators:
                # nb sample bagging
                max_samples = [0.6] if max_samples is None else max_samples
                for d in max_samples:
                    # nb features kept
                    n_feats = [300] if n_feats is None else n_feats
                    for e in n_feats:
                        params = dict({
                            "max_depth": b,
                            "n_jobs": -1,
                            "n_estimators": c,
                            "max_samples": d,
                            "verbose": 1,
                            "random_state": random_state if random_state > 0 else random.randint(0, 10000)
                        })
                        yield a, e, params


def prediction(clf, eval_feat, monitoring=False, keeping_index=None):
    if keeping_index is not None:
        reduce_feat = list()
        for v_feat in eval_feat:
            reduce_feat.append([feat for index, feat in enumerate(v_feat) if index in keeping_index])

        eval_feat = reduce_feat

    feat_ = np.array(eval_feat)
    #pickle.dump(feat_, open("feat.scikit", "wb"))

    if monitoring:
        print("Predictions")
        t_0 = float(time.time())

    pred_lab = clf.predict(feat_)

    if monitoring:
        t_1 = float(time.time())
        print("Time spend for prediction : {}".format(float(t_1 - t_0)))

    return pred_lab


def training(params, labels, features, monitoring=False, nb_feat=-1, feature_importances=None):
    print("Training")
    clf = RandomForestClassifier(**params)
    feat_ = np.array(features)
    lab_ = np.array(labels)

    if monitoring:
        t_0 = float(time.time())

    print("First fit for learning feature_importances")
    clf.fit(feat_, lab_)
    feature_importances = clf.feature_importances_

    if monitoring:
        t_1 = float(time.time())
        print("Time spend for learning feature_importances : {}".format(t_1 - t_0))

    # if we need to reduce the number of features (nb feat > 0)
    if nb_feat > 0:

        print("Second fit but with only the {} best features".format(nb_feat))

        # find the worst score of the first top nb_feats importance score
        min_importance_score = min(sorted(feature_importances, reverse=True)[:nb_feat])

        if min_importance_score > 0:
            keeping_index = set(
                [index for index, score in enumerate(feature_importances) if score >= min_importance_score])

            reduce_feat = list()
            for v_feat in feat_:
                reduce_feat.append([i for index, i in enumerate(v_feat) if index in keeping_index])

            feat_2 = np.array(reduce_feat)

            clf.fit(feat_2, lab_)
            return clf, keeping_index

    return clf, None


def scoring(pred_labs, true_labs):
    TP = 0
    TN = 0
    WC = 0
    FP = 0
    FN = 0

    for index, p_lab in enumerate(pred_labs):
        t_lab = true_labs[index]

        if t_lab == p_lab:
            if t_lab == "noise___":
                TN += 1
            else:
                TP += 1
        else:
            if t_lab == "noise___":
                FP += 1
            elif p_lab == "noise___":
                FN += 1
            else:
                WC += 1

    results = dict({
        "TP": TP,
        "TN": TN,
        "WC": WC,
        "FP": FP,
        "FN": FN,
        "precision": float(TP + TN) / float(TP + TN + FP + FN + WC),
        "total_sum": TP + TN + WC + FP + FN
    })

    return results
