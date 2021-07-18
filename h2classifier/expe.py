#!/usr/bin/env python3

import json
import random
import sys

import data_tools as dt
import rf_tools as rf

# PARAMS :
args = sys.argv

# Fichier des tailles de record avec direction (+ reception, - emmision)
f_rec = args[1]
f_rec_noise = args[2]




print("Init !")

with open(f_rec, "r") as f:
    data = json.load(f)

    # construiction des data pour le training et testing
    data = dict([(i,j) for i,j in data.items()][:200])

    d_train, d_test = dt.split_dataset(data, 4, 50)

with open(f_rec_noise, "r") as f:
    data = json.load(f)

    d_train_noise, d_test_noise = dt.split_dataset(data, 5000, 0)

    noise_label = list(d_train_noise.keys())[0]
    l_tmp = [i for i in d_train_noise[noise_label]]
    random.shuffle(l_tmp)
    d_train_noise[noise_label] = l_tmp[:500]


    stats = dt.stats_training(d_train)

    print("Start features extraction (It take some time)")
    test_lab, test_feat, index_features = dt.extract_features(d_test, stats)
    print("testing without noise : ok")
    train_lab, train_feat, _ = dt.extract_features(d_train, stats)
    print("training without noise : ok")
    _, train_noise_feat, _ = dt.extract_features(d_train_noise, stats)
    print("training noise : ok")
    _, test_noise_feat, _ = dt.extract_features(d_test_noise, stats)
    print("testing noise : ok")

print("Evaluation !")

test_lab_noise = ["noise___" for _ in test_noise_feat]
test_feat_all = test_feat + test_noise_feat

for params_ in rf.gen_params(n_estimators = [150]):

    n_train_noise, n_feats,  params = params_
    train_noise_feat = train_noise_feat[:n_train_noise]
    train_lab_noise = ["noise___" for _ in train_noise_feat]

    print("n_feats : {}, nb_train_noise : {}, params : {}".format(n_feats, n_train_noise, params))

    clf, keeping_index = rf.training(params, train_lab + train_lab_noise, train_feat + train_noise_feat, True, nb_feat = n_feats)
    #pickle.dump(clf, open("model.scikit", "wb"))

    pred_lab = rf.prediction(clf, test_feat_all, True, keeping_index = keeping_index)

    result = rf.scoring(pred_lab, test_lab + test_lab_noise)

    print(result)
