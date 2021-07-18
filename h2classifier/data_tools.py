#!/usr/bin/env python3
# coding : utf-8

#import numpy as np
import features_tools as ft
import random
import copy
from collections import Counter
import pickle


def split_dataset(data, n_ech_test, min_ech_train):
    """
        split the dataset => training / testing
    """

    data_test = dict()
    data_train = dict()

    for kw, l_lists in data.items() :
        if len(l_lists) > n_ech_test:
            random.shuffle(l_lists)
            if len(l_lists[n_ech_test:]) >= min_ech_train:
                data_train.setdefault(kw, l_lists[n_ech_test:])
                data_test.setdefault(kw, l_lists[:n_ech_test])
            else:
                print("{} : discard -> not enough sample training (lower than {})".format(kw, min_ech_train)) 
        else: 
            print("{} : discard -> not enough sample (lower than {})".format(kw, n_ech_test)) 

    return data_train, data_test
    

def stats_training(data) :

    sizes_list = []

    for _, l_lists in data.items() :
        sizes_list += [i for l in l_lists for i in l]

    count_req = Counter([abs(i) for i in sizes_list if i < 0.0])
    count_resp = Counter([i for i in sizes_list if i >= 0.0])

    stats = dict({"count_req" : count_req, "count_resp" : count_resp})

    return stats


def extract_features(data, stats):

    vect_feat = list()
    vect_label = list()

    features_index = ft.get_features_index(stats)

    for kw, l_traces in data.items() :
        for l in l_traces :
            feats = ft.get_features(l, stats)
            if feats is not None:
                vect_feat.append(feats)
                vect_label.append(kw)

    return vect_label, vect_feat, features_index

