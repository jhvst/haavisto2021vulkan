#!/usr/bin/env python3
# coding : utf-8

import numpy as np
from collections import Counter

def get_features_index(stats):

    legend = ["a1", "a2", "a3", 
             "b1mean", "b1max","b1min", "b1std", "b1median",
             "b2mean", "b2max","b2min", "b2std", "b2median",
             "c1", "c2"] + ["dserv{}".format(i) for i in range(1,21)] + [
             "dcli{}".format(i) for i in range(1,21)] + [
             "eserv_taille{}".format(i) for i in sorted([j for j, _ in stats["count_resp"].items()])] + [
             "ecli_taille{}".format(i) for i in sorted([j for j, _ in stats["count_req"].items()])] 

    return legend
 

def get_features(data, stats):

    features = dict()

    count_cli = stats["count_req"]
    count_serv = stats["count_resp"]

    rec_cli = []
    rec_serv = []
    chunks = list()
    bursts = list()
    chunk_tmp = None
    burst_tmp = None

    for index, el in enumerate(data):

        if index % 20 == 0:
            if chunk_tmp is not None :
                chunks.append(chunk_tmp) 
            chunk_tmp = 0

        if el >= 0:
            rec_serv.append(el)

            if chunk_tmp is not None:
                chunk_tmp += 1

            if burst_tmp is not None:
                burst_tmp.append(el)
        else:
            rec_cli.append(abs(el))

            if burst_tmp is not None:
                if len(burst_tmp) > 0:
                    bursts.append(sum(burst_tmp))
            burst_tmp = list()
            
            
    set_rec_serv = set(rec_serv)
    set_rec_cli = set(rec_cli)


    # 1 : nb total record
    features["1"] = len(data)

    # 2 : nb record sortants (req)
    features["2"] = len(rec_cli)

    # 3 : somme taille total des records (req + resp)
    features["3"] = sum(rec_cli) + sum(rec_serv)

    # TODO 3' : idee => taille des records sortants, utiliser ratio ?


    # 4-8 : bursts methode 2 - bursts
    # TODO : ajouter len(bursts) aux features

    if len(bursts) > 0:
        features["4-8"] = [np.mean(bursts), max(bursts), min(bursts), np.std(bursts), np.median(bursts)]
    else :
        features["4-8"] = [0]*5

    # 9-13 : burst methode 1 - chunks
    # TODO : ajouter len(chunks) aux features

    if len(chunks) > 0:
        features["9-13"] = [np.mean(chunks), max(chunks), min(chunks), np.std(chunks), np.median(chunks)]
    else :
        features["9-13"] = [0]*5
    

    # 14 : nb taille de records differents (entrant)
    features["14"] = len(set_rec_serv)

    # 15 : nb taille de records differents (sortant)
    features["15"] = len(set_rec_cli)


    # 16 - 35 : tailles présente dans la trace mais ayant le moins d'occurence globalement (entrant)
    features["16-35"] = sorted(list(set_rec_serv), key=lambda x:count_serv[x])[:20]

    if len(features["16-35"]) < 20 :
        features["16-35"] = features["16-35"] + [0]*(20-len(features["16-35"]))

    # 36 - 55 : tailles présente dans la trace mais ayant le moins d'occurence globalement (sortant)
    features["36-55"] = sorted(list(set_rec_cli), key=lambda x:count_cli[x])[:20]

    if len(features["36-55"]) < 20 :
        features["36-55"] = features["36-55"] + [0]*(20-len(features["36-55"]))

    # frequences tailles de records (entrants)
    sizes_allowed_sorted = sorted([size for size, _ in count_serv.items()])
    c = Counter(rec_serv)
    features["count_serv"] = [c[i] for i in sizes_allowed_sorted]
    
    if len(features["count_serv"]) != len(count_serv) :
        print("missing feat here ! 3")
        return None

    # frequences tailles de records (sortants)
    sizes_allowed_sorted = sorted([size for size, _ in count_cli.items()])
    c = Counter(rec_cli)
    features["count_cli"] = [c[i] for i in sizes_allowed_sorted]

    if len(features["count_cli"]) != len(count_cli) :
        print("missing feat here ! 4")
        return None

    features_vector = [features["1"] + features["2"] + features["3"]] + features["4-8"] + features["9-13"] + [
        features["14"] + features["15"]] + features["16-35"] + features["36-55"] + features["count_serv"] + features["count_cli"]

    return features_vector
