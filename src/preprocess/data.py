import sys
sys.path.append('../')
sys.path.append('./src/')
sys.path.append('../constants.py')


import os
import pandas as pd
from collections import defaultdict
import numpy as np
from constants import *
from utils import *
from preprocess.drugbank import *

def load_file_w_ext(path, suffix):
    for f in os.listdir(path):
        if f.endswith(suffix):
            return f
    return None


def get_graph_size(graph, subgraphidx=-1):
    if subgraphidx >= 0:
        return sum([len(ts) for h, ts in graph[subgraphidx].items()])
    s = 0
    for r, subgraph in graph.items():
        s += sum([len(ts) for h, ts in subgraph.items()])
    return s


# def load_dataset(data_id):
#     path = "./data/" + data_id + "/"
#     x, graph = load_file_w_ext(path, PK_SUFFIX_X), load_file_w_ext(path, PK_SUFFIX_KG)
#
#     if x is not None and graph is not None:
#         return pk_load(path + x), pk_load(path + graph)


def prep_decagon():
    h_idx = 0
    t_idx = 1
    df, cs, es, rs = load_csv_data(DECAGON_ORIG_RAW)

    pc2dbid = load_pc_drugbank_id_map()
    df[cs[h_idx]] = df[cs[h_idx]].apply(lambda x: pc2dbid[x] if x in pc2dbid else x)
    df[cs[t_idx]] = df[cs[t_idx]].apply(lambda x: pc2dbid[x] if x in pc2dbid else x)

    df.to_csv(DECAGON_RAW, index=None, header=True)


def create_dataset(name, path, adde=True, use_ssp=True):
    df, cs, es, rs = load_csv_data(path)
    print(len(es))

    if adde:
        df2, cs2, es2, rs2 = load_csv_data(DRUGBANK_MIXTURES)
        print(len(es2))
        ov = []
        for e in es2:
            if e in es:
                ov.append(e)
        print(len(ov))
        df2.columns = df.columns
        df = df.append(df2)

    h_idx = 0
    t_idx = 1
    r_idx = 2
    # df = pd.read_csv(path, header=0)
    # cs = df.columns.tolist()
    # df = df[cs[0:3]]
    #
    # es = (df[cs[h_idx]].append(df[cs[t_idx]])).unique().tolist()
    # print(len(es))
    # if adde:
    #     df2 = pd.read_csv(DRUGBANK_MIXTURES, header=0)
    #     cs2 = df2.columns.tolist()
    #     es2 = (df2[cs2[h_idx]].append(df2[cs2[t_idx]])).unique().tolist()
    #     print(len(es2))
    #     ov = []
    #     for e in es2:
    #         if e in es:
    #             ov.append(e)
    #     print(len(ov))
    #     df = [df, df2]

    es = (df[cs[h_idx]].append(df[cs[t_idx]])).unique().tolist()
    rs = df[cs[r_idx]].unique().tolist()
    es = {s: i for i, s in enumerate(es)}
    rs = {s: i for i, s in enumerate(rs)}
    print("es:", len(es))
    print("rs:", len(rs))

    # j = 0
    # k = 0
    graph = {i: defaultdict(list) for i in range(len(rs))}
    for i, row in df.iterrows():
        # l = rs[row[cs[r_idx]]]
        # if l == 86:
        #     j += 1
        # k += 1
        graph[rs[row[cs[r_idx]]]][es[row[cs[h_idx]]]].append(es[row[cs[t_idx]]])
    print("gs:", get_graph_size(graph))
    # print(i)
    # print(j,k)

    if use_ssp:     # TO DO: generate SSP for negative evidence
        pca = pd.read_csv('./data/deepddi/tanimoto_info_PCA50.csv', index_col=0)
        x = np.zeros((len(es), pca.shape[1]))
        for key, val in es.items():
            if key in pca.loc:
                x[val] = pca.loc[key].tolist()
    else:
        print("gs:", get_graph_size(graph, subgraphidx=86))
        x = np.eye(len(es))

    dspath = "./data/" + name + "/"
    if not os.path.exists(dspath):
        os.makedirs(dspath)

    i2 = path.rindex("/") if not path.endswith("/") else path[:-1].rindex("/")
    i1 = path[:i2].rindex("/")
    p = path[:i2]
    id = path[i1+1:i2]

    if use_ssp:
        pk_save(x, dspath + PK_PREFIX + "." + id + "-ssp" + "." + PK_SUFFIX_X)
        pk_save(graph, dspath + PK_PREFIX + "." + id + "-ssp" + "." + PK_SUFFIX_KG)
    else:
        pk_save(x, dspath + PK_PREFIX + "." + id + "." + PK_SUFFIX_X)
        pk_save(graph, dspath + PK_PREFIX + "." + id + "." + PK_SUFFIX_KG)



    return es, rs, graph


if __name__ == "__main__":
    #
    # TODO later try to mp db tp pc and see if more overlap between decagon and mixtures
    # prep_decagon()

    for data_id in [DATA_DEEPDDI]:#[DATA_DECAGON]:
        path = "./data/" + data_id + "/"
        # create_dataset(data_id + "-me", path + load_file_w_ext(path, "csv"))

        # use ssp features when not using negative evidence, TO DO: find a way to generate SSP for negative evidence
        create_dataset(data_id + "-me", path + load_file_w_ext(path, "csv"), adde=True)
        # create_dataset(data_id, DATA2RAW[data_id], adde=False)
