import sys
sys.path.append('../')
sys.path.append('./src/')
sys.path.append('./')
sys.path.append('./constants.py')

import torch
import xml.etree.ElementTree as ET
import numpy as np
import os
from constants import *

DRUGBANK_RAW = "./data/full_database.xml"
DRUGBANK_PRE = "{http://www.drugbank.ca}"


def save_lines(ls, path):
    with open(path, "w+") as f:
        for l in ls:
            f.write(l + "\n")


def save_map(map, path):
    with open(path, "w+") as f:
        for k, v in map.items():
            print(k,v)
            f.write(k + "," + v + "\n")


def load_map(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            ls = [l.strip().split(",") for l in f.readlines()]
            return {",".join(l[:-1]): l[-1] for l in ls}

    return {}


def save_lists(ls, path):
    with open(path, "w") as f:
        for l in ls:
            f.write(",".join(l) + "\n")


def load_lists(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [l.strip().split(",") for l in f.readlines()]

    return []


def load_drugbank_name_id_map():
    map = load_map(DRUGBANK_N2ID)
    if map:
        return map

    prefix = DRUGBANK_PRE
    tree = ET.parse(DRUGBANK_RAW)
    root = tree.getroot()
    for drug in root:
        id = None
        for ide in drug.findall(prefix + 'drugbank-id'):
            if "primary" in ide.attrib:
                if ide.attrib["primary"] == "true":
                    id = ide.text
                    break
        if id is None:  # ERROR
            continue

        name = drug.find(prefix + 'name').text
        names = [] if name is None else [name]
        for se in drug.find(prefix + 'synonyms'):
            names.append(se.text)
        if names:
            map.update({name: id for name in names})

    save_map(map, DRUGBANK_N2ID)
    return map


def load_pc_drugbank_id_map():
    map = load_map(DRUGBANK_PCID2ID)
    if map:
        return map

    prefix = DRUGBANK_PRE
    tree = ET.parse(DRUGBANK_RAW)
    root = tree.getroot()
    for drug in root:
        dbids = []
        # for dbid in drug.findall(prefix + 'drugbank-id'):
        #     dbids.append(dbid.text)
        for ide in drug.findall(prefix + 'drugbank-id'):
            if "primary" in ide.attrib:
                if ide.attrib["primary"] == "true":
                    dbids = [ide.text]
                    break
        if not dbids:  # ERROR
            continue

        eids = drug.find(prefix + 'external-identifiers')
        if eids is None:
            continue
        pccid = None
        pcsid = None
        pcaid = None
        for eid in eids:
            rtext = eid.find(prefix + 'resource').text.lower()
            if "pubchem" in rtext:
                id = str(eid.find(prefix + 'identifier').text).zfill(9)  # format as in decagon
                if "compound" in rtext:
                    pccid = "CID" + id
                    break
                elif "substance" in rtext:
                    pcsid = "SID" + id
                else:
                    pcaid = "AID" + id
        for pcid in [pccid,pcsid,pcaid]:
            if pcid is not None:
                for dbid in dbids:
                    map[pcid] = dbid
    save_map(map, DRUGBANK_PCID2ID)
    return map


# not adapted yet
# def load_drugbank_id_pcid_map():
#     map = load_map(DRUGBANK_ID2PCID)
#     if map:
#         return map
#
#     prefix = DRUGBANK_PRE
#     tree = ET.parse(DRUGBANK_RAW)
#     root = tree.getroot()
#     for drug in root:
#         dbids = []
#         for dbid in drug.findall(prefix + 'drugbank-id'):
#             dbids.append(dbid.text)
#         eids = drug.find(prefix + 'external-identifiers')
#         if eids is None:
#             continue
#         pccid = None
#         pcsid = None
#         pcaid = None
#         for eid in eids:
#             rtext = eid.find(prefix + 'resource').text.lower()
#             if "pubchem" in rtext:
#                 id = str(eid.find(prefix + 'identifier').text).zfill(9)  # format as in decagon
#                 if "compound" in rtext:
#                     pccid = "CID" + id
#                     break
#                 elif "substance" in rtext:
#                     pcsid = "SID" + id
#                 else:
#                     pcaid = "AID" + id
#         # TODO not sure which is more important, but decagon consider s cids so first use those
#         pcid = pccid if pccid is not None else pcsid if pcsid is not None else pcaid
#         if pcid is not None:
#             for dbid in dbids:
#                 map[dbid] = pcid
#             print(pcid, dbids)
#     save_map(map, DRUGBANK_ID2PCID)
#     return map


def load_db_mixtures():

    ls = load_lists(DRUGBANK_MIXTURES)
    if ls:
        return ls
    print('loaded list of mixtures')

    prefix = DRUGBANK_PRE
    tree = ET.parse(DRUGBANK_RAW)
    root = tree.getroot()
    for drug in root:
        ms = drug.find(prefix + 'mixtures')
        for m in ms.findall(prefix + 'mixture'):
            txt = m.find(prefix + 'ingredients').text
            if not "+" in txt:
                continue
            if txt not in ls:
                ls.append(txt)

    ls = [[s.strip() for s in txt.split("+")] for txt in ls]
    n2id = load_drugbank_name_id_map()
    ls2 = ["D1,D2,Interact"]
    for l in ls:
        l2 = [n2id[s] for s in l if s in n2id]
        if l2 and len(l2) > 1:
            for i in range(len(l2)):
                for j in range(i+1,len(l2)):
                    if i != j:
                        s = str(l2[i]) + "," + str(l2[j]) + ",rel"
                        if s not in ls2:
                            ls2.append(s)
                        # ls2.append([l2[i], l2[j], "rel"])
    save_lines(ls2, DRUGBANK_MIXTURES)
    print('saved interactions')
    return ls2


# version that does not create a binary matrix (which drops info)
# def load_db_mixtures():
#
#     ls = load_lists(DRUGBANK_MIXTURES)
#     if ls:
#         return ls
#
#     prefix = DRUGBANK_PRE
#     tree = ET.parse(DRUGBANK_RAW)
#     root = tree.getroot()
#     for drug in root:
#         ms = drug.find(prefix + 'mixtures')
#         for m in ms.findall(prefix + 'mixture'):
#             txt = m.find(prefix + 'ingredients').text
#             if not "+" in txt:
#                 continue
#             if txt not in ls:
#                 ls.append(txt)
#
#     ls = [[s.strip() for s in txt.split("+")] for txt in ls]
#     n2id = load_drugbank_name_id_map()
#     ls2 = []
#     for l in ls:
#         l2 = [n2id[s] for s in l if s in n2id]
#         if l2:
#             ls2.append(l2)
#         # print(l)
#         print(len(l)-len(l2),l2)
#     save_lists(ls2, DRUGBANK_MIXTURES)
#     return ls2


if __name__ == "__main__":
    print("prepare drugbank")
    # RUN ALL
    #
    load_drugbank_name_id_map()
    load_pc_drugbank_id_map()
    load_db_mixtures()


