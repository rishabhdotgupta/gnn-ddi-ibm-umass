import sys
sys.path.append('../')
sys.path.append('./src/')
sys.path.append('./')
sys.path.append('./constants.py')

import torch
import xml.etree.ElementTree as ET
import numpy as np
import os
from src.constants import *

r""" Check how many nodes overlap between DeepDDI and the safe edge db.
"""

def clean(bad, good):
    r""" Remove repeated pairs or pairs that occur in both sets.
    """
    bad_cols = [(a[0], a[1], a[2]) for a in bad]
    good_cols = [(a[0], a[1], a[2]) for a in good]

    bad_cols = list(set(bad_cols))
    good_cols = set(good_cols)

    for t in bad_cols:
        if t in good_cols:
            good_cols.remove(t)
    return bad_cols, list(good_cols)


def unique_drugs(split_data):
    cols = tuple(zip(*[(a[0], a[1]) for a in split_data]))
    return set(cols[0] + cols[1])

def rel_count(split_data):
    r""" For each relation, count the number of times they appear.
    """
    counts = {}
    for t in split_data:
        rel = t[2]
        if rel in counts:
            counts[rel] += 1
        else:
            counts[rel] = 1
    return counts

def get_freq(split_data, freq=500):
    rel_counts = rel_count(split_data)
    freq_rels = []
    for rel, count in rel_counts.items():
        if count > freq:
            freq_rels.append(rel)
    return set(freq_rels)

if __name__ == "__main__":
    print('Size of intersection between DeepDDI and DrugBank mixtures')
    with open(DEEPDDI_RAW_NE, 'r') as deepddi:
        with open(DECAGON_RAW, 'r') as decagon:
            deepddi_lines = deepddi.readlines()[1:]
            safe_id = max([int(a.split(',')[2]) for a in deepddi_lines])
            deepddi_safe = [a.strip().split(',') for a in deepddi_lines if int(a.split(',')[2]) == safe_id]
            print(deepddi_safe[0])

            decagon_lines = decagon.readlines()[1:] # remove header
            decagon_data = [a.strip().split(',') for a in decagon_lines]

            # only consider frequently occuring edges
            freq_rels = get_freq(decagon_data)
            decagon_data = [a for a in decagon_data if a[2] in freq_rels]


            decagon_drugs = unique_drugs(decagon_data)
            good_deepddi_drugs = unique_drugs(deepddi_safe)

            # get the drugs in both sets.
            drug_overlap = list(decagon_drugs & good_deepddi_drugs)

            # safe edges that have an endpoint with an overlapping drug.
            edge_overlap = [edge for edge in deepddi_safe if edge[0] in drug_overlap or edge[1] in drug_overlap]

            decagon_edge_overlap = [edge for edge in deepddi_safe if edge[0] in decagon_drugs or edge[1] in decagon_drugs]
            decagon_edge_overlap2 = [edge for edge in deepddi_safe if edge[0] in decagon_drugs and edge[1] in decagon_drugs]

            print('*** Results ***')

            print(' > # of bad (decagon) drugs: ', len(decagon_drugs))
            print(' > # of good drugs: ', len(good_deepddi_drugs))
            print(' > Size of Overlap: ', len(drug_overlap))

            print(' > # of decagon edges: ', len(decagon_data))
            # these two should be equal...
            print(' > # of edges in overlap: ', len(edge_overlap))
            print(' > # of safe edges in decagon: ', len(decagon_edge_overlap))
            print(' > # of safe edges in decagon: ', len(decagon_edge_overlap2))

