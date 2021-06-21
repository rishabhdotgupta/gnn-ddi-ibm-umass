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
    r""" Remove potential repeated pairs or pairs that occur in both sets.
    """
    bad_cols = [(a[0], a[1], a[2]) for a in bad]
    good_cols = [(a[0], a[1], a[2]) for a in good]

    bad_cols = list(set(bad_cols))
    good_cols = set(good_cols)

    for t in bad_cols:
        if t in good_cols:
            good_cols.remove(t)
    return bad_cols, list(good_cols)


if __name__ == "__main__":
    print('Size of intersection between DeepDDI and DrugBank mixtures')
    with open(DEEPDDI_RAW_NE, 'r') as deepddi:
        lines = deepddi.readlines()[1:] # remove header

        print('Numer of combos: ', len(lines))

        split_lines = [a.split(',') for a in lines]
        safe_id = max([int(a[2]) for a in split_lines])
        print('No interaction label: ', safe_id)
        bad_combos = [a for a in split_lines if int(a[2]) < safe_id]
        good_combos = [a for a in split_lines if int(a[2]) == safe_id]

        # remove 'good' pairs that may appear in the bad pairs too.
        bad_combos, good_combos = clean(bad_combos, good_combos)

        # drugs that appear in the pairs that result in adverse effects.
        bad_cols = tuple(zip(*[(a[0], a[1]) for a in bad_combos]))
        bad_drugs = set(bad_cols[0] + bad_cols[1])

        # drugs that appear in safe pairs.
        good_cols = tuple(zip(*[(a[0], a[1]) for a in good_combos]))
        good_drugs = set(good_cols[0] + good_cols[1])

        intersect = set(list(bad_drugs & good_drugs))

        # edges that connect have an endpoint with a drug in the overlap
        overlap_edges = [edge for edge in good_combos if edge[0] in intersect or edge[1] in intersect]
        overlap_edges2 = [edge for edge in good_combos if edge[0] in intersect and edge[1] in intersect]


        print('*** Results ***')
        print(' > # of "bad" drugs: ', len(bad_drugs))
        print(' > # of "good" drugs: ', len(good_drugs))

        print(' > # of drugs in total: ', len(bad_drugs) + len(good_drugs))
        print(' > Drugs in intersect: ', len(intersect))
        print(' > Edges in overlap: ', len(overlap_edges))
        print(' > Edges in overlap: ', len(overlap_edges2))
