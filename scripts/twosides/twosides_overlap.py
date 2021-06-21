r""" Check how many nodes overlap between TWOSIDES and the safe edge db.
"""

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

NEW_TWOSIDES = '../../data/twosides/TWOSIDES_DB.csv'


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


def rel_freq(combos):
    counts = {}
    for _, _, rel in combos:
        if rel in counts:
            counts[rel] += 1
        else:
            counts[rel] = 1
    return counts


def get_freq(combos, lower=500, upper=3000):
    r""" Get combos that have a reaction type which occurs more than thresh
    times.
    """
    freqs = rel_freq(combos)
    freq_combos = []  # track frequent combos
    for combo in combos:
        if freqs[combo[2]] > lower and freqs[combo[2]] < upper:
            freq_combos.append(combo)
    return freq_combos


def count_rel_types(combos):
    seen = set([])
    counter = 0
    for combo in combos:
        if combo[2] not in seen:
            seen.add(combo[2])
            counter += 1
    return counter


def count_combos(combos):
    seen = set([])
    counter = 0
    for combo in combos:
        drugs = (combo[0], combo[1])
        if drugs not in seen:
            seen.add(drugs)
            counter += 1
    return counter


if __name__ == "__main__":
    print('Size of intersection between DeepDDI and DrugBank mixtures')
    with open(DEEPDDI_RAW_NE, 'r') as deepddi:
        lines = deepddi.readlines()[1:]  # remove header

        print('Numer of combos: ', len(lines))

        split_lines = [a.split(',') for a in lines]
        safe_id = max([int(a[2]) for a in split_lines])
        good_combos = [a for a in split_lines if int(a[2]) == safe_id]

        # drugs that appear in safe pairs.
        good_cols = tuple(zip(*[(a[0], a[1]) for a in good_combos]))
        good_drugs = set(good_cols[0] + good_cols[1])

        twosides = open(NEW_TWOSIDES, 'r')
        twosides_lines = twosides.readlines()[1:]
        twosides_split = [a.strip().split(',') for a in twosides_lines]

        print('# of TWOSIDES Combos: ', len(twosides_split))

        twosides_split = get_freq(twosides_split)

        print('# of TWOSIDES FREQ Combos: ', len(twosides_split))

        for i in range(5):
            print(twosides_split[i])

        rel_types = count_rel_types(twosides_split)
        combo_types = count_combos(twosides_split)
        print('# of relations: ', rel_types)
        print('# of collapsedEdges: ', combo_types)

        two_cols = tuple(zip(*[(a[0], a[1]) for a in twosides_split]))
        two_drugs = set(two_cols[0] + two_cols[1])

        intersect = set(list(two_drugs & good_drugs))

        # edges that have an endpoint with a drug in the overlap
        overlap_edges = [
            edge for edge in good_combos
            if edge[0] in intersect or edge[1] in intersect
        ]
        overlap_edges2 = [
            edge for edge in good_combos
            if edge[0] in intersect and edge[1] in intersect
        ]

        print('*** Results ***')
        print(' > # of "bad" drugs: ', len(two_drugs))
        print(' > # of "good" drugs: ', len(good_drugs))

        print(' > # of drugs in total: ', len(two_drugs) + len(good_drugs))
        print(' > Drugs in intersect: ', len(intersect))
        print(' > Edges in overlap: ', len(overlap_edges))
        print(' > Edges in overlap: ', len(overlap_edges2))

        twosides.close()
