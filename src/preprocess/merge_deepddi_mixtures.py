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


if __name__ == "__main__":
    print("Merge DeepDDI with drugbank mixtures")
    with open(DEEPDDI_RAW, 'r') as deepddi:
        with open(DRUGBANK_MIXTURES, 'r') as mixtures:
            deepddi_lines = deepddi.readlines()
            header = deepddi_lines[0]
            # get the deepddi data lines and keep the header.
            deepddi_lines = [a.split(',') for a in deepddi_lines[1:]]

            # get mixture lines and remove the header
            mixture_lines = [a.split(',') for a in mixtures.readlines()[1:]]
            print('Loaded data')

            # create a label for the safe ddi edges.
            max_rel = max([int(a[2]) for a in deepddi_lines])
            safe_rel = str(max_rel + 1)

            # create safe lines using the new label
            safe_lines = [(a[:2] + [safe_rel]) for a in mixture_lines]

            print('Writing data')
            with open(DEEPDDI_RAW_NE, 'w+') as deepddi_me:
                # write the header
                deepddi_me.write(header)
                # write the two datasets
                for line in deepddi_lines:
                    deepddi_me.write(','.join(line))
                for line in safe_lines:
                    deepddi_me.write(','.join(line) + '\n')
    print('Finished Merge')


