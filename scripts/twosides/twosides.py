r""" Replace drug ids in twosides database with corresponding drugbank ids.
"""
import pandas as pd
import numpy as np

TWOSIDES = '../../data/twosides/TWOSIDES.csv'
TWOSIDES_TO_DB = '../../data/twosides/rxnorm-drugbank-omop-mapping-CLEANED.tsv'

NEW_TWOSIDES = '../../data/twosides/TWOSIDES_DB.csv'


def main():
    with open(TWOSIDES, 'r') as twosides:
        with open(TWOSIDES_TO_DB, 'r') as twosides_to_db:
            twosides_lines = twosides.readlines()[1:]
            two_db_map_lines = twosides_to_db.readlines()[1:]

            # get only the needed columns from twosides
            twosides_split = [a.strip().split(',') for a in twosides_lines]
            twosides_clean = [(a[0], a[2], a[4]) for a in twosides_split]

            for i in range(5):
                print(twosides_clean[5])
            # make a mapping from rx to db
            two_db_map_split = [
                a.strip().split('\t') for a in two_db_map_lines
            ]

            for i in range(5):
                print(two_db_map_split[i])

            two_db_map = dict([(a[0], a[2]) for a in two_db_map_split])

            new_twosides = open(NEW_TWOSIDES, 'w+')
            new_twosides.write('d1,d2,rel\n')
            for d1, d2, rel in twosides_clean:
                new_d1, new_d2 = d1, d2
                if d1 in two_db_map:
                    new_d1 = two_db_map[d1]
                if d2 in two_db_map:
                    new_d2 = two_db_map[d2]
                new_twosides.write(','.join([new_d1, new_d2, rel]) + '\n')

            new_twosides.close()


if __name__ == '__main__':
    main()
