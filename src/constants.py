DATA_DECAGON = "decagon"
DATA_DEEPDDI = "deepddi"
DATA_DEEPDDI_NE = 'deepddi-ne'
DATA_MIXTURES = "mixtures"

DATA = [DATA_DECAGON, DATA_DEEPDDI, DATA_MIXTURES]

DECAGON_ORIG_RAW = "./data/decagon/bio-decagon-combo.csv"
DECAGON_RAW = "./data/decagon/bio-decagon-combo2.csv"
DEEPDDI_RAW = "./data/deepddi/KnownDDI.csv"
DEEPDDI_RAW_NE = './data/deepddi-me/KnownDDI.csv'

PK_PREFIX = "ind"
PK_SUFFIX_KG = "graph"
PK_SUFFIX_X = "allx"

DRUGBANK_N2ID = "./data/db_n2id.txt"
# DRUGBANK_ID2PCID = "./data/db_id2pcid.txt"
DRUGBANK_PCID2ID = "./data/db_pcid2id.txt"
DRUGBANK_MIXTURES = "./data/mixtures/db_mixtures.txt"

DATA2RAW = {
    DATA_DECAGON: DECAGON_RAW,
    DATA_DEEPDDI: DEEPDDI_RAW,
    DATA_MIXTURES: DRUGBANK_MIXTURES
}

DIR_RESULTS = './results/'

M_DEDICOM = "dedicom"
M_DISTMULT = "distmult"
M_MLP = "mlp"
M_RGCN = "rgcn"
M_MPNN = "mpnn"
M_DECAGON = "decagon"
MODELS = [M_DEDICOM, M_DISTMULT, M_MLP, M_RGCN, M_MPNN, M_DECAGON]

DATASETS = {
     DATA_DEEPDDI: (DEEPDDI_RAW, ["Drug1","Drug2","Label"], 200),  #, DEEPDDI_SMILES),
     DATA_DECAGON: (DECAGON_ORIG_RAW, ["STITCH 1","STITCH 2","Polypharmacy Side Effect"], 500),  #, DECAGON_SMILES)
     DATA_DEEPDDI_NE: (DEEPDDI_RAW_NE, ['Drug1', 'Drug2', 'Label'], 200)
}
