# DDIME

**D**rug&ndash;**D**rug **I**nteraction **M**ultiple **E**vidence

This repository contains the source code for "Relation Dependent Sampling for Multi-Relational Link Prediction". We propose the Relation-Sampling Graph Convolutional Network or RS-GCN. It uses REINFORCE to learn sampling probabilities for each edge type. This model of graph sampling improves the memory use and compute time of RGCNs while also improving accuracy compared to standard random sampling.

Our experiments focus on Drug-Drug Interaction (DDI) detection: the prediction of unknown interactions or adverse side effects caused by taking multiple prescription drugs. These interactions can have large effects on patient mortality and morbidity, so knowledge of them can be crucial for drug development and the safe prescription of multiple drugs.

The code for our experiments is in ``src/main.py``. The baseline and proposed sampling methods are in ``src/model_sampling/``. The propsed sampling method is implemented across ``src/model_sampling/relational_sampler.py`` and ``src/train_eval.py``. Each sampling method uses the same underlying implementation, but overrides the function that computes sampling logits. The sampling implementation used by each sampler is ``src/model_sampling/sampling_impl.py``. We add negative evidence in ``src/batchify.py``. The classification report (evalution metrics) is in ``src/utils.py``.


## Experiments

We provide bash scripts in ``scripts/`` to recreate our DDI detection experiments. Scripts for DeepDDI and TWOSIDES are in ``scripts/experiment/deepddi/`` and ``scripts/experiment/TWOSIDES/`` respectively. Each experiment produces PNGs showing sampling statistics for the last completed evaluation. An example run of our proposed method may be

~~~~
python src/main.py \
    --seed=1 \
    --ne_train=0 \  # do not use negative evidence in place of negative samples
    --ne_valid=0 \
    --new_edge=0 \  # do not use negative evidence as an edge in the input graph
    --data_dir=./data/deepddi-me \  # use DeepDDI
    --data_ratio=100 \
    --batch_size=2000 \
    --train_size=60 \
    --valid_size=20 \
    --model_name=decagon \
    --lr=.001 \
    --hidden_dim=128 \
    --epoch=300 \
    --sample \
    --sampling_name=gumbel_sampler \  # use proposed sampling method
~~~~

For each of the Relational graph models, we use the same basic settings. We only apply changes to the sampling method and negative evidence in different experiemnts.


## Setup and Installing Requirements

To run our experiments, you should clone this repository.

~~~~
git clone https://github.com/CognitiveHorizons/ddime.git
~~~~

We use PyTorch and PyTorch Geometric. The installation of these varies by the target device. These packages are perhaps best installed by following their respective documentation:

~~~~
https://pytorch.org/
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
~~~~

The remaining requirements that were not installed as prerequisites for torch and torch geometric can be installed using pip ``pip install -r requirements.txt``. We recommend creating a virtual environment, but it should be safe to use pip within that environment.


## Preprocessing

We use two DDI datasets in our experiments: DeepDDI and a subset of TWOSIDES (TWOSIDES100). We use DrugBank mixture products and synergism data from DrugCombDB for negative evidence. Versions of these used in our experiments are included in the repository, _except for TWOSIDES_.


#### TWOSIDES

This step _must_ be done. It is relatively straight-forward to download TWOSIDES. Again, it is too large to include in the repo. You can run the script ``scripts/twosides/download_twosides.sh``.


#### DeepDDI

The DeepDDI dataset is already included in the repo: ``data/deepddi/KnownDDI.csv``.


#### Extract Mixtures (Optional)

This is optional because we include the needed subsets of the mixture products used in our experiments.

We use mixture products from DrugBank as part of our negative evidence (``https://www.drugbank.ca/releases/latest``). To extract the mixture products you need the drugbank full XML file with the path to it set in src/preprocessing/drugbank.py. The drugbank xml file can be downloaded into the ``data/`` directory. It is a bit large to include in the repo, so if wanted, we recommend downloading it in the data directory. Once it is downloaded, run the preprocessing scripts.

~~~~
python src/preprocess/data.py
python src/preprocess/drugbank.py
~~~~

#### Synergism

We include the synergism data and DeepDDI negative evidence datasets for use with TWOSIDES in ``data/TWOSIDES``. The synergism data is from DrugCombDB.
