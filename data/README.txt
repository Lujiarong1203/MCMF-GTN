Datasets
========

The datasets used in the paper are not distributed with this repository.
Place each dataset in the subfolders described below before running the
scripts. The default data directory is ./data (see mcmf_gtn/config.py).

1. Elliptic (public)
   ------------------
   The Elliptic Bitcoin transaction dataset is publicly available on
   Kaggle:
     https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

   Download the three CSV files and place them under:
     data/Elliptic/
       elliptic_txs_features.csv
       elliptic_txs_edgelist.csv
       elliptic_txs_classes.csv

2. ETX Network (third-party)
   -------------------------
   The ETX Network is an Ethereum transaction graph collected from
   Etherscan. Access to the raw graph is provided by the authors of the
   original study upon reasonable request. The loader expects a NetworkX
   MultiDiGraph pickle under:
     data/ETX Network.pickle

3. RD Bitcoin (third-party)
   ------------------------
   RD Bitcoin is an extended version of the Elliptic dataset introduced by
   Wang et al. (2024). Access to the raw files is provided by the authors
   of the original study upon reasonable request. The loader expects:
     data/RD bitcoin/rd_features0.1.csv
     data/RD bitcoin/rd_edgelists.csv
     data/RD bitcoin/rd_classes.csv
