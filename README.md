# MCMF-GTN

Multi-Channel Multi-Feature Graph Transformer Network for anomalous account
detection in cryptocurrency transaction networks.

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.5, PyTorch Geometric 2.8.

## Repository layout

```
mcmf_gtn/
  config.py              hyperparameters
  data/                  dataset loaders (Elliptic, ETX Network, RD Bitcoin)
  features/
    subgraph.py          balanced target sampling + 1-hop subgraph
    btrw.py              biased temporal random walk + Skip-Gram
    btrw_embeddings.py   walk-based node embeddings
    to_pyg_data.py       PyG Data objects, sparse adjacency channels
  models/
    gtn.py               channel weighting, meta-path product, top-K pruning
    mcmf_gtn.py          MCMF-GTN and MCMF-GCN
    ef_gcn.py            fixed-mean / direct / learned edge-feature GCN
    gcn_classifier.py    GCN backbone
    gat_baseline.py      GAT baseline
  train_eval/metrics.py  evaluation metrics
scripts/
  train.py               training entry point
  evaluate.py            evaluate a saved checkpoint
  smoke_test.py          synthetic-data end-to-end check
baselines/               recent-baseline adapters (see baselines/README.md)
data/README.txt          dataset sources and expected file layout
```

## Training protocol

All models are trained for 100 epochs with Adam (learning rate 1e-4, no
decay), batch size 128, and gradient clipping at 1.0. The transaction
subgraph is built once per dataset by balanced target sampling over the
complete graph; the biased random walks and the Skip-Gram training span the
whole subgraph, so every subgraph node receives an embedding. The temporal
split governs supervision, model selection, and evaluation: the checkpoint
with the best validation F1 among the evaluation snapshots (epochs 50 and
100) is restored, the decision threshold maximizes validation F1 over
0.10-0.95 in steps of 0.05, and all metrics are computed on the test split.

## Usage

```bash
# main model (meta-path matrix product with per-row top-K pruning)
python scripts/train.py --dataset elliptic --variant MCMF-GTN

# controlled configurations
python scripts/train.py --dataset elliptic --variant GCN          # (a)
python scripts/train.py --dataset elliptic --variant EFGCN-FIXED  # (b)
python scripts/train.py --dataset elliptic --variant EFGCN-DIRECT # (c)
python scripts/train.py --dataset elliptic --variant MCMF-GCN     # (d)
python scripts/train.py --dataset elliptic --variant MLP
python scripts/train.py --dataset elliptic --variant GAT

# other datasets (see data/README.txt for data placement)
python scripts/train.py --dataset etx --variant MCMF-GTN
python scripts/train.py --dataset rd_bitcoin --variant MCMF-GTN

# evaluate a saved checkpoint
python scripts/evaluate.py --dataset elliptic --variant MCMF-GTN \
    --checkpoint outputs/elliptic_MCMF-GTN_seed42_best.pt

# end-to-end check on synthetic data (no dataset required)
python scripts/smoke_test.py
```

Results (per-seed metrics and summaries) are written to `outputs/`.

## Datasets

The datasets are not redistributed with this repository; see
`data/README.txt` for the sources and the expected file layout.
