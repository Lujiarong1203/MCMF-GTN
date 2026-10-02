# MCMF-GTN

Modular reference implementation of multi-channel adjacency construction, biased temporal random walks, feature fusion and sparse meta-path graph learning for cryptocurrency transaction-node classification.

This release provides the core pipeline, conventional matched comparisons, component variants and evaluation tools. It is not the historical experiment repository. Synthetic smoke tests check execution, gradients and checkpoint reloads; they do not reproduce the numerical results in the manuscript. Raw datasets and trained research checkpoints are not distributed.

## Installation

Python 3.9 or newer is required. Install a PyTorch build appropriate for the machine first, then:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m mcmf_gtn --help
```

The tested runtime is Python 3.11, NumPy 2.4 and PyTorch 2.5.1, with CPU and CUDA 12.4. Other compatible dependency versions are not asserted to have been tested.

## Pipeline

1. Read node IDs, labels, event times and benchmark features; preserve real transaction amounts and repeated events.
2. Remove invalid ETX zero-amount events and isolated/low-activity nodes. Select all eligible illicit centers and an equal number of licit centers without replacement.
3. Apply the temporal protocol. Elliptic and RD use snapshots 1-34 for training, 35-44 for validation and 45-49 for testing, with 10% of training centers moved to validation. ETX sorts all eligible accounts by first event time and node ID to define 70/10/20 membership, then applies the selected target mask.
4. Build separate one-hop predecessor/successor induced subgraphs within the event window of each split. Only selected labeled centers contribute to the loss or reported scores.
5. Fit feature imputation/scaling, timestamp normalization, amount normalization and Skip-Gram on training data only. Generate strictly time-increasing directed walks. ETX combines time proximity and amount; the Bitcoin datasets use time proximity alone.
6. Concatenate static features with the walk embeddings. Form row-normalized channel-weighted adjacency factors with self-loops, multiply the factors in order and retain the largest K stored entries per row after each product. Retained values keep their gradients. No dense N-by-N fallback is used.
7. Train with validation-only checkpoint selection. Freeze the selected checkpoint, score the test centers and export predictions, confusion counts and metrics.

Edges are represented as source-row/target-column entries; propagation aggregates successor features. Pruning does not renormalize the surviving rows. A sparse-product expansion guard reports excessive temporary work rather than silently changing the model.

## Reported settings

| Setting | Elliptic | ETX Network | RD Bitcoin |
|---|---:|---:|---:|
| Static feature dimension | 166 | 14 | 185 |
| Edge channels | topology, timestamp | topology, timestamp, amount | topology, timestamp |
| BTRW dimension | 128 | 128 | 128 |
| Fused dimension | 294 | 142 | 313 |

The main configuration uses three GT factors, GCN dimensions 128-64-32, BatchNorm, ReLU, dropout 0.5, Adam, learning rate 0.0001, learning-rate decay by 0.1 every 10 epochs, 100 epochs, labeled-center batches of 128 and seeds 42, 123, 456, 789 and 1024. The ETX transition coefficient is 0.6. Checkpoint selection uses the highest validation positive-class F1, keeping the first tie.

`configs/paper.json` contains the reported settings. The exact K, walk length/count, Skip-Gram settings, activity cutoff, feature scaling, weight decay and all conventional-baseline details are not fully reported in the manuscript. Their reference defaults and provenance are exposed by:

```bash
python -m mcmf_gtn config --dataset etx
```

Reference defaults include K=10, walk length=80, 10 walks per start, window=10, 5 negative samples and 5 Skip-Gram epochs. They must not be described as confirmed historical settings. Formal training requires an author configuration with confirmed values or the explicit `--accept-implementation-defaults` acknowledgement. The fixed decision rule is a positive probability of at least 0.5. Thresholds are never chosen using test labels.

### Unseen-node embeddings

A token-based Skip-Gram model cannot directly look up a node absent from its training vocabulary. The manuscript does not specify this case. The default `inference_policy: "strict"` therefore stops rather than filling a missing embedding with zeros or random values.

An optional `"frozen_context"` reference policy estimates new node vectors against frozen training context vectors, using unlabeled evaluation walks. It changes neither the training embeddings nor the classifier, but requires reachable training-context anchors and stops when they are absent. This is an implementation choice, not a claimed reconstruction of the original procedure. In particular, independent Bitcoin snapshots may provide no such anchors. Confirm the actual experiment policy before claiming real-data reproduction. The synthetic smoke graphs contain explicit shared anchors to test the executable inference path.

Normalization uses training-fitted statistics without silently clipping later values. Future timestamp and amount weights can exceed the observed training range; nonnegative values remain required. ETX's fourteen reference statistics are, in order: in-degree, out-degree, incoming amount, outgoing amount, incoming event count, outgoing event count, maximum incoming amount, maximum outgoing amount, minimum incoming amount, minimum outgoing amount, mean incoming amount, mean outgoing amount, last event time and activity span. These definitions and feature scaling should be checked against the author's original run configuration.

## Commands

Keep data outside the repository; `data/SOURCES.txt` lists original sources and accepted schemas. Pickle input is disabled unless `--trusted-pickle` is explicitly supplied for a trusted file.

```bash
python -m mcmf_gtn prepare --dataset elliptic --data /path/to/elliptic --out outputs/preparation
python -m mcmf_gtn train --config configs/paper.json --dataset etx --data /path/to/etx --out outputs/main --accept-implementation-defaults
python -m mcmf_gtn evaluate --checkpoint outputs/main/etx/full/seed_42/checkpoint.pt --data /path/to/etx --out outputs/test
```

The training command runs five seeds by default; `--seeds 42` selects one. Put an author-confirmed inference policy and any explicitly chosen implementation settings in the JSON configuration. A failed strict unseen-node check is an intentional safeguard, not permission to refit embeddings on the test set.

### Matched comparisons and component variants

| Variant | Difference from the full construction |
|---|---|
| `mlp` | Fused node inputs, no graph propagation |
| `gat` | Conventional topology attention baseline |
| `topology_gcn` | Unweighted connectivity with the same GCN backbone |
| `fixed_mean` | Equal-weight channel fusion, no meta-path products |
| `direct_ef` | Small learned edge-feature mapping, no GT block |
| `edge_weight` | Learned scalar fusion without adjacency products |
| `topology_gtn` | Topology channel only; retain meta-path products |
| `no_amount` | Remove the amount adjacency channel; ETX only |
| `no_timestamp` | Remove the timestamp adjacency channel |
| `no_embedding` | Static features only |
| `embedding_only` | Walk embeddings only |
| `uniform_trw` | Remove transition bias; retain temporal eligibility |
| `fixed_product` | Equal channel coefficients; retain products |
| `full` | Full MCMF-GTN |

Channel-removal variants change adjacency inputs only; the walk branch stays unchanged. This isolates adjacency channels from transition bias. `no_embedding` removes the complete BTRW branch. `uniform_trw` separately removes the bias. The conventional graph variants use the same prepared targets, fused inputs and three-layer GCN backbone except GAT, which has its own attention architecture. Direct-EF uses a 16-unit hidden mapping. The edge-weight variant averages the per-factor fused adjacencies with learned scalar factor coefficients.

```bash
python -m mcmf_gtn train --config configs/paper.json --dataset etx --data /path/to/etx --variant no_amount --out outputs/no_amount --accept-implementation-defaults
```

Contemporary external-model comparisons require the corresponding authors' implementations and controlled adaptation. TGTOD, Polynormer and DUALFormer are not silently replaced with conventional layers in this release. Their historical reproduction scripts, bulk scheduling and unpublished tuning records are outside the public core package.

### Imbalance and robustness

For imbalance evaluation, first score the complete eligible labeled test pool once with each frozen checkpoint. Then sample targets from those prediction files, without rebuilding the context graph, retraining or changing predictions:

```bash
python -m mcmf_gtn evaluate --checkpoint outputs/main/etx/full/seed_42/checkpoint.pt --data /path/to/etx --all-test-targets --out outputs/pool_42
python -m mcmf_gtn imbalance --predictions outputs/pool_42/predictions.csv --ratios 1 5 10 100 1000 --positive-count 300 --out outputs/imbalance
python -m mcmf_gtn robustness --checkpoint outputs/main/etx/full/seed_42/checkpoint.pt --data /path/to/etx --out outputs/robustness
```

Use all five seed prediction files for across-model-seed uncertainty. Positive targets are fixed and negative subsets are nested and shared across files by sampling seed. Pools must have the same IDs and labels. An infeasible ratio stops with the required and available counts; there is no oversampling. A controlled 1:1000 test is not described as a full natural 2551:1 evaluation. Keeping positive predictions fixed also fixes recall; F1 and PR-AUC can still change with prevalence.

Robustness applies 10%/30% test-edge dropout or Gaussian noise with sigma 0.1/0.3 to training-transformed static inputs. All adjacency channels use the same retained events, and the GT products are reconstructed. Walk embeddings remain fixed in this reference protocol; the checkpoint is never retrained. Output metadata records this scope. Use a distinct output directory for each run.

Metrics include positive-class precision, recall and F1, accuracy, ROC-AUC, PR-AUC (non-interpolated average precision), MCC, balanced accuracy, G-Mean, FPR and confusion counts. Undefined ranking metrics are null. Across-seed standard deviations use ddof=1. All machine-readable scores are fractions; multiply by 100 only for percentage presentation.

## Smoke tests

```bash
python -m pytest -q
python -m mcmf_gtn smoke --dataset all --all-variants --out outputs/smoke
python -m mcmf_gtn smoke --dataset all --seeds 42 --device cuda --out outputs/smoke_cuda
```

In environments with unrelated pytest plugins, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before running the tests. Smoke mode uses in-memory synthetic graphs and deliberately reduced dimensions and epochs. It never downloads source data or targets manuscript scores. Each smoke run trains, saves and restores a checkpoint and verifies prediction parity. Run configuration, split IDs, fitted transforms and data fingerprints are saved alongside each checkpoint.

## Layout

```text
configs/         reported and synthetic-smoke configurations
data/SOURCES.txt data origins and format requirements
mcmf_gtn/data/   readers and temporal subgraph protocol
mcmf_gtn/features/ normalization, walks and Skip-Gram
mcmf_gtn/models/ sparse meta-path construction and classifiers
mcmf_gtn/        training, evaluation, metrics and command line
tests/           unit and end-to-end checks
```

The code retains the existing project's MIT license. Dataset permissions remain governed by the original providers.
