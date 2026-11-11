# Recent baseline adapters

Adapters that train TGTOD, Polynormer, and DUALFormer under the shared
protocol used for the main comparison (identical temporal splits, the five
seeds 42/123/456/789/1024, normalization statistics, subgraph construction,
test sets, 100 epochs, batch size 128, learning rate 1e-4).

## Included implementations

- `tgtod/` — official implementation of TGTOD (Liu et al., PAKDD 2025),
  license file kept
- `polynormer/` — official implementation of Polynormer (Deng et al., ICLR
  2024), license file kept
- `dualformer/model.py` — implementation of DUALFormer (Zhuo et al., ICLR
  2025) following the paper's description

Each adapter in `adapters/` exposes a single entry point, `build_model()`,
that returns a model callable as `forward(x, edge_index)` or `forward(x,
edge_index, edge_attr)`. The adapter file is the only place that depends on
the baseline's code; the shared harness in `run_baseline.py` handles data
loading, training, and evaluation under the protocol above.

The baselines receive the dataset transaction features without the BTRW
embeddings, which are specific to MCMF-GTN.

## Dependencies

TGTOD additionally requires torch-sparse and torch-scatter:

```
pip install torch-sparse torch-scatter \
    -f https://data.pyg.org/whl/torch-<torch-version>.html
```

## Usage

```bash
python baselines/run_baseline.py --dataset elliptic --baseline tgtod
python baselines/run_baseline.py --dataset etx --baseline polynormer
python baselines/run_baseline.py --dataset rd_bitcoin --baseline dualformer
```

Run `python scripts/smoke_test.py` to verify that every baseline trains on
synthetic data.
