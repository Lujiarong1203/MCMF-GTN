"""Read canonical CSVs or explicitly supported native dataset formats."""

import csv
import pickle
from pathlib import Path

import numpy as np

from mcmf_gtn.config import dataset_spec
from mcmf_gtn.data.types import GraphData


def _csv(path, header=True):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    names = rows.pop(0) if header else None
    width = len(names) if header else len(rows[0])
    if names is not None and len(set(names)) != len(names):
        raise ValueError(f"duplicate CSV columns: {path}")
    if any(len(row) != width for row in rows):
        raise ValueError(f"malformed CSV row width: {path}")
    return names, rows


def _require(names, required, path):
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)} in {path}")


def _ids(values):
    result = np.asarray([str(value).strip() for value in values], dtype=str)
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError("node IDs must be nonempty and unique; duplicate ID found")
    return result


def _numbers(values, description, missing=False):
    try:
        result = np.asarray([float(x) if str(x).strip() else np.nan for x in values], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"nonnumeric {description}") from exc
    if np.isinf(result).any() or (not missing and not np.isfinite(result).all()):
        raise ValueError(f"{description} must be finite")
    return result


def _label(value, native=False):
    value = str(value).strip().lower()
    mapping = {"-1": -1, "0": 0, "1": 1}
    if native:
        mapping = {"1": 1, "2": 0, "unknown": -1, "3": -1, "-1": -1, "illicit": 1, "licit": 0}
    if value not in mapping:
        raise ValueError(f"invalid label: {value!r}")
    return mapping[value]


def _endpoints(node_ids, rows, source_column, target_column):
    lookup = {node_id: i for i, node_id in enumerate(node_ids)}
    result = np.empty((2, len(rows)), dtype=np.int64)
    for i, row in enumerate(rows):
        for side, column in enumerate((source_column, target_column)):
            key = str(row[column]).strip()
            if key not in lookup:
                raise ValueError(f"unknown edge endpoint: {key!r}")
            result[side, i] = lookup[key]
    return result


def _first_times(edge_index, edge_times, fallback):
    result = np.full(len(fallback), np.inf, dtype=np.float64)
    for side in edge_index:
        np.minimum.at(result, side, edge_times)
    return np.where(np.isfinite(result), result, fallback)


def _canonical(name, directory):
    node_path, edge_path = directory / "nodes.csv", directory / "edges.csv"
    names, nodes = _csv(node_path)
    dim = dataset_spec(name).static_dim
    features = [f"f{i}" for i in range(dim)]
    _require(names, ["node_id", "label", "timestamp", *features], node_path)
    found = {key for key in names if key.startswith("f") and key[1:].isdigit()}
    if found != set(features):
        raise ValueError(f"{name} requires exactly {dim} static feature columns")
    if not nodes:
        raise ValueError("nodes.csv contains no nodes")
    node_ids = _ids([row[names.index("node_id")] for row in nodes])
    labels = np.array([_label(row[names.index("label")]) for row in nodes], dtype=np.int64)
    times = _numbers([row[names.index("timestamp")] for row in nodes], "node timestamps")
    x = np.column_stack([_numbers([row[names.index(key)] for row in nodes], f"feature {key}", missing=True) for key in features])
    edge_names, edges = _csv(edge_path)
    _require(edge_names, ["source", "target", "timestamp"], edge_path)
    edge_index = _endpoints(node_ids, edges, edge_names.index("source"), edge_names.index("target"))
    edge_times = _numbers([row[edge_names.index("timestamp")] for row in edges], "edge timestamps")
    amounts = None
    if name == "etx":
        _require(edge_names, ["amount"], edge_path)
        amounts = _numbers([row[edge_names.index("amount")] for row in edges], "edge amounts")
        times = _first_times(edge_index, edge_times, times)
    return GraphData(node_ids, x, labels, times, edge_index, edge_times, amounts)


def _native_bitcoin(name, path):
    directory = path.parent if path.is_file() else path
    if name == "elliptic":
        feature_path = path if path.is_file() else directory / "elliptic_txs_features.csv"
        _, rows = _csv(feature_path, header=False)
        time_column, id_column = 1, 0
        class_path = directory / "elliptic_txs_classes.csv"
        edge_path = directory / "elliptic_txs_edgelist.csv"
    else:
        if path.is_file():
            feature_path = path
        else:
            matches = sorted(directory.glob("rd_features*.csv"))
            if len(matches) != 1:
                raise ValueError("ambiguous RD feature files; supply the explicit chosen feature CSV path")
            feature_path = matches[0]
        names, rows = _csv(feature_path)
        _require(names, ["txId"], feature_path)
        time_names = [key for key in ("time_stamp", "Time step", "time_step", "timestamp") if key in names]
        if len(time_names) != 1:
            raise ValueError("RD features require one explicit timestamp column (time_stamp)")
        time_column, id_column = names.index(time_names[0]), names.index("txId")
        class_path, edge_path = directory / "rd_classes.csv", directory / "rd_edgelists.csv"
    dim = dataset_spec(name).static_dim
    if not rows or any(len(row) != dim + 1 for row in rows):
        raise ValueError(f"{name} requires exactly {dim} features including time, plus one transaction ID")
    node_ids = _ids([row[id_column] for row in rows])
    times = _numbers([row[time_column] for row in rows], "node timestamps")
    # Keep time as feature zero; transaction identifiers never enter the feature matrix.
    columns = [time_column] + [i for i in range(len(rows[0])) if i not in (time_column, id_column)]
    x = np.column_stack([_numbers([row[i] for row in rows], "node features", missing=True) for i in columns])
    class_names, class_rows = _csv(class_path)
    _require(class_names, ["txId", "class"], class_path)
    class_ids = _ids([row[class_names.index("txId")] for row in class_rows])
    mapping = {node_id: _label(row[class_names.index("class")], native=True) for node_id, row in zip(class_ids, class_rows)}
    labels = np.array([mapping.get(node_id, -1) for node_id in node_ids], dtype=np.int64)
    edge_names, edge_rows = _csv(edge_path)
    _require(edge_names, ["txId1", "txId2"], edge_path)
    edge_index = _endpoints(node_ids, edge_rows, edge_names.index("txId1"), edge_names.index("txId2"))
    return GraphData(node_ids, x, labels, times, edge_index, times[edge_index[0]].copy())


def _native_etx(path, trusted_pickle):
    if trusted_pickle is not True:
        raise ValueError("ETX pickle requires explicit trusted_pickle=True; pickle can execute code")
    if path.is_dir():
        matches = sorted([*path.glob("*.pickle"), *path.glob("*.pkl")])
        if len(matches) != 1:
            raise ValueError("supply the explicit ETX pickle path")
        path = matches[0]
    import networkx as nx

    with path.open("rb") as stream:
        network = pickle.load(stream)
    if not isinstance(network, nx.Graph) or not network.is_directed():
        raise ValueError("ETX pickle must contain a directed NetworkX graph")
    original_ids = [node for node in network.nodes if network.degree(node) > 0]
    node_ids = _ids(original_ids)
    lookup = {node: i for i, node in enumerate(original_ids)}
    labels = []
    for node in original_ids:
        attr = network.nodes[node]
        value = attr.get("label")
        if value is None:
            value = attr.get("phishing", -1)
        labels.append(-1 if value is None else _label(int(value) if isinstance(value, (bool, np.bool_)) else value))
    endpoints, event_times, amounts = [], [], []
    for source, target, attr in network.edges(data=True):
        timestamp, amount = attr.get("timestamp"), attr.get("amount")
        if timestamp is None:
            timestamp = attr.get("time")
        if amount is None:
            amount = attr.get("value")
        if timestamp is None or amount is None:
            raise ValueError("ETX edges require timestamp/time and amount/value; missing values cannot be fabricated")
        endpoints.append((lookup[source], lookup[target]))
        event_times.append(timestamp)
        amounts.append(amount)
    edge_index = np.asarray(endpoints, dtype=np.int64).reshape(-1, 2).T
    edge_times = _numbers(event_times, "edge timestamps")
    edge_amounts = _numbers(amounts, "edge amounts")
    times = _first_times(edge_index, edge_times, np.full(len(node_ids), np.inf))
    return GraphData(node_ids, np.zeros((len(node_ids), 14)), np.asarray(labels, dtype=np.int64), times, edge_index, edge_times, edge_amounts)


def load_dataset(name, path, trusted_pickle=False):
    """Load data without changing source files. Native RD accepts a feature file path."""
    dataset_spec(name)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir() and (path / "nodes.csv").exists():
        raw = _canonical(name, path)
    elif name == "etx":
        raw = _native_etx(path, trusted_pickle)
    else:
        raw = _native_bitcoin(name, path)
    raw.validate()
    return raw
