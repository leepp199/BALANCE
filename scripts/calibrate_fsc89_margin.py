#!/usr/bin/env python3
"""Leakage-safe FSC-89 base/novel margin calibration.

The pipeline has three explicit stages:

1. ``manifest`` binds a legacy 59-base checkpoint to its SHA-256 and class
   protocol.  This prevents accidentally extracting validation features with
   the normal 69-base checkpoint.
2. ``extract`` uses only that checkpoint's ``model.encode``.  Classes 0--58
   provide train-mean base prototypes and validation queries.  Classes 59--68
   provide disjoint train support pools and pseudo-novel validation queries.
   The FSC-89 test CSV is never opened; no row is selected and no waveform is
   opened or encoded for classes 69--88.
3. ``fit`` simulates 5-way and 10-way episodes with exactly five supports per
   pseudo-novel class.  It selects one interpretable threshold on
   ``max_novel_cosine - max_base_cosine``; no learned router is involved.

All artifacts are self-describing and hash-linked.  The script never downloads
models or data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = ROOT / ".offline_cache"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


SCHEMA_VERSION = 1
BASE_CLASSES = tuple(range(59))
PSEUDO_NOVEL_CLASSES = tuple(range(59, 69))
SEALED_TEST_CLASSES = tuple(range(69, 89))
TRAIN_CSV = "Fsc89-mini-fsci_train.csv"
VAL_CSV = "Fsc89-mini-fsci_val.csv"
REQUIRED_COLUMNS = {
    "label", "FSD_MIX_SED_filename", "start_time", "data_folder"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_torch(payload: Mapping, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), output)


def _save_json(payload: Mapping, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: Path) -> Mapping:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _class_range(text: str) -> List[int]:
    """Parse a comma/range class declaration such as ``0-58``."""
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            if hi < lo:
                raise argparse.ArgumentTypeError(f"descending class range: {part}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("class declaration contains duplicates")
    return sorted(values)


def create_manifest(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trained_classes = _class_range(args.trained_classes)
    if trained_classes != list(BASE_CLASSES):
        raise ValueError(
            "FSC-89 pseudo-unseen calibration requires a checkpoint trained "
            "on exactly zero-based classes 0--58"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fsc89_checkpoint_protocol_manifest",
        "dataset": "FSC-89",
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_filename": checkpoint.name,
        "training_classes_zero_based": trained_classes,
        "pseudo_novel_classes_zero_based": list(PSEUDO_NOVEL_CLASSES),
        "sealed_test_classes_zero_based": list(SEALED_TEST_CLASSES),
        "feature_api": "model.encode",
        "attestation": (
            "The supplied checkpoint was trained without labels or waveforms "
            "from classes 59--88."
        ),
    }
    output = Path(args.output).resolve()
    _save_json(payload, output)
    print(f"manifest={output} checkpoint_sha256={payload['checkpoint_sha256']}")


def verify_manifest(checkpoint: Path, manifest_path: Path) -> Mapping:
    manifest = _load_json(manifest_path)
    expected = {
        "artifact_type": "fsc89_checkpoint_protocol_manifest",
        "dataset": "FSC-89",
        "training_classes_zero_based": list(BASE_CLASSES),
        "pseudo_novel_classes_zero_based": list(PSEUDO_NOVEL_CLASSES),
        "sealed_test_classes_zero_based": list(SEALED_TEST_CLASSES),
        "feature_api": "model.encode",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"invalid checkpoint manifest field {key!r}")
    actual_hash = sha256_file(checkpoint)
    if manifest.get("checkpoint_sha256") != actual_hash:
        raise ValueError("checkpoint SHA-256 does not match its protocol manifest")
    return manifest


def _read_metadata(metadata_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read only train/val tables; deliberately has no test-table code path."""
    train_path = metadata_root / TRAIN_CSV
    val_path = metadata_root / VAL_CSV
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    for name, frame in (("train", train), ("val", val)):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{name} metadata is missing columns: {sorted(missing)}")
        frame["label"] = frame["label"].astype(int)
    return train, val


def _rows_for_classes(frame: pd.DataFrame, classes: Iterable[int]) -> pd.DataFrame:
    allowed = set(int(value) for value in classes)
    selected = frame[frame["label"].isin(allowed)].copy()
    selected["_source_row"] = selected.index.astype(int)
    selected.reset_index(drop=True, inplace=True)
    present = set(selected["label"].unique().tolist())
    missing = allowed - present
    if missing:
        raise ValueError(f"metadata has no rows for classes: {sorted(missing)}")
    if set(selected["label"].unique()) & set(SEALED_TEST_CLASSES):
        raise RuntimeError("sealed FSC-89 test classes entered calibration metadata")
    return selected


def build_protocol_rows(metadata_root: Path) -> Dict[str, pd.DataFrame]:
    train, val = _read_metadata(metadata_root)
    base_train = _rows_for_classes(train, BASE_CLASSES)
    base_query = _rows_for_classes(val, BASE_CLASSES)
    pseudo_all = _rows_for_classes(train, PSEUDO_NOVEL_CLASSES)
    support_parts, query_parts = [], []
    for class_id in PSEUDO_NOVEL_CLASSES:
        rows = pseudo_all[pseudo_all["label"] == class_id]
        split = int(len(rows) * 0.8)
        if split < 5 or len(rows) - split < 1:
            raise ValueError(
                f"class {class_id} cannot form a 5-shot support/query split"
            )
        support_parts.append(rows.iloc[:split])
        query_parts.append(rows.iloc[split:])
    pseudo_support = pd.concat(support_parts, ignore_index=True)
    pseudo_query = pd.concat(query_parts, ignore_index=True)
    support_ids = set(zip(pseudo_support["label"], pseudo_support["_source_row"]))
    query_ids = set(zip(pseudo_query["label"], pseudo_query["_source_row"]))
    if support_ids & query_ids:
        raise RuntimeError("pseudo-novel support/query metadata overlap")
    return {
        "base_train": base_train,
        "base_query": base_query,
        "pseudo_support": pseudo_support,
        "pseudo_query": pseudo_query,
    }


def protocol_counts(rows: Mapping[str, pd.DataFrame]) -> Dict[str, Dict[int, int]]:
    return {
        name: {
            int(class_id): int(count)
            for class_id, count in frame.groupby("label").size().items()
        }
        for name, frame in rows.items()
    }


class _WaveRows(Dataset):
    def __init__(self, frame: pd.DataFrame, dataroot: Path):
        self.labels = frame["label"].astype(int).tolist()
        self.paths: List[Path] = []
        for row in frame.itertuples(index=False):
            filename = row.FSD_MIX_SED_filename.replace(
                ".wav", f"_{int(float(row.start_time) * 44100)}.wav"
            )
            self.paths.append(
                dataroot / "audio" / str(row.data_folder) / filename
            )
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            preview = missing[:3]
            raise FileNotFoundError(f"missing FSC-89 audio files: {preview}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, _ = torchaudio.load(str(self.paths[index]))
        return waveform.squeeze(0), self.labels[index]


def _load_encoder(args: argparse.Namespace, checkpoint: Path) -> torch.nn.Module:
    from network import MYNET
    from train_unopenset import (
        _drop_retired_checkpoint_keys,
        args_parser,
        dict2namespace,
        set_seed,
    )

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["train"]
    defaults = vars(args_parser().parse_args(["--dataroot", str(args.dataroot)]))
    defaults.update(config)
    defaults.update({
        "dataset": "FMC",
        "dataroot": str(args.dataroot),
        "fsc89_metadata_root": str(args.metadata_root),
        "num_base": len(BASE_CLASSES),
        "num_labeled_classes": len(BASE_CLASSES),
        "train_classes": len(BASE_CLASSES),
    })
    model_args = dict2namespace(defaults)
    set_seed(args.seed)
    model = MYNET(model_args, mode="extract_feature")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint_payload.get("params", checkpoint_payload)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a state dict")
    state = _drop_retired_checkpoint_keys(state, source=str(checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return model


@torch.inference_mode()
def _encode_rows(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    dataroot: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[int, torch.Tensor]:
    dataset = _WaveRows(frame, dataroot)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    per_class: Dict[int, List[torch.Tensor]] = {}
    model.eval()
    model.mode = "extract_feature"
    for waveforms, labels in loader:
        features = model.encode(waveforms.to(device, non_blocking=True)).cpu()
        if features.ndim != 2:
            raise RuntimeError(f"model.encode returned shape {tuple(features.shape)}")
        for class_id in labels.unique(sorted=True):
            value = int(class_id)
            per_class.setdefault(value, []).append(features[labels == class_id])
    return {key: torch.cat(parts).float() for key, parts in per_class.items()}


def extract_geometry(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = verify_manifest(checkpoint, manifest_path)
    metadata_root = Path(args.metadata_root).resolve()
    dataroot = Path(args.dataroot).resolve()
    rows = build_protocol_rows(metadata_root)
    counts = protocol_counts(rows)
    if args.metadata_only:
        print(json.dumps({"metadata_only": True, "counts": counts}, sort_keys=True))
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = _load_encoder(args, checkpoint).to(device)
    encoded = {
        name: _encode_rows(
            model, frame, dataroot, device, args.batch_size, args.num_workers
        )
        for name, frame in rows.items()
    }
    base_means = torch.stack([
        encoded["base_train"][class_id].double().mean(0).float()
        for class_id in BASE_CLASSES
    ])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fsc89_pseudo_unseen_geometry",
        "dataset": "FSC-89",
        "offline": True,
        "encoder": "current model.encode",
        "feature_api": "model.encode",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "training_classes_zero_based": list(BASE_CLASSES),
        "base_classes_zero_based": list(BASE_CLASSES),
        "pseudo_novel_classes_zero_based": list(PSEUDO_NOVEL_CLASSES),
        "sealed_test_classes_zero_based": list(SEALED_TEST_CLASSES),
        "base_prototype_source": "train_mean_from_same_encoder",
        "pseudo_support_source": "train_csv_first_80_percent_per_class",
        "query_source": {
            "base": "validation_csv",
            "pseudo_novel": "train_csv_final_20_percent_per_class",
        },
        "test_csv_opened": False,
        "encoded_class_max": max(PSEUDO_NOVEL_CLASSES),
        "counts": counts,
        "base_train_means": base_means,
        "base_query_features": [
            encoded["base_query"][class_id] for class_id in BASE_CLASSES
        ],
        "pseudo_support_features": [
            encoded["pseudo_support"][class_id]
            for class_id in PSEUDO_NOVEL_CLASSES
        ],
        "pseudo_query_features": [
            encoded["pseudo_query"][class_id]
            for class_id in PSEUDO_NOVEL_CLASSES
        ],
    }
    output = Path(args.output).resolve()
    _save_torch(payload, output)
    print(
        f"geometry={output} device={device} "
        f"base_train={sum(counts['base_train'].values())} "
        f"pseudo_support_pool={sum(counts['pseudo_support'].values())} "
        f"validation_queries={sum(counts['base_query'].values()) + sum(counts['pseudo_query'].values())}"
    )


def validate_geometry(geometry: Mapping) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fsc89_pseudo_unseen_geometry",
        "dataset": "FSC-89",
        "offline": True,
        "encoder": "current model.encode",
        "feature_api": "model.encode",
        "training_classes_zero_based": list(BASE_CLASSES),
        "base_classes_zero_based": list(BASE_CLASSES),
        "pseudo_novel_classes_zero_based": list(PSEUDO_NOVEL_CLASSES),
        "sealed_test_classes_zero_based": list(SEALED_TEST_CLASSES),
        "base_prototype_source": "train_mean_from_same_encoder",
        "test_csv_opened": False,
        "encoded_class_max": 68,
    }
    for key, value in expected.items():
        if geometry.get(key) != value:
            raise ValueError(f"invalid geometry protocol field {key!r}")
    checkpoint_hash = geometry.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError("geometry is missing a SHA-256-bound checkpoint")
    base_means = geometry.get("base_train_means")
    support = geometry.get("pseudo_support_features")
    base_query = geometry.get("base_query_features")
    novel_query = geometry.get("pseudo_query_features")
    if not torch.is_tensor(base_means) or base_means.ndim != 2:
        raise ValueError("base_train_means must be a rank-2 tensor")
    if base_means.size(0) != len(BASE_CLASSES):
        raise ValueError("base_train_means must contain exactly 59 prototypes")
    if not all(isinstance(value, list) and len(value) == expected_count for value, expected_count in (
        (support, len(PSEUDO_NOVEL_CLASSES)),
        (base_query, len(BASE_CLASSES)),
        (novel_query, len(PSEUDO_NOVEL_CLASSES)),
    )):
        raise ValueError("geometry class-pool counts do not match the protocol")
    feature_dim = base_means.size(1)
    for pool in list(support) + list(base_query) + list(novel_query):
        if not torch.is_tensor(pool) or pool.ndim != 2 or pool.size(1) != feature_dim:
            raise ValueError("geometry contains an invalid feature pool")
    if any(len(pool) < 5 for pool in support):
        raise ValueError("every pseudo-novel support pool needs at least five samples")


def margin_scores(
    query: torch.Tensor, base_prototypes: torch.Tensor, novel_prototypes: torch.Tensor
) -> torch.Tensor:
    query = F.normalize(query.float(), dim=1)
    base = F.normalize(base_prototypes.float(), dim=1)
    novel = F.normalize(novel_prototypes.float(), dim=1)
    return (query @ novel.t()).max(1).values - (query @ base.t()).max(1).values


def make_episode_scores(
    geometry: Mapping,
    ways: int,
    rng: np.random.Generator,
    partition: str = "all",
    calibration_fraction: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[int]]]:
    if ways not in (5, 10):
        raise ValueError("the strict FSC-89 pseudo-unseen protocol supports 5 or 10 ways")
    if partition not in ("all", "calibration", "audit"):
        raise ValueError("partition must be all, calibration, or audit")

    def select_partition(pool: torch.Tensor, minimum: int) -> Tuple[torch.Tensor, int]:
        if partition == "all":
            return pool, 0
        split = int(len(pool) * calibration_fraction)
        if split < minimum or len(pool) - split < minimum:
            raise ValueError(
                f"feature pool of size {len(pool)} cannot form disjoint "
                f"calibration/audit partitions with minimum {minimum}"
            )
        if partition == "calibration":
            return pool[:split], 0
        return pool[split:], split

    chosen = sorted(rng.choice(len(PSEUDO_NOVEL_CLASSES), ways, replace=False).tolist())
    support_indices: Dict[int, List[int]] = {}
    novel_prototypes = []
    for local_class in chosen:
        pool, offset = select_partition(
            geometry["pseudo_support_features"][local_class], minimum=5
        )
        indices = sorted(rng.choice(len(pool), 5, replace=False).tolist())
        support_indices[int(PSEUDO_NOVEL_CLASSES[local_class])] = [
            index + offset for index in indices
        ]
        novel_prototypes.append(pool[indices].double().mean(0).float())
    novel_prototypes_tensor = torch.stack(novel_prototypes)
    base_query = torch.cat([
        select_partition(pool, minimum=1)[0]
        for pool in geometry["base_query_features"]
    ])
    novel_query = torch.cat([
        select_partition(
            geometry["pseudo_query_features"][local_class], minimum=1
        )[0]
        for local_class in chosen
    ])
    base_score = margin_scores(
        base_query, geometry["base_train_means"], novel_prototypes_tensor
    )
    novel_score = margin_scores(
        novel_query, geometry["base_train_means"], novel_prototypes_tensor
    )
    return base_score, novel_score, support_indices


def _episode_metrics(
    records: Sequence[Tuple[int, torch.Tensor, torch.Tensor]], threshold: float
) -> Dict[str, Mapping]:
    by_way: Dict[int, List[Tuple[float, float, float]]] = {}
    for ways, base_score, novel_score in records:
        base_recall = float((base_score < threshold).float().mean())
        novel_recall = float((novel_score >= threshold).float().mean())
        balanced = 0.5 * (base_recall + novel_recall)
        by_way.setdefault(ways, []).append((balanced, base_recall, novel_recall))
    result: Dict[str, Mapping] = {}
    for ways, values in sorted(by_way.items()):
        array = np.asarray(values, dtype=np.float64)
        result[str(ways)] = {
            "episodes": len(values),
            "balanced_accuracy_mean": float(array[:, 0].mean()),
            "balanced_accuracy_std": float(array[:, 0].std(ddof=0)),
            "base_recall_mean": float(array[:, 1].mean()),
            "novel_recall_mean": float(array[:, 2].mean()),
        }
    return result


def select_threshold(
    records: Sequence[Tuple[int, torch.Tensor, torch.Tensor]]
) -> Tuple[float, float]:
    """Maximize way/episode/group-balanced accuracy with one scalar margin."""
    scores, labels, weights = [], [], []
    num_ways = len(set(ways for ways, _, _ in records))
    episodes_per_way: Dict[int, int] = {}
    for ways, _, _ in records:
        episodes_per_way[ways] = episodes_per_way.get(ways, 0) + 1
    for ways, base_score, novel_score in records:
        group_weight = 1.0 / (num_ways * episodes_per_way[ways] * 2.0)
        scores.extend([base_score.numpy(), novel_score.numpy()])
        labels.extend([
            np.zeros(len(base_score), dtype=np.int64),
            np.ones(len(novel_score), dtype=np.int64),
        ])
        weights.extend([
            np.full(len(base_score), group_weight / len(base_score)),
            np.full(len(novel_score), group_weight / len(novel_score)),
        ])
    score = np.concatenate(scores)
    label = np.concatenate(labels)
    weight = np.concatenate(weights)
    fpr, tpr, thresholds = roc_curve(label, score, sample_weight=weight)
    objective = 0.5 * (tpr + 1.0 - fpr)
    best_value = float(objective.max())
    candidates = np.flatnonzero(np.isclose(objective, best_value, atol=1e-12))
    # Conservative deterministic tie-break: the largest threshold protects base
    # accuracy when several operating points have identical balanced accuracy.
    index = int(candidates[np.argmax(thresholds[candidates])])
    return float(thresholds[index]), float(roc_auc_score(label, score, sample_weight=weight))


def fit_margin(args: argparse.Namespace) -> None:
    geometry_path = Path(args.geometry).resolve()
    geometry = torch.load(geometry_path, map_location="cpu", weights_only=True)
    validate_geometry(geometry)
    if args.episodes < 2:
        raise ValueError("--episodes must be at least 2 per way")
    if not 0.0 < args.audit_fraction < 1.0:
        raise ValueError("--audit-fraction must be between zero and one")
    rng = np.random.default_rng(args.seed)
    calibration_records: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    audit_records: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    support_audit: List[Mapping] = []
    audit_count = max(1, int(round(args.episodes * args.audit_fraction)))
    calibration_count = args.episodes - audit_count
    if calibration_count < 1:
        raise ValueError("--episodes and --audit-fraction leave no calibration episodes")
    calibration_fraction = 1.0 - args.audit_fraction
    for ways in (5, 10):
        for split, count, target in (
            ("calibration", calibration_count, calibration_records),
            ("audit", audit_count, audit_records),
        ):
            for episode in range(count):
                base_score, novel_score, indices = make_episode_scores(
                    geometry,
                    ways,
                    rng,
                    partition=split,
                    calibration_fraction=calibration_fraction,
                )
                target.append((ways, base_score, novel_score))
                support_audit.append({
                    "ways": ways,
                    "episode": episode,
                    "split": split,
                    "support_indices_by_class": indices,
                })
    threshold, calibration_auroc = select_threshold(calibration_records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fsc89_pseudo_unseen_margin_gate",
        "dataset": "FMC",
        "dataset_protocol": "FSC-89",
        "offline": True,
        "encoder": "current model.encode",
        "feature_api": "model.encode",
        "geometry_sha256": sha256_file(geometry_path),
        "calibration_checkpoint_sha256": geometry["checkpoint_sha256"],
        "checkpoint_sha256": geometry["checkpoint_sha256"],
        "calibration_encoder_num_base": 59,
        "final_num_base": 69,
        "training_classes_zero_based": list(BASE_CLASSES),
        "validation_classes_zero_based": list(PSEUDO_NOVEL_CLASSES),
        "sealed_test_classes_zero_based": list(SEALED_TEST_CLASSES),
        "test_classes_accessed": [],
        "base_prototype_source": "train_mean_from_same_encoder",
        "base_fc_source": "train_mean",
        "novel_prototype_source": "exactly_5_shot_mean_from_same_encoder",
        "shot": 5,
        "support_shots": 5,
        "session_ways": [5, 10],
        "deployment_session_ways": [5, 10, 15, 20],
        "extrapolated_session_ways": [15, 20],
        "episodes_per_way": args.episodes,
        "calibration_episodes_per_way": calibration_count,
        "audit_episodes_per_way": audit_count,
        "calibration_fraction": calibration_fraction,
        "seed": args.seed,
        "score_name": "novel_top1_cosine_minus_base_top1_cosine",
        "score": "novel_max_minus_base_max_mean_prototype_cosine",
        "score_pipeline": {
            "metric": "cosine",
            "feature_centering": False,
            "base_logits": "same_encoder_train_mean_prototypes",
            "novel_logits": "same_encoder_exact_5_shot_mean_prototypes",
            "aggregation_stage": "before_support_bank_or_logit_bias",
        },
        "decision_rule": "choose_novel_if_score_greater_than_or_equal_to_threshold",
        "calibration_split_disjoint": True,
        "support_query_split_disjoint": True,
        "calibration_audit_feature_split_disjoint": True,
        "selection_objective": "mean_way_episode_group_balanced_accuracy",
        "margin_threshold": threshold,
        "calibration_weighted_auroc": calibration_auroc,
        "calibration_metrics": _episode_metrics(calibration_records, threshold),
        "audit_metrics": _episode_metrics(audit_records, threshold),
        "support_draw_audit": support_audit,
    }
    output = Path(args.output).resolve()
    _save_torch(payload, output)
    print(
        f"gate={output} threshold={threshold:.8f} "
        f"calibration_auroc={calibration_auroc:.6f} "
        f"audit={json.dumps(payload['audit_metrics'], sort_keys=True)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="hash-bind a verified 59-base checkpoint")
    manifest.add_argument("--checkpoint", required=True)
    manifest.add_argument("--trained-classes", required=True, help="must be exactly 0-58")
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(handler=create_manifest)

    extract = subparsers.add_parser("extract", help="extract leakage-safe model.encode geometry")
    extract.add_argument("--config", default="configs/exp_fsc89.yml")
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--dataroot", required=True)
    extract.add_argument("--metadata-root", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--batch-size", type=int, default=128)
    extract.add_argument("--num-workers", type=int, default=0)
    extract.add_argument("--seed", type=int, default=3420)
    extract.add_argument("--device", default="auto")
    extract.add_argument(
        "--metadata-only", action="store_true",
        help="validate split membership/counts without loading audio or the model",
    )
    extract.set_defaults(handler=extract_geometry)

    fit = subparsers.add_parser("fit", help="fit the scalar 5/10-way margin threshold")
    fit.add_argument("--geometry", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--episodes", type=int, default=50)
    fit.add_argument("--audit-fraction", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=3420)
    fit.set_defaults(handler=fit_margin)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(getattr(args, "seed", 3420))
    np.random.seed(getattr(args, "seed", 3420))
    torch.manual_seed(getattr(args, "seed", 3420))
    args.handler(args)


if __name__ == "__main__":
    main()
