from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def configured_split_fraction(
    name: str,
    default: float,
) -> float:
    raw_value = os.environ.get(name, "").strip()

    if not raw_value:
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float; got {raw_value!r}") from exc

    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1); got {value}")

    return value


def configured_split_candidates(
    name: str,
    default: int,
) -> int:
    raw_value = os.environ.get(name, "").strip()

    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}") from exc

    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")

    return value


DEFAULT_TEST_SIZE = configured_split_fraction(
    name="KEPLER_VETTING_TEST_SIZE",
    default=0.20,
)
DEFAULT_VAL_SIZE = configured_split_fraction(
    name="KEPLER_VETTING_VAL_SIZE",
    default=0.20,
)
DEFAULT_N_CANDIDATES = configured_split_candidates(
    name="KEPLER_VETTING_N_SPLIT_CANDIDATES",
    default=128,
)

SPLIT_MODE = f"grouped_by_kepid_test{DEFAULT_TEST_SIZE:.2f}_val{DEFAULT_VAL_SIZE:.2f}"


def validate_split_inputs(
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got shape {labels.shape}")

    if groups.ndim != 1:
        raise ValueError(f"groups must be 1D; got shape {groups.shape}")

    if labels.shape[0] != groups.shape[0]:
        raise ValueError(
            f"labels/groups length mismatch: {labels.shape[0]} vs {groups.shape[0]}"
        )

    if labels.shape[0] == 0:
        raise ValueError("cannot split empty labels")

    if set(labels.tolist()) - {0, 1}:
        raise ValueError(
            f"labels must only contain 0/1 values; got {sorted(set(labels.tolist()))}"
        )

    if np.unique(groups).shape[0] < 3:
        raise ValueError("need at least 3 unique groups for train/val/test split")

    return labels.astype(np.int64), groups


def validate_no_group_overlap(
    splits: dict[str, np.ndarray],
    groups: np.ndarray,
) -> None:
    group_sets = {
        split_name: set(groups[indices].tolist())
        for split_name, indices in splits.items()
    }

    split_names = sorted(group_sets)

    for idx, left_name in enumerate(split_names):
        for right_name in split_names[idx + 1 :]:
            overlap = group_sets[left_name] & group_sets[right_name]

            if overlap:
                examples = sorted(overlap)[:10]
                raise ValueError(
                    f"group leakage between {left_name} and {right_name}; "
                    f"examples={examples}"
                )


def split_score(
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    target_fractions: dict[str, float],
) -> float:
    global_positive_rate = float(labels.mean())
    n_total = labels.shape[0]

    score = 0.0

    for split_name, indices in splits.items():
        split_labels = labels[indices]

        if split_labels.shape[0] == 0:
            return float("inf")

        if len(np.unique(split_labels)) < 2:
            return float("inf")

        split_fraction = split_labels.shape[0] / n_total
        positive_rate = float(split_labels.mean())

        score += abs(split_fraction - target_fractions[split_name])
        score += 5.0 * abs(positive_rate - global_positive_rate)

    return score


def group_shuffle_once(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, np.ndarray]:
    all_indices = np.arange(labels.shape[0])

    first_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_val_idx, test_idx = next(
        first_splitter.split(
            all_indices,
            labels,
            groups,
        )
    )

    val_size_within_train_val = val_size / (1.0 - test_size)

    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size_within_train_val,
        random_state=seed + 104729,
    )

    train_rel_idx, val_rel_idx = next(
        second_splitter.split(
            train_val_idx,
            labels[train_val_idx],
            groups[train_val_idx],
        )
    )

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    splits = {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }

    validate_no_group_overlap(
        splits=splits,
        groups=groups,
    )

    return splits


def split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> dict[str, np.ndarray]:
    labels, groups = validate_split_inputs(
        labels=labels,
        groups=groups,
    )

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")

    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1); got {val_size}")

    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size + val_size must be < 1; got {test_size + val_size}"
        )

    target_fractions = {
        "train": 1.0 - test_size - val_size,
        "val": val_size,
        "test": test_size,
    }

    best_splits = None
    best_score = float("inf")
    last_error = None

    for candidate_idx in range(n_candidates):
        candidate_seed = seed * 1009 + candidate_idx

        try:
            candidate_splits = group_shuffle_once(
                labels=labels,
                groups=groups,
                seed=candidate_seed,
                test_size=test_size,
                val_size=val_size,
            )

            candidate_score = split_score(
                labels=labels,
                splits=candidate_splits,
                target_fractions=target_fractions,
            )

        except Exception as exc:
            last_error = exc
            continue

        if candidate_score < best_score:
            best_splits = candidate_splits
            best_score = candidate_score

    if best_splits is None:
        raise RuntimeError(
            "could not create a valid grouped split; "
            f"last_error={type(last_error).__name__}: {last_error}"
        )

    return best_splits


def describe_split(
    labels: np.ndarray,
    groups: np.ndarray,
    splits: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    labels, groups = validate_split_inputs(
        labels=labels,
        groups=groups,
    )

    rows = []

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        indices = splits[split_name]
        split_labels = labels[indices]
        split_groups = groups[indices]

        rows.append(
            {
                "split": split_name,
                "n_rows": int(indices.shape[0]),
                "n_groups": int(np.unique(split_groups).shape[0]),
                "n_negative": int((split_labels == 0).sum()),
                "n_positive": int((split_labels == 1).sum()),
                "positive_rate": float(split_labels.mean()),
            }
        )

    return rows
