from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from kepler_vetting.processing.common import PROCESSED_NPZ_PATH


OUT_DIR = Path("outputs/figures/examples")
EXAMPLES_PER_LABEL = 3


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "example"


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(x, y, linewidth=1)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not PROCESSED_NPZ_PATH.exists():
        raise FileNotFoundError(f"missing processed dataset: {PROCESSED_NPZ_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_NPZ_PATH)

    labels = data["labels"]
    kepid = data["kepid"]
    kepoi_name = data["kepoi_name"]
    disposition = data["disposition"]

    global_phase = data["global_phase"]
    global_view = data["global_view"]
    local_phase = data["local_phase"]
    local_view = data["local_view"]

    written = []

    for label in [1, 0]:
        indices = np.where(labels == label)[0][:EXAMPLES_PER_LABEL]
        label_name = "planet_like" if label == 1 else "false_positive"

        for idx in indices:
            name = safe_filename(f"{label_name}_{kepoi_name[idx]}_{kepid[idx]}")

            global_path = OUT_DIR / f"{name}_global.png"
            local_path = OUT_DIR / f"{name}_local.png"

            title_base = (
                f"{kepoi_name[idx]} / KIC {kepid[idx]} / "
                f"{disposition[idx]} / label={label}"
            )

            save_line_plot(
                x=global_phase,
                y=global_view[idx],
                title=f"Global phase-folded view: {title_base}",
                xlabel="Orbital phase",
                ylabel="Relative flux",
                output_path=global_path,
            )

            save_line_plot(
                x=local_phase[idx],
                y=local_view[idx],
                title=f"Local transit view: {title_base}",
                xlabel="Orbital phase",
                ylabel="Relative flux",
                output_path=local_path,
            )

            written.extend([global_path, local_path])

    print("wrote example plots:")
    for path in written:
        print(" ", path)

    print("PLOT_PROCESSED_EXAMPLES_OK")


if __name__ == "__main__":
    main()