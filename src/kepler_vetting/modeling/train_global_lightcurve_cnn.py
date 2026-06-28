from __future__ import annotations

from pathlib import Path

from kepler_vetting.modeling.lightcurve_common import (
    LightcurveTrainingPaths,
    run_phase_view_cnn_baseline,
)


VIEW_NAME = "global_view"
MODEL_NAME = "global_view_cnn"

METRICS_DIR = Path("outputs/metrics")
MODEL_DIR = Path("artifacts/models")

PATHS = LightcurveTrainingPaths(
    per_seed_metrics_path=METRICS_DIR / "global_lightcurve_cnn_metrics_by_seed.csv",
    summary_metrics_path=METRICS_DIR / "global_lightcurve_cnn_metrics_summary.csv",
    predictions_path=METRICS_DIR / "global_lightcurve_cnn_predictions.csv",
    training_history_path=METRICS_DIR / "global_lightcurve_cnn_training_history.csv",
    model_path=MODEL_DIR / "global_view_cnn.pt",
)


def main() -> None:
    run_phase_view_cnn_baseline(
        view_name=VIEW_NAME,
        model_name=MODEL_NAME,
        progress_description="global CNN seeds",
        paths=PATHS,
        success_marker="TRAIN_GLOBAL_LIGHTCURVE_CNN_OK",
    )


if __name__ == "__main__":
    main()