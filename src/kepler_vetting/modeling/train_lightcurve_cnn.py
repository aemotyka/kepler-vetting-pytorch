from __future__ import annotations

from kepler_vetting.modeling.lightcurve_common import (
    LightcurveTrainingPaths,
    run_phase_view_cnn_baseline,
)

from kepler_vetting.processing.common import RUN_METRICS_DIR, RUN_MODEL_DIR

VIEW_NAME = "local_view"
MODEL_NAME = "local_view_cnn"

METRICS_DIR = RUN_METRICS_DIR
MODEL_DIR = RUN_MODEL_DIR

PATHS = LightcurveTrainingPaths(
    per_seed_metrics_path=METRICS_DIR / "lightcurve_cnn_metrics_by_seed.csv",
    summary_metrics_path=METRICS_DIR / "lightcurve_cnn_metrics_summary.csv",
    predictions_path=METRICS_DIR / "lightcurve_cnn_predictions.csv",
    training_history_path=METRICS_DIR / "lightcurve_cnn_training_history.csv",
    model_path=MODEL_DIR / "local_view_cnn.pt",
)


def main() -> None:
    run_phase_view_cnn_baseline(
        view_name=VIEW_NAME,
        model_name=MODEL_NAME,
        progress_description="local CNN seeds",
        paths=PATHS,
        success_marker="TRAIN_LIGHTCURVE_CNN_OK",
    )


if __name__ == "__main__":
    main()
