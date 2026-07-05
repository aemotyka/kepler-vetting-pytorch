from __future__ import annotations

import pandas as pd

from kepler_vetting.processing.common import (
    MODEL_READINESS_REPORT_PATH,
    MODEL_READY_MAX_GLOBAL_MISSING_FRACTION,
    MODEL_READY_MAX_LOCAL_MISSING_FRACTION,
    MODEL_READY_MIN_CLEAN_POINTS,
    MODEL_READY_MIN_FITS_FILES,
)
from kepler_vetting.processing.model_readiness import (
    build_model_readiness_report,
    reason_counts,
)


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    report = build_model_readiness_report()

    MODEL_READINESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(MODEL_READINESS_REPORT_PATH, index=False)

    print("readiness thresholds:")
    print("  min_clean_points:", MODEL_READY_MIN_CLEAN_POINTS)
    print("  min_fits_files:", MODEL_READY_MIN_FITS_FILES)
    print("  max_global_missing_fraction:", MODEL_READY_MAX_GLOBAL_MISSING_FRACTION)
    print("  max_local_missing_fraction:", MODEL_READY_MAX_LOCAL_MISSING_FRACTION)

    print_section("row counts")
    print("total_processed_rows:", len(report))
    print("model_ready_rows:", int(report["model_ready"].sum()))
    print("not_model_ready_rows:", int((~report["model_ready"]).sum()))

    print_section("model-ready counts by label")
    print(
        pd.crosstab(
            report["binary_label"],
            report["model_ready"],
            rownames=["binary_label"],
            colnames=["model_ready"],
        )
    )

    print_section("model-ready counts by disposition")
    print(
        pd.crosstab(
            report["koi_disposition"],
            report["model_ready"],
            rownames=["koi_disposition"],
            colnames=["model_ready"],
        )
    )

    counts = reason_counts(report)
    print_section("failure reason counts")
    if counts:
        for reason, count in counts.most_common():
            print(f"{reason}: {count}")
    else:
        print("none")

    print_section("coverage summary")
    print(
        report[
            [
                "n_fits_files",
                "n_clean_points",
                "clean_fraction",
                "time_span_days",
                "global_missing_bin_fraction_before_interp",
                "local_missing_bin_fraction_before_interp",
                "global_min_abs_phase",
                "local_min_abs_phase",
            ]
        ].describe()
    )

    print_section("worst global missing-bin rows")
    print(
        report.sort_values(
            "global_missing_bin_fraction_before_interp",
            ascending=False,
        )[
            [
                "kepid",
                "kepoi_name",
                "koi_disposition",
                "binary_label",
                "n_fits_files",
                "n_clean_points",
                "global_missing_bin_fraction_before_interp",
                "local_missing_bin_fraction_before_interp",
                "model_ready",
                "model_ready_failure_reasons",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print_section("worst local missing-bin rows")
    print(
        report.sort_values(
            "local_missing_bin_fraction_before_interp",
            ascending=False,
        )[
            [
                "kepid",
                "kepoi_name",
                "koi_disposition",
                "binary_label",
                "n_fits_files",
                "n_clean_points",
                "global_missing_bin_fraction_before_interp",
                "local_missing_bin_fraction_before_interp",
                "model_ready",
                "model_ready_failure_reasons",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print_section("local minima farthest from phase zero")
    print(
        report.sort_values(
            "local_min_abs_phase",
            ascending=False,
        )[
            [
                "kepid",
                "kepoi_name",
                "koi_disposition",
                "binary_label",
                "local_min_phase",
                "local_min_flux",
                "local_min_abs_phase",
                "model_ready",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("wrote:", MODEL_READINESS_REPORT_PATH)
    print("DIAGNOSE_MODEL_READINESS_OK")


if __name__ == "__main__":
    main()
