from __future__ import annotations

import pandas as pd

from kepler_vetting.processing.common import (
    MANIFEST_PATH,
    TABULAR_FEATURES,
    local_fits_paths_for_row,
    stitch_lightcurves,
    validate_manifest_columns,
)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    validate_manifest_columns(manifest)

    print("manifest_path:", MANIFEST_PATH)
    print("manifest_rows:", len(manifest))
    print()

    print("label counts:")
    print(manifest["binary_label"].value_counts().sort_index())
    print()

    print("disposition counts:")
    print(manifest["koi_disposition"].value_counts())
    print()

    availability_rows = []

    for _, row in manifest.iterrows():
        paths, missing = local_fits_paths_for_row(row)

        record = {
            "kepid": int(row["kepid"]),
            "kepoi_name": row["kepoi_name"],
            "koi_disposition": row["koi_disposition"],
            "binary_label": int(row["binary_label"]),
            "local_fits_files": len(paths),
            "missing_fits_files": len(missing),
            "n_raw_points": 0,
            "n_clean_points": 0,
            "clean_fraction": 0.0,
            "time_span_days": 0.0,
            "inspect_ok": False,
            "failure_reason": "",
        }

        if not paths:
            record["failure_reason"] = "no local FITS files found"
            availability_rows.append(record)
            continue

        try:
            _, _, stats = stitch_lightcurves(paths)
            record.update(
                {
                    "n_raw_points": stats["n_raw_points"],
                    "n_clean_points": stats["n_clean_points"],
                    "clean_fraction": stats["clean_fraction"],
                    "time_span_days": stats["time_span_days"],
                    "inspect_ok": True,
                }
            )
        except Exception as exc:
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"

        availability_rows.append(record)

    availability = pd.DataFrame(availability_rows)

    print("local FITS availability:")
    print(
        availability[
            [
                "kepid",
                "kepoi_name",
                "koi_disposition",
                "local_fits_files",
                "missing_fits_files",
                "inspect_ok",
            ]
        ].to_string(index=False)
    )
    print()

    print("clean point summary:")
    print(
        availability[
            [
                "local_fits_files",
                "missing_fits_files",
                "n_raw_points",
                "n_clean_points",
                "clean_fraction",
                "time_span_days",
            ]
        ].describe()
    )
    print()

    numeric_cols = [col for col in TABULAR_FEATURES if col in manifest.columns]
    print("tabular feature summary:")
    print(manifest[numeric_cols].describe())
    print()

    failures = availability[~availability["inspect_ok"]]
    if not failures.empty:
        print("inspection failures:")
        print(
            failures[
                [
                    "kepid",
                    "kepoi_name",
                    "failure_reason",
                ]
            ].to_string(index=False)
        )
        raise SystemExit(1)

    print("INSPECT_MANIFEST_OK")


if __name__ == "__main__":
    main()
