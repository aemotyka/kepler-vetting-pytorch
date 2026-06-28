from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import random
import re
import time

import pandas as pd


KOI_PATH = Path("data/raw/koi_q1_q17_dr25.csv")
OUT_PATH = Path("data/metadata/lightcurve_manifest.csv")

TARGETS_PER_CLASS = 10
MAX_FILES_PER_TARGET = 4
RANDOM_SEED = 42
MAX_DIRECTORY_ATTEMPTS = 4
RETRY_SLEEP_SECONDS = 5

LABEL_MAP = {
    "CONFIRMED": 1,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0,
}


def kepler_lightcurve_dir(kepid: int) -> str:
    kic = f"{int(kepid):09d}"
    return f"https://archive.stsci.edu/pub/kepler/lightcurves/{kic[:4]}/{kic}/"


def read_url_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")

def read_url_text_with_retries(url: str) -> str:
    last_error = None

    for attempt in range(1, MAX_DIRECTORY_ATTEMPTS + 1):
        try:
            return read_url_text(url)
        except Exception as exc:
            last_error = exc

            if attempt < MAX_DIRECTORY_ATTEMPTS:
                print(
                    f"retry {attempt}/{MAX_DIRECTORY_ATTEMPTS} failed for {url}: "
                    f"{type(exc).__name__}: {exc}"
                )
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        f"failed after {MAX_DIRECTORY_ATTEMPTS} attempts for {url}: {last_error}"
    )

def list_long_cadence_fits(directory_url: str) -> list[str]:
    html = read_url_text_with_retries(directory_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
    return sorted(
        urljoin(directory_url, href)
        for href in hrefs
        if href.endswith("_llc.fits")
    )


def main() -> None:
    if not KOI_PATH.exists():
        raise FileNotFoundError(f"Missing KOI CSV: {KOI_PATH}")

    df = pd.read_csv(KOI_PATH)
    df = df.dropna(subset=["kepid", "kepoi_name", "koi_disposition", "koi_period", "koi_time0bk"])
    df = df[df["koi_disposition"].isin(LABEL_MAP)].copy()

    df["kepid"] = df["kepid"].astype(int)
    df["binary_label"] = df["koi_disposition"].map(LABEL_MAP)

    rng = random.Random(RANDOM_SEED)

    positive_rows = df[df["binary_label"] == 1].to_dict("records")
    negative_rows = df[df["binary_label"] == 0].to_dict("records")

    rng.shuffle(positive_rows)
    rng.shuffle(negative_rows)

    selected = []
    seen_kepids = set()

    def collect(rows: list[dict], desired_label: int) -> None:
        class_count = 0

        for row in rows:
            if class_count >= TARGETS_PER_CLASS:
                break

            kepid = int(row["kepid"])

            # Keep the first dataset simple: one KOI per star.
            # Later we can support multi-KOI stars with grouped splits.
            if kepid in seen_kepids:
                continue

            directory_url = kepler_lightcurve_dir(kepid)

            try:
                file_urls = list_long_cadence_fits(directory_url)
            except Exception as exc:
                print(f"skip kepid={kepid}: {type(exc).__name__}: {exc}")
                continue

            if not file_urls:
                print(f"skip kepid={kepid}: no long-cadence FITS files")
                continue

            selected_urls = file_urls[:MAX_FILES_PER_TARGET]

            selected.append(
                {
                    "kepid": kepid,
                    "kepoi_name": row["kepoi_name"],
                    "koi_disposition": row["koi_disposition"],
                    "binary_label": desired_label,
                    "koi_period": row["koi_period"],
                    "koi_time0bk": row["koi_time0bk"],
                    "koi_duration": row.get("koi_duration"),
                    "koi_depth": row.get("koi_depth"),
                    "koi_model_snr": row.get("koi_model_snr"),
                    "koi_prad": row.get("koi_prad"),
                    "koi_steff": row.get("koi_steff"),
                    "koi_slogg": row.get("koi_slogg"),
                    "koi_srad": row.get("koi_srad"),
                    "koi_kepmag": row.get("koi_kepmag"),
                    "directory_url": directory_url,
                    "n_available_fits": len(file_urls),
                    "selected_fits_urls": "|".join(selected_urls),
                }
            )

            seen_kepids.add(kepid)
            class_count += 1

            print(
                f"selected label={desired_label} "
                f"kepid={kepid} "
                f"koi={row['kepoi_name']} "
                f"disposition={row['koi_disposition']} "
                f"files={len(file_urls)}"
            )

    collect(positive_rows, desired_label=1)
    collect(negative_rows, desired_label=0)

    out = pd.DataFrame(selected)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print()
    print("wrote:", OUT_PATH)
    print("shape:", out.shape)
    print("binary_label_counts:")
    print(out["binary_label"].value_counts().sort_index())
    print("disposition_counts:")
    print(out["koi_disposition"].value_counts())


if __name__ == "__main__":
    main()
