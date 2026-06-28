from pathlib import Path
import time
from urllib.request import Request, urlopen

import pandas as pd


MANIFEST_PATH = Path("data/metadata/lightcurve_manifest.csv")
OUT_ROOT = Path("data/raw/lightcurves")

MAX_DOWNLOAD_ATTEMPTS = 4
RETRY_SLEEP_SECONDS = 5


def download_url(url: str, output_path: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with urlopen(req, timeout=240) as response:
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    tmp_path.replace(output_path)


def download_with_retries(url: str, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    last_error = None

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            download_url(url, output_path)
            return
        except Exception as exc:
            last_error = exc

            if tmp_path.exists():
                tmp_path.unlink()

            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                print(
                    f"retry {attempt}/{MAX_DOWNLOAD_ATTEMPTS} failed for {url}: "
                    f"{type(exc).__name__}: {exc}"
                )
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        f"failed after {MAX_DOWNLOAD_ATTEMPTS} attempts for {url}: {last_error}"
    )


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    attempted = 0
    downloaded = 0
    skipped = 0
    failed = 0

    for _, row in manifest.iterrows():
        kepid = int(row["kepid"])
        kic = f"{kepid:09d}"

        urls = str(row["selected_fits_urls"]).split("|")

        for url in urls:
            attempted += 1
            filename = url.rsplit("/", 1)[-1]
            output_path = OUT_ROOT / kic / filename

            if output_path.exists() and output_path.stat().st_size > 0:
                skipped += 1
                print(f"skip existing: {output_path}")
                continue

            try:
                print(f"download: {url}")
                download_with_retries(url, output_path)
                downloaded += 1
                print(f"wrote: {output_path}")
            except Exception as exc:
                failed += 1
                print(f"FAILED url={url}: {type(exc).__name__}: {exc}")

    print()
    print("attempted:", attempted)
    print("downloaded:", downloaded)
    print("skipped:", skipped)
    print("failed:", failed)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()