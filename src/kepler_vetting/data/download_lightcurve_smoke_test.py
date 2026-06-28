from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import re

import pandas as pd
from astropy.io import fits


KOI_PATH = Path("data/raw/koi_q1_q17_dr25.csv")
OUT_DIR = Path("data/raw/lightcurves")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def kepler_lightcurve_dir(kepid: int) -> str:
    kic = f"{int(kepid):09d}"
    return f"https://archive.stsci.edu/pub/kepler/lightcurves/{kic[:4]}/{kic}/"


def read_url_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def download_url(url: str, output_path: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as response:
        with output_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def list_long_cadence_fits(directory_url: str) -> list[str]:
    html = read_url_text(directory_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
    return sorted(
        urljoin(directory_url, href)
        for href in hrefs
        if href.endswith("_llc.fits")
    )


df = pd.read_csv(KOI_PATH)
df = df.dropna(subset=["kepid", "koi_disposition"])
df["kepid"] = df["kepid"].astype(int)

# Try a small labeled sample from the NASA KOI table.
sample = df.sort_values(["koi_disposition", "kepid"]).head(200)

chosen = None

for _, row in sample.iterrows():
    kepid = int(row["kepid"])
    directory_url = kepler_lightcurve_dir(kepid)

    try:
        files = list_long_cadence_fits(directory_url)
    except Exception as exc:
        print(f"skip kepid={kepid}: {type(exc).__name__}: {exc}")
        continue

    if files:
        chosen = row, directory_url, files
        break

if chosen is None:
    raise RuntimeError("Could not find a downloadable long-cadence FITS file in the first 200 KOI rows.")

row, directory_url, files = chosen

print("selected_kepid:", int(row["kepid"]))
print("selected_koi:", row["kepoi_name"])
print("selected_label:", row["koi_disposition"])
print("directory_url:", directory_url)
print("available_long_cadence_files:", len(files))
print("first_file_url:", files[0])

output_path = OUT_DIR / files[0].rsplit("/", 1)[-1]

if output_path.exists():
    print("already_downloaded:", output_path)
else:
    print("downloading_to:", output_path)
    download_url(files[0], output_path)

print("opening_fits:", output_path)

with fits.open(output_path) as hdul:
    hdul.info()
    lightcurve_table = hdul[1].data
    columns = list(lightcurve_table.columns.names)

    print("rows:", len(lightcurve_table))
    print("columns:", columns)

    for col in ["TIME", "SAP_FLUX", "PDCSAP_FLUX", "QUALITY"]:
        if col in columns:
            values = lightcurve_table[col]
            print(f"{col}_first_5:", values[:5])

print("SMOKE_TEST_OK")
