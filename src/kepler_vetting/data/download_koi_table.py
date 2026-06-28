from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import urlretrieve


OUT_PATH = Path("data/raw/koi_q1_q17_dr25.csv")


QUERY = """
select
  kepid,
  kepoi_name,
  koi_disposition,
  koi_pdisposition,
  koi_period,
  koi_time0bk,
  koi_duration,
  koi_depth,
  koi_prad,
  koi_teq,
  koi_insol,
  koi_model_snr,
  koi_steff,
  koi_slogg,
  koi_srad,
  koi_kepmag
from q1_q17_dr25_koi
where koi_disposition in ('CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE')
"""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        + quote_plus(QUERY)
        + "&format=csv"
    )

    urlretrieve(url, OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()