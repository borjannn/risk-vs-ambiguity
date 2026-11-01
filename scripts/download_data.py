"""Download the raw choices13k and CPC18 releases into data/raw/.

Usage: uv run scripts/download_data.py
"""

from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

C13K = "https://raw.githubusercontent.com/jcpeterson/choices13k/main/"
CPC18 = "https://zenodo.org/api/records/2571510/files/"

FILES = {
    "choices13k/c13k_selections.csv": C13K + "c13k_selections.csv",
    "choices13k/c13k_problems.json": C13K + "c13k_problems.json",
    "cpc18/all_cpc18_raw_data.csv": CPC18 + quote("all CPC18 raw data.csv") + "/content",
    "cpc18/cpc18_raw_data_dictionary.pdf": CPC18
    + quote("Dictionary for the choice prediction competition 2018 raw data.pdf")
    + "/content",
}


def main():
    for rel, url in FILES.items():
        dest = RAW / rel
        if dest.exists():
            print(f"exists  {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch   {rel}")
        urlretrieve(url, dest)


if __name__ == "__main__":
    main()
