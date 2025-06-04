import os
import gzip
import shutil
from typing import Iterable

import requests


class IMDBDatasetDownloader:
    """Download and extract IMDb dataset files."""

    BASE_URL = "https://datasets.imdbws.com/"

    FILES = [
        "title.basics.tsv.gz",
        "title.crew.tsv.gz",
        "title.ratings.tsv.gz",
        "name.basics.tsv.gz",
        "title.principals.tsv.gz",
    ]

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    def download(self, files: Iterable[str] | None = None) -> None:
        """Download and extract the specified IMDb dataset files."""
        files = files or self.FILES
        os.makedirs(self.output_dir, exist_ok=True)
        for filename in files:
            url = self.BASE_URL + filename
            gz_path = os.path.join(self.output_dir, filename)
            tsv_path = gz_path[:-3]  # remove .gz
            if os.path.exists(tsv_path):
                print(f"File {tsv_path} already exists, skipping download.")
                continue
            print(f"Downloading {url}")
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(gz_path, "wb") as f_out:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(chunk)
            print(f"Extracting {gz_path}")
            with gzip.open(gz_path, "rb") as f_in, open(tsv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
        print("Download complete.")
