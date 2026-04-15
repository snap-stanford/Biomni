"""Download training datasets from Zenodo."""
import requests
from pathlib import Path

from tqdm.auto import tqdm

from definitions import ROOT_DIR


def download_file(url: str, file_path: Path) -> None:
    """Download a file in chunks.

    :param url: File download URL
    :param file_path: File save path
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        # total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm(
            # total=total_size,
            desc=file_path.name,
            unit="B",
            unit_scale=True,
        )
        with file_path.open(mode="wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()
    
    return


if __name__ == "__main__":
    training_datasets = [
        {
            "filename": "chembl_train.gz",
            "url": "https://zenodo.org/record/6627127/files/chembl_train.gz",
        },
        {
            "filename": "purged_chembl_sliced.smi.gz",
            "url": "https://zenodo.org/record/6627127/files/purged_chembl_sliced.smi.gz?download=1",
        },
    ]

    download_folder = Path(ROOT_DIR).joinpath("training_sets")
    for training_dataset in training_datasets:
        download_file(
            url=training_dataset["url"],
            file_path=download_folder.joinpath(training_dataset["filename"]),
        )