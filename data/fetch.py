'''
Module to fetch both pdb and volume data for testing.
'''

from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError

import requests
import pandas as pd
import numpy as np

DATA = Path(__file__).parent

def fetch_pdb(pdb_id: str, data_dir: Path = DATA, label = None) -> Path:
    '''
    Fetch a PDB file by its ID.
    '''
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    path = data_dir / f"{pdb_id if label is None else label}.pdb"

    if path.exists():
        return path
    path, _ = urlretrieve(url, path)
    return path

def get_volume_url(emdb_id: str) -> str:
    '''
    Get the URL for a volume map file by its ID.
    '''
    meta_url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
    meta = requests.get(meta_url).json()
    try:
        match meta['map']:
            case {'url': url}:
                return url
            case {'file': file}:
                return f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb_id}/map/{file}"
            case _:
                return f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb_id}/map/{emdb_id}.map.gz"
    except KeyError:
        raise requests.HTTPError(f"Could not find volume map url to begin with for {emdb_id}")

def fetch_volume(emdb_id: str, data_dir: Path = DATA, label = None) -> Path:
    '''
    Fetch a volume map file by its ID.
    '''
    # url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb_id}/map/{emdb_id}.map.gz"
    url = get_volume_url(emdb_id)
    path = data_dir / f"{emdb_id if label is None else label}.map.gz"

    if path.exists():
        return path
    path, _ = urlretrieve(url, path)
    return path

def fetch(record: pd.Series, data_dir: Path = DATA) -> pd.Series:
    '''
    Fetch a record from the EMDB or PDB database.
    '''
    fetched = []
    try:
        fetched.append(fetch_pdb(record['pdb_entry'], data_dir, record['emdb_entry']))
        fetched.append(fetch_volume(record['emdb_entry'], data_dir))
    except (HTTPError,requests.HTTPError):
        print(f"Failed to fetch data for {record['emdb_entry']}")
        for path in fetched:
            path.unlink()
        fetched = ["", ""]
    record['pdb_path'] = str(fetched[0])
    record['volume_path'] = str(fetched[1])
    return record

if __name__ == "__main__":
    from scipy.stats import zscore
    selection = pd.read_csv(Path(__file__).parent / "fetch.csv")
    prob = np.exp(-zscore(selection['total_atoms']) ** 2 / 2)
    subset = selection.sample(20,weights=prob, random_state=42)
    result = subset.apply(fetch, axis=1, data_dir=DATA / "benchmark")
    result.to_csv(Path(__file__).parent / "fetch_sample.csv", index=False)
