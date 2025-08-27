'''
Module with tools to ensure the reference point for rotation is in the center of the 2D or 3D density.
'''

from pathlib import Path

import numpy as np
from mrcfile import read as read_mrc

from tools.spider_files3 import open_volume

def open_density(path: Path) -> np.ndarray:
    '''open density from file, supports .vol and .mrc formats'''
    match path:
        case Path(suffix='.vol'):
            return open_volume(path).transpose(2, 1, 0)  # Spider files are in zyx order
        case Path(suffix='.mrc') | Path(suffix='.mrcs'):
            res = read_mrc(path).astype(np.float32)
            return res.transpose(2, 1, 0) if res.ndim == 3 else res

def center_of_mass(density: np.ndarray) -> np.ndarray:
    '''compute center of mass of a density'''
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in density.shape], indexing='ij'), axis=-1)
    total_mass = np.sum(density)
    if total_mass == 0:
        return np.zeros(3)
    return np.sum(coords * density[..., None], axis=tuple(np.arange(density.ndim))) / total_mass

def np_shift(arr: np.ndarray, shift: tuple[int, ...], axis: tuple[int, ...], fill: float = 0.0) -> np.ndarray:
    '''roll array by given shift, zeroing out the part crossing over the boundary'''
    res = np.roll(arr, shift, axis=tuple(range(arr.ndim)))
    for ax, sh in zip(axis, shift):
        ax = ax % arr.ndim
        idx = (slice(None),) * ax
        idx += (slice(sh,None),) if sh < 0 else (slice(None,sh),)
        res[idx] = fill
    return res

def center_shift(density: np.ndarray) -> np.ndarray:
    '''open density and compute center of mass, aligned to the center of the volume'''
    com = center_of_mass(density)
    shift = ((np.array(density.shape) - 1) // 2 - com).round(0).astype(int)
    return np_shift(density, shift, axis=tuple(range(density.ndim)))
