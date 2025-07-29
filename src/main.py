
from pathlib import Path
from time import time

import numpy as np
import torch

from tools.xmipp import AtomLine
from tools.spider_files3 import open_volume

DATA = Path(__file__).parents[1] / "data"

if __name__ == "__main__":
    coordinates = np.stack([atom.coordinates for atom in AtomLine.from_pdb(DATA / "3uat_psi=135_tilt=45_rot=120_better.pdb")])
    voxels = open_volume(DATA / "3uat.vol")

    if torch.cuda.is_available():
        from align.volume_heat_kernel_torch import Registrator
        from tools.utils_torch import from_numpy
        coordinates = from_numpy(coordinates)
        voxels = from_numpy(voxels)
    else:
        from align.volume_heat_kernel import Registrator

    print("Preprocessing...")
    t0 = time()
    registrator = Registrator(n_spherical=8, n_inplane=8, l_max = 10,k_res = 2)
    registrator.set_reference(voxels = voxels)
    registrator.filter_k(thresh=0.8).preprocess()
    print(f"Preprocessing done in {time() - t0:.2f} seconds")

    print("Aligning...")
    t1 = time()
    for rotation in registrator.align(coordinates, sigma=1.0).items():
        print(rotation)
    print(f"Alignment done in {time() - t1:.2f} seconds")

    print(registrator.correlation_frame().nlargest(10,'correlation').to_string(index=True, float_format=lambda x: f"{x:.4f}"))