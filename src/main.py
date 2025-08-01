
from pathlib import Path
from time import time

import numpy as np

from tools.xmipp import AtomLine
from tools.spider_files3 import open_volume

DATA = Path(__file__).parents[1] / "data"

if __name__ == "__main__":
    coordinates = np.stack([atom.coordinates for atom in AtomLine.from_pdb(DATA / "3uat_psi=135_tilt=45_rot=120.pdb")])
    voxels = open_volume(DATA / "3uat.vol").transpose((2,1,0))

    from align.volume_heat_kernel import Registrator

    print("Preprocessing...")
    t0 = time()
    registrator = Registrator(n_spherical=36, n_inplane=36, l_max = 5,k_res = 2)
    registrator.set_reference(voxels = voxels)
    registrator.filter_k(thresh=0.8).preprocess()
    print(f"Preprocessing done in {time() - t0:.2f} seconds")

    print("Aligning...")
    t1 = time()
    prediction = registrator.align(coordinates, sigma=1.0)
    print(f"Predicted inverse rotation: {', '.join(f'{angle}:{val:.2f}' for angle,val in prediction.items())}")

    print(f"Alignment done in {time() - t1:.2f} seconds")

    print(registrator.correlation_frame().nlargest(10,'correlation').to_string(index=True, float_format=lambda x: f"{x:.4f}"))