'''
Roundtrip tests: Derive a rotated density from a model and retrieve the original angles.
'''
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import mrcfile
import starfile
import pytest

from tools.xmipp import AtomLine, Xmipp
from tests import DATA
from tools.spider_files3 import open_volume
from tools.center import center_shift
from align.volume_heat_kernel import Registrator

def test_roundtrip_3d(pdb: str = "3uat.pdb", angles: tuple[float, float, float] = (135, 45, 120)):
    coordinates = np.stack([atom.coordinates for atom in AtomLine.from_pdb(DATA / pdb)])
    rotated_coordinates = coordinates @ Rotation.from_euler('ZYZ', angles, degrees=True).as_matrix().T

    xmipp = Xmipp(DATA / "test_data")
    volume_path = xmipp.volume_from_pdb(rotated_coordinates)
    voxels = open_volume(volume_path).transpose((2, 1, 0))
    voxels = center_shift(voxels)
    voxels -= voxels.mean()
    
    registrator = Registrator(n_spherical=72, n_inplane=72, l_max=15, k_res=2).load(voxels, thresh=0.8)
    prediction = tuple(registrator.align(coordinates, sigma=1.0).values())

    assert np.allclose(prediction, angles, atol=5), f"Expected {angles}, got {prediction}"

@pytest.mark.skip(reason="This test is for 2D densities, which are not yet working")
def test_roundtrip_2d(pdb: str = "3uat.pdb"):
    coordinates = np.stack([atom.coordinates for atom in AtomLine.from_pdb(DATA / pdb)])

    xmipp = Xmipp(DATA / "test_data")
    mrcs_path, angles_path = xmipp.simulate(DATA / pdb, 5, 10)
    pixels_stack = mrcfile.read(mrcs_path)
    angles = starfile.read(angles_path)
    for pixels,angles in zip(pixels_stack,angles[['anglePsi','angleTilt','angleRot']].itertuples(False,None)):
        pixels = center_shift(pixels)
        pixels -= pixels.mean()
        registrator = Registrator(n_spherical=72, n_inplane=72, l_max=15, k_res=2).load_2d(pixels, thresh=0.8)
        prediction = tuple(registrator.align(coordinates, sigma=1.0).values())

        assert np.allclose(np.asarray(prediction), np.asarray(angles), atol=5), f"Expected {angles}, got {prediction}"
