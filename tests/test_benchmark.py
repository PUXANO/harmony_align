'''
Tests of bulk cases containing error metrics including runtime and deviation.
'''

from typing import Generator, Self
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import mrcfile

from tools.xmipp import AtomLine, Xmipp
# from tests import DATA
from tools.spider_files3 import open_volume as open_spider_volume
from align.volume_heat_kernel import Registrator
from tools.center import center_shift, np_shift

np.set_printoptions(precision=2, suppress=True)
DATA = Path(__file__).parents[1] / "data"

def rot_str(self) -> str:
    '''
    Generate a random rotation string in the format 'Z Y Z'.
    '''
    return ','.join([f'{angle:.0f}' for angle in self.as_euler('ZYZ', degrees=True)])

def open_volume(path: Path) -> tuple[np.ndarray,np.ndarray] :
    '''
    Open a volume file and return its data in xyz axis order.
    '''
    match path:
        case Path(suffix='.mrcs') | Path(suffix='.gz'):
            with mrcfile.open(path, mode='r') as mrc:
                return mrc.data.transpose((2, 1, 0)), np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        case Path(suffix='.vol'):
            return open_spider_volume(path).transpose((2, 1, 0)), np.ones(3) #TODO get resolution from header
        
    raise ValueError(f"Unsupported file format: {path.suffix}")

# setattr(Rotation, '__str__', rot_str)

@contextmanager
def no_seed_update(temp_seed = None) -> Generator[None, None, None]:
    '''
    A context not updating the seed.'''
    state = np.random.get_state()
    np.random.seed(temp_seed)  # For reproducibility
    try:
        yield
    finally:
        np.random.set_state(state)

def random_rotation() -> Rotation:
    '''
    Generate a random rotation in quaternion format.
    '''
    quat = np.random.randn(4)
    quat /= np.linalg.norm(quat)
    return Rotation.from_quat(quat)

@dataclass
class Benchmark:
    pdb: str
    n_atoms: int
    rotation: str
    prediction: str
    deviation: float
    runtime: float
    prep_time: float
    sym_axis: np.ndarray

    box_size: int = 100
    voxel_size: float = 1.0
    sigma: float = 1.0

    @classmethod
    def create(cls, pdb: Path, volume: Path = None, N: int = 1, sigma = 1.0, **smpling_opts) -> Generator[Self,None,None]:
        '''
        Benchmark the alignment of a density derived from a PDB file.
        '''
        coordinates = np.stack([atom.coordinates for atom in AtomLine.from_pdb(pdb)])
        coordinate_shift = coordinates.mean(axis=0)
        coordinates -= coordinate_shift

        if volume is None:
            # Simulate data
            with TemporaryDirectory(dir = DATA) as tmpdir:
                xmipp = Xmipp(Path(tmpdir))
                volume_path = xmipp.volume_from_pdb(coordinates)
                voxels,voxel_size = open_volume(volume_path)

        else:
            # Load existing data
            try:
                voxels, voxel_size = open_volume(volume)
            except:
                print(f"{pdb.stem} -- failed to open volume {volume}")
                return

        # voxel_shift = (np.asarray(voxels.shape) // 2 - coordinate_shift / voxel_size).round(0).astype(int)
        # voxels = np_shift(voxels, voxel_shift, axis=(0,1,2))
        voxels = center_shift(voxels)
        voxels -= voxels.mean()

        grid_coordinates = coordinates / voxel_size
        sigma /= voxel_size

        smpling_opts = dict(n_spherical=72, n_inplane=72, l_max=8, k_res=2) | smpling_opts
        if not voxels.shape[0] == voxels.shape[1] == voxels.shape[2]: 
            print(f"{pdb.stem} -- Non-cubic volume: extend Grid class to handle this")
            return
        try:
            registrator = Registrator(**smpling_opts, grid_size = voxels.shape[0]).load(voxels, thresh=0.8)
        except:
            print(f"{pdb.stem} -- preprocessing failed")
            return
        for _ in range(N):
            try:
                rotation = random_rotation()
                rotated_grid_coordinates = grid_coordinates @ rotation.as_matrix()
                prediction = registrator.align(rotated_grid_coordinates, sigma=sigma)
                print(f"Aligning {pdb.stem} with rotation {rot_str(rotation)}")
                prediction_rotation = Rotation.from_euler('ZYZ', tuple(prediction.values()), degrees=True)
                print(f"Prediction rotation: {rot_str(prediction_rotation)}")
                deviation = np.degrees((net := rotation * prediction_rotation.inv()).magnitude())

                yield Benchmark(pdb = pdb.stem,
                                n_atoms = len(coordinates),
                                rotation = rot_str(rotation),
                                prediction = rot_str(prediction_rotation),
                                sym_axis= net.as_rotvec(degrees=True),
                                deviation = deviation,
                                runtime = registrator._latest_timings['align'],
                                prep_time = registrator._latest_timings['load'],
                                voxel_size = voxel_size[0],
                                box_size = registrator.r.shape[0],
                                sigma = sigma)
            except:
                print(f"{pdb.stem} -- alignment failed")
                break

def test_benchmark():
    '''
    Run the benchmark test.
    '''
    benchmark_data = pd.read_csv(DATA / "fetch_sample.csv")
    np.random.seed(42)  # For reproducibility

    benchmark_files = benchmark_data[['pdb_path', 'volume_path']].map(lambda pth: DATA / Path(pth))

    result = pd.DataFrame([benchmark for pdb, vol in benchmark_files.itertuples(False) for benchmark in Benchmark.create(pdb, vol, N=5, l_max=8, k_res=0.5)])
    print(result)
    result.to_csv(DATA / "benchmark.csv", index=False)
    if __name__ == 'tests.test_benchmark':
        assert result.deviation.max() < 7.0, "Maximum deviation exceeds threshold"

if __name__ == '__main__':
    test_benchmark()
    print("Benchmark test completed.")
