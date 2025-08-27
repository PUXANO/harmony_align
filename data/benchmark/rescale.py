'''
Module to rescale volumes to a cubic box of size close to 100.
'''

from pathlib import Path

from scipy.ndimage import zoom, gaussian_filter
import mrcfile
import numpy as np

def resize(volume_path: Path | str, box_size: int):
    '''Resize the volume in volume_path to a cubic box of size box_size.'''
    with mrcfile.open(volume_path, mode='r+') as mrc:
        factors = np.array([box_size / s for s in mrc.data.shape])
        voxel_size = tuple(getattr(mrc.voxel_size,axis) for axis in 'xyz')
        data = gaussian_filter(mrc.data, sigma=factors/2)  # Smooth before resizing to reduce artifacts
        new_data = zoom(data, zoom=factors, order=1, grid_mode=True, mode='grid-constant')
        mrc.set_data(new_data)
        mrc.voxel_size = tuple(vox / factor for factor, vox in zip(factors,voxel_size))
        mrc.update_header_from_data()

def resize_100(volume_path: Path | str):
    '''Resize the volume in volume_path to a cubic box of size close to 100.'''
    try:
        with mrcfile.open(volume_path, mode='r', header_only=True) as mrc:
            box_size = max(mrc.header.nx, mrc.header.ny, mrc.header.nz)
    except:
        return
    if box_size < 200:
        return
    divisors = [factor for factor in range(2,(box_size // 100) + 1) if box_size % factor == 0 and box_size < factor * 200]
    if divisors:
        resize(volume_path, box_size // divisors[-1])
    else:
        resize(volume_path, 100)
    print(f"Resized {volume}")


if __name__ == '__main__':
    for volume in Path(__file__).parent.glob('*'):
        try:
            resize_100(volume)
        except Exception as e:
            print(e)
            print(f"Failed to resize {volume.stem}")
            continue
