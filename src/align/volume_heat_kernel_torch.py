'''

'''


from typing import Generator, Self
from time import time
from pathlib import Path

import torch
from torch import Tensor
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from tools.wigner import WignerGalleryTorch
from tools.utils_torch import Grid, IGrid, from_numpy, as_tensor, DEVICE
from tools.sequences import SphericalSequence

def euler_to_tensor(euler: tuple | Tensor) -> Tensor:
    '''Convert Euler angles in ZYZ convention to a rotation matrix tensor.'''
    return from_numpy(Rotation.from_euler('ZYZ', euler, degrees=True).as_matrix())

class Spatial(SphericalSequence, Grid):
    def __init__(self, grid_size = 100, scale = 1.0, l_max: int = 10):
        SphericalSequence.__init__(self, l_max)
        bound = (grid_size - 1) / 2 * scale
        ticks = torch.linspace(-bound,bound,grid_size, device = DEVICE)
        Grid.__init__(self,*torch.meshgrid(ticks,ticks,ticks, indexing='ij'))
        self.k = [2 * torch.pi / grid_size * ticks] * (l_max + 1)

    def Slm(self,voxels: Tensor, k_profile: Tensor) -> Generator[Tensor,None,None]:
        '''Eigenfunctions for the volume'''
        for l,m in self.lm():
            yield torch.sum(self.fourier_bessel_expansion(l,m,k_profile[l]) * voxels[...,None] * self.dV,dim=(0,1,2))

    def Sl(self,voxels: Tensor, k_profile: Tensor) -> Generator[Tensor,None,None]:
        multiplet = []
        for (l,m),Slm in zip(self.lm(),self.Slm(voxels, k_profile)):
            multiplet.append(Slm)
            if l == m:
                yield torch.stack(multiplet)
                multiplet = []

    def volume_map(self,coordinates: Tensor,sigma: float) -> Tensor:
        # return torch.exp(-0.5 / sigma**2 * torch.sum((self.grid[...,None,:] - coordinates)**2,-1)).sum(-1)
        res = torch.zeros_like(self.r)
        for atom in coordinates:
            res += torch.exp(-0.5 / sigma**2 * torch.sum((self.grid - atom)**2,-1))
        return res
    
    def Vlm(self, coordinates: Tensor, sigma: float, k_profile: Tensor, k_density: Tensor = None) -> Generator[Tensor,None,None]:
        coordinates_as_grid = Grid(*coordinates.T)
        for l,m in self.lm():
            # TODO precompute k-density
            density = torch.exp(- self.k[l] **2 * sigma**2 / 2) if k_density is None else k_density[l]
            yield torch.sum(coordinates_as_grid.fourier_bessel_expansion(l,m,k_profile[l]) * density,dim=0) #should be 1D

    def Vl(self, coordinates: Tensor, sigma: float, k_profile: Tensor, k_density: Tensor = None) -> Generator[Tensor,None,None]:
        multiplet = []
        for (l,m),Vlm in zip(self.lm(),self.Vlm(coordinates,sigma, k_profile, k_density)):
            multiplet.append(Vlm)
            if l == m:
                yield torch.stack(multiplet,dim=0)
                multiplet = []

    def l(self) -> Generator[int,None,None]:
        '''Generate l for the spherical harmonics'''
        for l in range(self.l_max + 1):
            yield l

class Registrator(Spatial):
    '''
    Object to register a given set of 3D coordinates as a spherical volume
    '''
    def __init__(self, 
                 voxels: Tensor = None,
                 grid_size = 100, 
                 scale = 1.0, 
                 l_max: int = 10,
                 n_spherical: int = 36,
                 n_inplane: int = 36,
                 k_res: int = 1):
        super().__init__(grid_size, scale, l_max)
        self.k_profile = [2 * torch.pi / grid_size * torch.linspace(0, grid_size // 2, (grid_size//2)*k_res + 1, device=DEVICE)] * (l_max + 1)
        self.k_density = [torch.exp(-self.k[l] **2 / 2).to(DEVICE) for l in range(l_max + 1)]
        self.voxels = voxels
        self.sl = None
        self.gallery = WignerGalleryTorch(n_spherical, n_inplane,l_max)
        self.preprocessed = None

        self._latest_correlations = None # temporary storage for the latest correlations

    def set_reference(self, reference_coordinates: Tensor = None, reference_rotation: Tensor | tuple = torch.eye(3), voxels: Tensor = None) -> Self:
        '''
        Set the reference coordinates and rotation for the registration
        '''
        try:
            if voxels is not None:
                self.voxels = voxels
                return self
            
            assert reference_coordinates is not None, "Either voxels or reference_coordinates must be provided"

            if isinstance(reference_rotation, tuple):
                reference_rotation = euler_to_tensor(reference_rotation)
            self.voxels = self.volume_map(reference_coordinates @ reference_rotation.T, 1.0)
            
            return self
        finally:
            if self.voxels is not None:
                self.sl = list(self.Sl(self.voxels, self.k_profile))
    
    def relevant_k(self, thresh: float | int) -> list[tuple[int,int, int]]:
        '''
        Filter the heat kernel eigenvalues per rotational mode by relevance to the given voxels
        '''
        if self.voxels is None:
            raise ValueError("No voxels set for filtering")
        idx = np.array([(l,m,i_k) for l,m in self.lm() for i_k, k in enumerate(self.k_profile[l])])
        moments = torch.concatenate([torch.abs(s.flatten()) for s in self.sl])
        permutation = torch.argsort(moments).tolist()[::-1]
        idx = idx[permutation]
        moments = moments[permutation]
        if isinstance(thresh, float):
            mask = torch.cumsum(moments ** 2,0) < thresh * (moments @ moments)
            thresh = torch.sum(mask).item()
        return idx[:thresh]
    
    def filter_k(self, thresh: float | int) -> Self:
        '''
        Filter the heat kernel eigenvalues per rotational mode by relevance to the given voxels
        '''
        relevant_modes = self.relevant_k(thresh)
        k_profile = [list() for _ in range(self.l_max + 1)]
        k_density = [list() for _ in range(self.l_max + 1)]
        sl = [list() for _ in range(self.l_max + 1)]
        for l,_,i_k in relevant_modes:
            k_profile[l].append(self.k_profile[l][i_k])
            k_density[l].append(self.k_density[l][i_k])
            sl[l].append(self.sl[l][:,i_k])
        self.k_profile = [as_tensor(k_l) for k_l in k_profile]
        self.k_density = [as_tensor(d_l) for d_l in k_density]
        self.sl = [torch.stack(ms_per_k,dim=-1) if ms_per_k else torch.empty(1,1,device = DEVICE) for ms_per_k in sl]
        return self
    
    def preprocess(self) -> Self:
        '''
        Preprocess the voxel moments per rotation, inplace operation
        '''
        self.preprocessed = [torch.einsum('gmw,wk->gmk',
                                       gallery,
                                       sl) for gallery,sl in zip(self.gallery.matrices,self.Sl(self.voxels, self.k_profile))]

        return self

    def correlations(self, coordinates: Tensor, sigma: float = 1.0) -> Tensor:
        '''
        Compute the correlations between the atom coordinate moments and the preprocessed volume moments

        TODO The gaussian density factor / heat kernel eigenvalue could be included in the preprocessed moments.
        '''
        if self.preprocessed is None:
            raise ValueError("Preprocessing not done, call preprocess() first")
        correlation = sum([torch.einsum('mk,gmk->g',Vlm,prep) for Vlm, prep in zip(self.Vl(coordinates,sigma, self.k_profile, self.k_density),self.preprocessed)])
        return correlation

    def align(self, coordinates: Tensor, sigma=1.0):
        ''' 
        Align the given coordinates to the reference volume using the preprocessed moments.
        Returns the best fit rotation from the gallery.
        '''
        self._latest_correlations = self.correlations(coordinates, sigma)
        best_fit = torch.argmax(torch.abs(self._latest_correlations),dim=0)
        return self.gallery[best_fit]
    
    def correlation_frame(self, labels=['correlation']) -> pd.DataFrame:
        '''
        Postprocess the registration results, yielding rotations for each coordinate set
        '''
        if self._latest_correlations is None:
            print("No correlations computed yet, call align() first")
            return pd.DataFrame()

        gallery_angles = [','.join([str(int(angle)) for angle in self.gallery[i].values()]) for i in range(len(self.gallery))]
        
        return pd.DataFrame(torch.abs(self._latest_correlations).cpu().numpy(),
                            index = gallery_angles,
                            columns = labels)

def prepare_3d(pdb: Path, rotation: tuple[float,float,float]) -> Path:
    '''
    Prepare a 3D structure from a PDB file, applying the given rotation.
    '''
    from tools.xmipp import Xmipp, AtomLine
    coordinates = np.stack([atom.coordinates for atom in AtomLine.from_pdb(pdb)])
    coordinates -= coordinates.mean(axis=0)  # Center the coordinates
    rotation_matrix = Rotation.from_euler('ZYZ',rotation,degrees=True).as_matrix()
    target_folder = pdb.parent / pdb.stem
    target_folder.mkdir(exist_ok=True)
    approximation = coordinates @ rotation_matrix.T
    Xmipp(target_folder).volume_from_pdb(pdb, 'reference')
    AtomLine.to_pdb(approximation, target_folder / 'approximation.pdb')

    return target_folder

if __name__ == '__main__':
    from tools.xmipp import AtomLine
    from pathlib import Path
    from tools.spider_files3 import open_volume
    print("Preparing data...")
    prapare_path = prepare_3d(Path('/mnt/proj1/dd-24-27/Zernike/data/3uat.pdb'), (135.0, 45.0, 120.0))
    coordinates = from_numpy(np.stack([atom.coordinates for atom in AtomLine.from_pdb(prapare_path / "approximation.pdb")]))
    voxels = from_numpy(open_volume(prapare_path / 'reference.vol'))
    print(f"Coordinates shape: {coordinates.shape}, Voxels shape: {voxels.shape}")

    print("Preprocessing...")
    t0 = time()
    registrator = Registrator(n_spherical=8, n_inplane=8, l_max = 15)
    registrator.set_reference(voxels = voxels)
    registrator.filter_k(thresh=0.8).preprocess()
    print(f"Preprocessing done in {time() - t0:.2f} seconds")

    print("Aligning...")
    t1 = time()
    for rotation in registrator.align(coordinates, sigma=1.0).items():
        print(rotation)
    print(f"Alignment done in {time() - t1:.2f} seconds")

    print(registrator.correlation_frame().to_string(index=True, float_format=lambda x: f"{x:.4f}"))
    