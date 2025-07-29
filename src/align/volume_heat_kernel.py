'''

'''


from typing import Generator, Self
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd

from tools.wigner import WignerGallery
from tools.utils import Grid, jn_zeros
from tools.sequences import SphericalSequence

class Spatial(SphericalSequence, Grid):
    def __init__(self, grid_size = 100, scale = 1.0, l_max: int = 10):
        SphericalSequence.__init__(self, l_max)
        bound = (grid_size - 1) / 2 * scale
        ticks = np.linspace(-bound,bound,grid_size)
        Grid.__init__(self,*np.meshgrid(ticks,ticks,ticks, indexing='ij'))
        self.k = [2 * np.pi / grid_size * ticks] * (l_max + 1)

    def Slm(self,voxels: np.ndarray, k_profile: np.ndarray) -> Generator[np.ndarray,None,None]:
        '''Eigenfunctions for the volume'''
        for l,m in self.lm():
            yield np.sum(self.fourier_bessel_expansion(l,m,k_profile[l]) * voxels[...,None] * self.dV,axis=(0,1,2))

    def Sl(self,voxels: np.ndarray, k_profile: np.ndarray) -> Generator[np.ndarray,None,None]:
        multiplet = []
        for (l,m),Slm in zip(self.lm(),self.Slm(voxels, k_profile)):
            multiplet.append(Slm)
            if l == m:
                yield np.stack(multiplet)
                multiplet = []

    def volume_map(self,coordinates: np.ndarray,sigma: float) -> np.ndarray:
        # return np.exp(-0.5 / sigma**2 * np.sum((self.grid[...,None,:] - coordinates)**2,-1)).sum(-1)
        res = np.zeros_like(self.r)
        for atom in coordinates:
            res += np.exp(-0.5 / sigma**2 * np.sum((self.grid - atom)**2,-1))
        return res
    
    def Vlm(self, coordinates: np.ndarray, sigma: float, k_profile: np.ndarray, k_density: np.ndarray = None) -> Generator[np.ndarray,None,None]:
        coordinates_as_grid = Grid(*coordinates.T)
        for l,m in self.lm():
            # TODO precompute k-density
            density = np.exp(- self.k[l] **2 * sigma**2 / 2) if k_density is None else k_density[l]
            yield np.sum(coordinates_as_grid.fourier_bessel_expansion(l,m,k_profile[l]) * density,axis=0) #should be 1D

    def Vl(self, coordinates: np.ndarray, sigma: float, k_profile: np.ndarray, k_density: np.ndarray = None) -> Generator[np.ndarray,None,None]:
        multiplet = []
        for (l,m),Vlm in zip(self.lm(),self.Vlm(coordinates,sigma, k_profile, k_density)):
            multiplet.append(Vlm)
            if l == m:
                yield np.array(multiplet)
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
                 voxels: np.ndarray = None,
                 grid_size = 100, 
                 scale = 1.0, 
                 l_max: int = 10,
                 n_spherical: int = 36,
                 n_inplane: int = 36,
                 k_res : int = 1):
        super().__init__(grid_size, scale, l_max)
        self.k_profile = self.get_k_profile(grid_size, l_max, k_res)
        self.k_density = [np.exp(-self.k[l] **2 / 2) for l in range(l_max + 1)]
        self.voxels = voxels
        self.sl = list(self.Slm(voxels, self.k_profile)) if self.voxels is not None else []
        self._gallery = (n_spherical, n_inplane,l_max)
        self.preprocessed = None

        self._latest_correlations = None # temporary storage for the latest correlations

    @property
    def gallery(self) -> WignerGallery:
        '''create on the fly since QArrays are not serializable for parallelization '''
        return WignerGallery(*self._gallery)

    @classmethod
    def get_k_profile(cls, grid_size: int, l_max: int, k_res: int) -> list[np.ndarray]:
        '''
        Get the k-profile for the spherical harmonics. 
        
        Pick k's that set the spherical Bessel functions to zero at the spherical grid boundary.

        Returns k_res * (grid_size // 2) k-values per l.
        '''
        r_max = (grid_size // 2)
        return list(jn_zeros(l_max, k_res * (grid_size // 2)) / r_max)

    def set_reference(self, reference_coordinates: np.ndarray = None, reference_rotation: np.ndarray | tuple = np.eye(3), voxels: np.ndarray = None) -> Self:
        '''
        Set the reference coordinates and rotation for the registration
        '''
        try:
            if voxels is not None:
                self.voxels = voxels
                return self
            
            assert reference_coordinates is not None, "Either voxels or reference_coordinates must be provided"

            if isinstance(reference_rotation, tuple):
                reference_rotation = Rotation.from_euler('ZYZ', reference_rotation, degrees=True).as_matrix()
            self.voxels = self.volume_map(reference_coordinates @ reference_rotation.T, 1.0)
            
            return self
        finally:
            if self.voxels is not None:
                self.sl = self.Sl_parallel(self.k_profile)
    
    def relevant_k(self, thresh: float | int) -> list[tuple[int,int, int]]:
        '''
        Filter the heat kernel eigenvalues per rotational mode by relevance to the given voxels
        '''
        if self.voxels is None:
            raise ValueError("No voxels set for filtering")
        idx = np.array([(l,m,i_k) for l,m in self.lm() for i_k, k in enumerate(self.k_profile[l])])
        moments = np.concatenate([np.abs(s.flatten()) for s in self.sl])
        permutation = np.argsort(moments)[::-1]
        idx = idx[permutation]
        moments = moments[permutation]
        if isinstance(thresh, float):
            mask = np.cumsum(moments ** 2) < thresh * (moments @ moments)
            thresh = np.sum(mask)
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
        self.k_profile = k_profile
        self.k_density = k_density
        self.sl = [np.stack(ms_per_k,axis=-1) 
                   if ms_per_k else np.zeros((2*l+1,len(self.k_profile[l]))) 
                   for l,ms_per_k in enumerate(sl)]
        return self
    
    def preprocess(self) -> Self:
        '''
        Preprocess the voxel moments per rotation, inplace operation
        '''
        self.preprocessed = [np.einsum('gmw,wk->gmk',
                                       gallery,
                                       sl) for gallery,sl in zip(self.gallery.matrices,self.Sl(self.voxels, self.k_profile))]

        return self
    
    def Sl_m(self, l:int, k_profile_l: np.ndarray) -> list[np.ndarray]:
       return np.stack([np.sum(self.fourier_bessel_expansion(l,m,k_profile_l) * self.voxels[...,None] * self.dV,axis=(0,1,2)) 
                        for m in range(-l, l + 1)], axis=0)

    def Sl_parallel(self, k_profile: np.ndarray) -> list[np.ndarray]:
        '''
        Compute the spherical harmonics for the given voxels in parallel.
        Returns a list of multiplets, one for each l.
        '''
        res = [None] * (self.l_max + 1)
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(Registrator.Sl_m,self, l, k_prof): l for l, k_prof in enumerate(k_profile)}
            for future in as_completed(futures):
                l = futures[future]
                res[l] = future.result()
        return res

    def correlations(self, coordinates: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        '''
        Compute the correlations between the atom coordinate moments and the preprocessed volume moments

        TODO The gaussian density factor / heat kernel eigenvalue could be included in the preprocessed moments.
        '''
        if self.preprocessed is None:
            raise ValueError("Preprocessing not done, call preprocess() first")
        correlation = np.sum([np.einsum('mk,gmk->g',Vlm,prep) for Vlm, prep in zip(self.Vl(coordinates,sigma, self.k_profile, self.k_density),self.preprocessed)],axis=0)
        return correlation

    def align(self, coordinates: np.ndarray, sigma=1.0):

        self._latest_correlations = self.correlations(coordinates, sigma)
        best_fit = np.argmax(np.abs(self._latest_correlations),axis=0)
        return self.gallery[best_fit]
    
    def correlation_frame(self, labels=['correlation']) -> pd.DataFrame:
        '''
        Postprocess the registration results, yielding rotations for each coordinate set
        '''
        if self._latest_correlations is None:
            print("No correlations computed yet, call align() first")
            return pd.DataFrame()

        gallery_angles = [','.join([str(int(angle)) for angle in self.gallery[i].values()]) for i in range(len(self.gallery))]
        
        return pd.DataFrame(np.abs(self._latest_correlations),
                            index = gallery_angles,
                            columns = labels)

if __name__ == '__main__':
    from tools.xmipp import AtomLine
    from pathlib import Path
    from tools.spider_files3 import open_volume
    coordinates = np.stack([atom.coordinates for atom in AtomLine.from_pdb(Path(__file__).parent / "data" / "prepared2" / "approximation.pdb")])
    voxels = open_volume(Path(__file__).parent / "data" / "prepared2" / 'reference.vol')

    print("Preprocessing...")
    t0 = time()
    registrator = Registrator(n_spherical=8, n_inplane=8, l_max = 3)
    registrator.set_reference(voxels = voxels)
    registrator.filter_k(thresh=0.8).preprocess()
    print(f"Preprocessing done in {time() - t0:.2f} seconds")

    print("Aligning...")
    t1 = time()
    for rotation in registrator.align(coordinates, sigma=1.0).items():
        print(rotation)
    print(f"Alignment done in {time() - t1:.2f} seconds")

    print(registrator.correlation_frame().to_string(index=True, float_format=lambda x: f"{x:.4f}"))
    