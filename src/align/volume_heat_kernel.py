'''
Module to decompose a density into its moments in a spherical-bessel expansion and compare them to the corresponding moments of a Gaussian blur of discrete (atom) positions.

The main class is `Registrator`: the problem at hand is often referred to as `3D registration`. The moments from the volume are referred to as `Slm` (spherical harmonics) and the moments from the coordinates as `Vlm` (volume moments). The `Registrator` class can be used to preprocess a volume, filter the relevant modes, and align a set of coordinates to the volume by finding the best rotation.

(c) 2025 Puxano BV, licensed under the Apache License 2.0.
'''


from typing import Generator, Self
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count
from time import time

import numpy as np
import pandas as pd

from tools.wigner import WignerGallery
from tools.utils import Grid, jn_zeros
from tools.sequences import SphericalSequence

class SphericalGrid(SphericalSequence, Grid):
    '''
    Grid allowing decomposition in spherical harmonics and spherical Bessel functions.
    '''
    def __init__(self, grid_size = 100, scale = 1.0, l_max: int = 10):
        SphericalSequence.__init__(self, l_max)
        bound = (grid_size - 1) / 2 * scale
        ticks = np.linspace(-bound,bound,grid_size)
        Grid.__init__(self,*np.meshgrid(ticks,ticks,ticks, indexing='ij'))

    def voxel_moment(self, l: int, m: int, voxels: np.ndarray, k_profile_l: np.ndarray) -> np.ndarray:
        '''
        Compute the spherical harmonic moment for the given voxel grid.
        '''
        return np.sum(self.fourier_bessel_expansion(l,m,k_profile_l) * voxels[...,None] * self.dV,axis=(0,1,2))

    def voxel_moments_lm(self,voxels: np.ndarray, k_profile: np.ndarray) -> Generator[np.ndarray,None,None]:
        '''Moments for the given voxel distribution over this Grid.'''
        for l,m in self.lm():
            yield self.voxel_moment(l,m,voxels,k_profile[l])

    def voxel_moments_l(self,voxels: np.ndarray, k_profile: np.ndarray) -> Generator[np.ndarray,None,None]:
        multiplet = []
        for (l,m),Slm in zip(self.lm(),self.voxel_moments_lm(voxels, k_profile)):
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
    
    def coordinate_moments_lm(self, coordinates: np.ndarray, sigma: float, k_profile: np.ndarray, k_density: np.ndarray = None) -> Generator[np.ndarray,None,None]:
        '''Generate all moments from the coordinates in sequence-order, for each l and m.'''
        coordinates_as_grid = Grid(*coordinates.T)
        for l,m in self.lm():
            density = np.exp(- k_profile[l] **2 * sigma**2 / 2) if k_density is None else k_density[l]
            yield np.sum(coordinates_as_grid.fourier_bessel_expansion(l,m,k_profile[l]) * density,axis=0) #should be 1D

    def coordinate_moments_l(self, coordinates: np.ndarray, sigma: float, k_profile: np.ndarray, k_density: np.ndarray = None) -> Generator[np.ndarray,None,None]:
        multiplet = []
        for (l,m),Vlm in zip(self.lm(),self.coordinate_moments_lm(coordinates,sigma, k_profile, k_density)):
            multiplet.append(Vlm)
            if l == m:
                yield np.array(multiplet)
                multiplet = []

class SphericalGridParallel(SphericalGrid):
    '''
    SphericalGrid with parallelized computation of the (precomputed) volume moments.
    '''
    def __init__(self, grid_size = 100, scale = 1.0, l_max: int = 10):
        super().__init__(grid_size, scale, l_max)
        self._parallel_attributes = set(self.__dict__)

    def voxel_moments_lm(self, voxels: np.ndarray, k_profile: np.ndarray) -> list[np.ndarray]:
        '''
        Compute the spherical harmonics for the given voxels in parallel.
        Returns a list of multiplets, one (2l+1)-plet for each l.
        '''
        res = [[None]*(2*l+1) for l in range(self.l_max + 1)]

        # NOTE the serialization cost is significant, so more workers can turn out slower.
        with ProcessPoolExecutor(max_workers=cpu_count() - 2) as executor:
            futures = {executor.submit(SphericalGrid.voxel_moment,self, l, m, voxels, k_prof): (l,m) 
                       for l, k_prof in enumerate(k_profile) for m in range(-l, l + 1)}
            for future in as_completed(futures):
                l, m = futures[future]
                res[l][m+l] = future.result()
        return [np.stack(multiplet,axis=0) for multiplet in res]
    
    def __getstate__(self):
        '''
        Custom serialization to avoid extra pickling to the ProcessPoolExecutor: ignore all child attributes.
        '''
        if __name__ == "__main__":
            return super().__getstate__()
        child_attributes = set(self.__dict__) - self._parallel_attributes   
        return {k:self.__dict__.get(k) for k in self._parallel_attributes} | {k:None for k in child_attributes}

class Registrator(SphericalGridParallel):
    '''
    Object to register a given set of 3D coordinates as a spherical volume.
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
        self.k_density = [np.exp(-self.k_profile[l] **2 / 2) for l in range(l_max + 1)]
        self.voxels = voxels

        # cached variables
        self.gallery = WignerGallery(n_spherical, n_inplane,l_max)
        self._preprocessed = None
        self._volume_moments = list(self.voxel_moments_lm(voxels, self.k_profile)) if self.voxels is not None else []
        self._latest_correlations = None # temporary storage for the latest correlations

    @classmethod
    def get_k_profile(cls, grid_size: int, l_max: int, k_res: int) -> list[np.ndarray]:
        '''
        Get the k-profile for the radial grid, mapping to eigenvalues of the heat kernel.
        
        Pick k's that set the spherical Bessel functions to zero at the spherical grid boundary.

        Returns k_res * (grid_size // 2) k-values per l.
        '''
        r_max = (grid_size // 2)
        return list(jn_zeros(l_max, k_res * (grid_size // 2)) / r_max)

    def set_reference(self, voxels: np.ndarray) -> Self:
        '''
        Set the reference coordinates and rotation for the registration.

        This precomputes the volume moments for the given voxels.
        '''
        self.voxels = voxels
        self._volume_moments = self.voxel_moments_lm(voxels, self.k_profile)
        return self
    
    def relevant_k(self, thresh: float | int) -> list[tuple[int,int, int]]:
        '''
        Filter the heat kernel eigenvalues per rotational mode by relevance to the given voxels
        '''
        if self.voxels is None:
            raise ValueError("No voxels set for filtering")
        idx = np.array([(l,m,i_k) for l,m in self.lm() for i_k, k in enumerate(self.k_profile[l])])
        moments = np.concatenate([np.abs(s.flatten()) for s in self._volume_moments])
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
            sl[l].append(self._volume_moments[l][:,i_k])
        self.k_profile = k_profile
        self.k_density = k_density

        self._volume_moments = [np.stack(ms_per_k,axis=-1) 
                   if ms_per_k else np.zeros((2*l+1,len(self.k_profile[l]))) 
                   for l,ms_per_k in enumerate(sl)]
        
        return self
    
    def preprocess(self) -> Self:
        '''
        Preprocess the voxel moments per rotation, inplace operation
        '''
        self._preprocessed = [np.einsum('gmw,wk->gmk',
                                       gallery,
                                       sl) for gallery,sl in zip(self.gallery.matrices,self._volume_moments)]

        return self

    def load(self, voxels: np.ndarray, thresh: float = 0.8) -> Self:
        '''
        One-liner for all preprocessing steps: set reference, filter modes, and preprocess.
        '''
        try:
            t0 = time()
            return self.set_reference(voxels).filter_k(thresh).preprocess()
        finally:
            print(f"Preprocessing took {time() - t0:.2f} seconds")

    def correlations(self, coordinates: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        '''
        Compute the correlations between the atom coordinate moments and the preprocessed volume moments
        '''
        if self._preprocessed is None:
            raise ValueError("Preprocessing not done, call preprocess() first")
        correlation = np.sum([np.einsum('mk,gmk->g',Vlm,prep) for l, (Vlm, prep) in enumerate(zip(self.coordinate_moments_l(coordinates,sigma, self.k_profile, self.k_density),self._preprocessed)) if l > 0],axis=0)
        return correlation

    def align(self, coordinates: np.ndarray, sigma=1.0):
        '''perform alignment of this grid and voxel-densities with the given coordinates'''
        try:
            t0 = time()
            self._latest_correlations = self.correlations(coordinates, sigma)
            best_fit = np.argmax(np.abs(self._latest_correlations),axis=0)
            return self.gallery[best_fit]
        finally:
            print(f"Alignment took {time() - t0:.2f} seconds")

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

    