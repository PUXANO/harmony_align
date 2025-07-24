'''
Module to compute volume decompositions
'''
from typing import Self
from dataclasses import dataclass, field

import numpy as np

import torch
from torch import Tensor
from scipy.special import sph_harm_y, spherical_jn

from tools.interp1d import interp1d

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SQRT_2_DIV_PI = torch.sqrt(torch.tensor(2 / torch.pi)).to(DEVICE)
SQRT2 = torch.sqrt(torch.tensor(2.0)).to(DEVICE)

def from_numpy(array: Tensor) -> Tensor:
    '''Convert a numpy array to a torch tensor, ensuring it is on the correct device.'''
    if isinstance(array, Tensor):
        return array.to(DEVICE)
    return torch.from_numpy(array).to(DEVICE).float()

def as_tensor(array) -> Tensor:
    '''Convert a numpy array to a torch tensor, ensuring it is on the correct device.'''
    if isinstance(array, Tensor):
        return array
    return torch.tensor(array, device=DEVICE)


def torch_Ylm(l: int, m: int, theta: Tensor, phi: Tensor) -> Tensor:
    '''
    Compute spherical harmonics Ylm at angles theta and phi.
    
    NOTE Associated Legendre polynomials are computed using scipy's sph_harm_y, thus on CPU.
    '''
    Yl0 = from_numpy(np.real(sph_harm_y(l,abs(m),theta.cpu().numpy(), 0.0))) 
    match m:
        case 0:
            return Yl0
        case _ if m > 0:
            return SQRT2 * Yl0 * torch.cos(m * phi)
        case _ if m < 0:
            return SQRT2 * Yl0 * torch.sin(-m * phi)
        
def torch_jn(n: int, r: Tensor) -> Tensor:
    '''
    Compute spherical Bessel function of the first kind jn at radius r.
    
    NOTE This uses scipy's spherical_jn, thus on CPU.
    '''
    return from_numpy(spherical_jn(n, r.cpu().numpy()))

class InterpYl0:
    '''
    Interpolator for spherical harmonics Yl0, which are real-valued.
    '''
    
    def __init__(self, l_max: int, theta: Tensor | int):
        if isinstance(theta, int):
            theta = torch.linspace(0, np.pi, theta, device=DEVICE)
        self.l_max = l_max
        self.theta = theta
        self.values = [torch_Ylm(l,0,theta,0.0) for l in range(l_max + 1)]

    def __call__(self, l:int, theta: Tensor) -> Tensor:
        return interp1d(self.theta, self.values[l], theta.reshape(-1)).reshape(theta.shape)
    
    def Ylm(self, l, m, theta: Tensor, phi: Tensor) -> Tensor:
        '''
        Compute spherical harmonics Ylm at angles theta.
        
        NOTE Associated Legendre polynomials are computed using scipy's sph_harm_y, thus on CPU.
        '''
        Yl0 = self(l, theta)
        match m:
            case 0:
                return Yl0
            case _ if m > 0:
                return SQRT2 * Yl0 * torch.cos(m * phi)
            case _ if m < 0:
                return SQRT2 * Yl0 * torch.sin(-m * phi)
    
class InterpJn:
    '''
    Interpolator for spherical Bessel functions jn, which are real-valued.
    '''
    def __init__(self, n_max: int, r: Tensor):
        self.n_max = n_max
        self.r = r
        self.values = [torch_jn(n,r) for n in range(n_max + 1)]

    def __call__(self, n: int, r: Tensor) -> Tensor:
        return interp1d(self.r, self.values[n], r.reshape(-1)).reshape(r.shape)


@dataclass
class Grid:
    x: Tensor
    y: Tensor
    z: Tensor = None

    grid: Tensor = None
    r: Tensor = None
    theta: Tensor = None
    phi: Tensor = None
    dV: Tensor = None

    @classmethod
    def regular(cls, start = -50, stop = 49, num = 100) -> Self:
        ticks = torch.linspace(start,stop,num, device=DEVICE)
        return cls(*torch.meshgrid(ticks,ticks,ticks,indexing='ij'))
    
    def __post_init__(self):
        if self.z is not None:
            self.grid = torch.stack([self.x,self.y,self.z],-1)
        else:
            self.grid = torch.stack([self.x,self.y],-1)
        self.r = torch.linalg.norm(self.grid,axis=-1)
        self.theta = torch.arccos(self.z / self.r_reg) if self.z is not None else torch.pi / 2
        self.phi = torch.atan2(self.y,self.x)
        grid = self.grid.reshape((-1,self.grid.shape[-1]))
        self.dV = torch.prod(grid.max(axis=0).values - grid.min(axis=0).values) / len(grid)

        self.cached_Ylm: dict[tuple[int,int],Tensor] = {}
        self.cached_jn: dict[int,Tensor] = {}

    @property
    def r_reg(self) -> Tensor:
        return torch.clip(self.r,1.e-8,None)
    
    def Ylm(self,l,m) -> Tensor:
        res = self.cached_Ylm.get((l,m),torch_Ylm(l,m,self.theta,self.phi))
        if (l,m) not in self.cached_Ylm:
            self.cached_Ylm[(l,m)] = res
        return res
    
    def jn(self,n) -> Tensor:
        res = self.cached_jn.get(n,torch_jn(n,self.r))
        if n not in self.cached_jn:
            self.cached_jn[n] = res
        return res
    
    def fourier_bessel_expansion(self,l, m, k: Tensor) -> Tensor:
        '''
        compute the plane wave expansion of this Grid at wavevector k

        Assuming k is a 1D array of frequencies, 
        the resulting shape is a cartesion product of Grid and k
        '''
        return torch_jn(l, self.r[...,None] * k) * self.Ylm(l,m)[...,None] * k * SQRT_2_DIV_PI
    
@dataclass
class IGrid(Grid):
    '''
    Grid with a fixed radius, used for spherical harmonics and Bessel functions.
    '''
    l_max: int = 3

    def __post_init__(self):
        super().__post_init__()
        grid_size = len(self.r)
        n_theta = 2 * grid_size
        kr = 2 * torch.pi * torch.linspace(0, grid_size // 2, grid_size, device=DEVICE)

        self.interpYl0 = InterpYl0(self.l_max, n_theta)
        self.interpJn = InterpJn(self.l_max, kr)

    def fourier_bessel_expansion_interp(self,l, m, grid: Grid, k: Tensor) -> Tensor:
        '''
        compute the plane wave expansion of this Grid at wavevector k

        Assuming k is a 1D array of frequencies, 
        the resulting shape is a cartesion product of Grid and k
        '''
        return self.interpJn(l, grid.r[...,None] * k) * self.interpYl0.Ylm(l,m,grid.theta, grid.phi)[...,None] * k * SQRT_2_DIV_PI