'''
Module to evaluate distributions over a grid, like Zernike plynomials or spherical harmonics.
'''
from typing import Self
from dataclasses import dataclass, field

import numpy as np
from scipy.special import sph_harm_y, spherical_jn, eval_jacobi
from scipy.optimize import brentq
from scipy.sparse import coo_array

## Zernike functionality - TODO remove? ##

def computeRadial(n: int, l: int, r: np.ndarray) -> np.ndarray:
    '''
    radial Zernike polynomials in 3D

    cfr. https://arxiv.org/pdf/0809.2368, eq (43)
    '''
    if n < l or (n-l) % 2:
        return np.zeros_like(r) if isinstance(r,np.ndarray) else 0.0
    a = (n-l) // 2
    q = l + 3/2
    return np.sqrt(2*n + 3) * r**l * eval_jacobi(a,0,q-1,2 * r **2 - 1)

def computeZernikes3D(l1, n, l2, m, pos, r_max = 1.0):

    # General variables
    pos_r = pos / r_max
    r = np.linalg.norm(pos_r, axis=-1)
    x, y, z = pos_r[...,0], pos_r[...,1], pos_r[...,2] 

    # Variables needed for l2 >= 5
    phi = np.atan2(y, x)
    r_reg = np.where(r>1.e-8,r,1.e-8)
    theta = np.where(r > 0, np.arccos(z / r_reg), 0.0)

    R = computeRadial(l1,n,r)
    Y = RealSphericalHarmonics.Ylm(l2,m,theta, phi)

    return np.where(r <= 1.0, R * Y, 0.0)

def computeZernikesFT(n,l,m, pos, drop_phase: bool = False):
    '''
    get Fourier transform

    cfr. https://arxiv.org/pdf/1510.04837, eq 36 (alpha -> 1, n -> 2*n + l, p -> n, 2 pi x -> k)
    '''
    r = np.linalg.norm(pos, axis=-1)
    r_reg = np.where(r>1.e-8,r,1.e-8)
    x, y, z = pos[...,0], pos[...,1], pos[...,2] 

    phi = np.atan2(y, x)
    theta = np.arccos(z / r_reg)
    N2 = np.pi / 4 / (2*n + l + 1) # == Int |Ylm j_{2n+l+1}/r|^2
    # TODO check phase/sign
    real = RealSphericalHarmonics.Ylm(l,m,theta, phi) * spherical_jn(2*n + l + 1, r) / r_reg  / np.sqrt(N2)  
    return real if drop_phase else real * (-1)**n * 1.j**l

## ## ##

def jn_zeros(n_max: int, n_roots: int):
    '''
    Compute the zeros of the spherical Bessel functions of the first kind.

    https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html
    '''
    jn = lambda r, i: spherical_jn(i, r)
    points = np.arange(1,n_roots+n_max+1) * np.pi
    roots = [points]
    for i in range(1,n_max+1):
        roots.append(points := [brentq(jn, points[j], points[j+1], (i,)) for j in range(n_roots+n_max-i)])
    return np.array([row[:n_roots] for row in roots])

class RealSphericalHarmonics:
    # conversion coefficients for real spherical harmonics
    cos_minus = lambda m: (-1) ** m / np.sqrt(2)
    cos_plus = lambda m: 1 / np.sqrt(2)
    sin_minus = lambda m: 1j / np.sqrt(2) * (-1) ** m
    sin_plus = lambda m: -1j / np.sqrt(2)

    Us = {}
    '''Cache for the conversion matrices U_{mm'}.'''

    @classmethod
    def _to_realspace(cls,l: int) -> np.ndarray:
        '''Internal computation of the conversion matrix U_{mm'} for a given l.'''
        vals = {(0,0): 1.0}
        for m in range(1, l + 1):
            vals[(m, m)] = cls.cos_plus(m)
            vals[(m, -m)] = cls.cos_minus(m)
            vals[(-m, m)] = cls.sin_plus(m)
            vals[(-m, -m)] = cls.sin_minus(m)
        iis,jjs = zip(*[(i+l,j+l) for i,j in vals.keys()])
        return coo_array((list(vals.values()),(iis,jjs)), shape=(2*l+1,2*l+1)).toarray()
    
    @classmethod
    def U(cls, l: int) -> np.ndarray:
        '''Get the conversion matrix mapping complex spherical harmonics to real spherical harmonics for a given l.'''
        if l not in cls.Us:
            cls.Us[l] = cls._to_realspace(l)
        return cls.Us[l]
    
    @classmethod
    def Ylm(cls, l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        '''Compute real spherical harmonics Y^real_{lm} at angles theta and phi.'''
        match m:
            case 0:
                return np.real(sph_harm_y(l, m, theta, phi))
            case _ if m > 0:
                return 2 * cls.cos_plus(m) * np.real(sph_harm_y(l, m, theta, 0.0)) * np.cos(m * phi)
            case _ if m < 0:
                return np.real(2j * cls.sin_plus(m) * sph_harm_y(l, -m, theta, 0.0)) * np.sin(-m * phi)
        return np.zeros_like(theta)

@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray = None

    grid: np.ndarray = None
    r: np.ndarray = None
    theta: np.ndarray = None
    phi: np.ndarray = None
    dV: np.ndarray = None

    cached_Xlnm: dict[tuple[int,int,int],np.ndarray] = field(default_factory=dict)

    @classmethod
    def regular(cls, start = -50, stop = 49, num = 100) -> Self:
        ticks = np.linspace(start,stop,num)
        return cls(*np.meshgrid(ticks,ticks,ticks,indexing='ij'))
    
    def __post_init__(self):
        if self.z is not None:
            self.grid = np.stack([self.x,self.y,self.z],-1)
        else:
            self.grid = np.stack([self.x,self.y],-1)
        self.r = np.linalg.norm(self.grid,axis=-1)
        self.theta = np.arccos(self.z / self.r_reg) if self.z is not None else np.pi / 2
        self.phi = np.atan2(self.y,self.x)
        grid = self.grid.reshape((-1,self.grid.shape[-1]))
        self.dV = np.prod(grid.max(axis=0) - grid.min(axis=0)) / len(grid)

        self.cached_Ylm: dict[tuple[int,int],np.ndarray] = {}
        self.cached_jn: dict[int,np.ndarray] = {}
        self.cached_Xlnm: dict[tuple[int,int,int],np.ndarray] = {}

    @property
    def r_reg(self) -> np.ndarray:
        '''Return the regularized radius, avoiding division by zero.'''
        return np.maximum(self.r,1.e-8)
    
    def Ylm(self,l,m) -> np.ndarray:
        '''Compute the real spherical harmonics Y^real_{lm} at the angles of this grid'''
        res = self.cached_Ylm.get((l,m),RealSphericalHarmonics.Ylm(l,m,self.theta,self.phi))
        if (l,m) not in self.cached_Ylm:
            self.cached_Ylm[(l,m)] = res
        return res
    
    def jn(self,n) -> np.ndarray:
        '''Compute the spherical Bessel function of the first kind j_n at the radii of this grid.'''
        res = self.cached_jn.get(n,spherical_jn(n,self.r))
        if n not in self.cached_jn:
            self.cached_jn[n] = res
        return res
    
    def fourier_bessel_expansion(self,l, m, k: np.ndarray) -> np.ndarray:
        '''
        compute the plane wave expansion of this Grid at wavevector k

        Assuming k is a 1D array of frequencies, 
        the resulting shape is a cartesion product of Grid and k
        '''
        ball = self.r[...,None] < np.abs(self.grid).max()
        return spherical_jn(l, self.r[...,None] * k) * self.Ylm(l,m)[...,None] * k * np.sqrt(2 / np.pi) * ball
    
    ## Zernike methods TODO remove? ##
    
    def Rnl(self, n, l) -> np.ndarray:
        return np.where(self.r <= 1.0,computeRadial(n,l,self.r),0.0)
    
    def Znlm(self, n, l, m) -> np.ndarray:
        return self.Rnl(n,l) * self.Ylm(l,m)
    
    def Xnlm(self, n, l, m) -> np.ndarray:
        factory = lambda: spherical_jn(2*n+l+1,self.r) / self.r_reg * self.Ylm(l,m) * np.sqrt(4 * (2 * n + l + 1) / np.pi)
        res = self.cached_Xlnm.get((n,l,m),factory())
        if (n,l,m) not in self.cached_Xlnm:
            self.cached_Xlnm[(n,l,m)] = res
        return res
    
