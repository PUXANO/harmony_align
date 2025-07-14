from typing import Generator

import numpy as np
from spherical.wigner import Wigner
from quaternionic.arrays import array

def to_realspace_Y(l: int) -> np.ndarray:
    '''
    Return a (2l+1) x (2l+1) matrix such that

    Y^real_{lm} = U_{mm'}Y_{lm'}

    for Y_{lm} ~ exp(im\phi) the usual spherical harmonics 
    and Y^real_{lm} ~ cos(m\phi) if m>=0 else sin(m\phi).

    cfr. https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    '''
    A = np.eye(2*l + 1)
    B = np.fliplr(A)
    b = np.diag([-1.j*(-1)**m/np.sqrt(2) for m in range(-l,0)] + [0] + [1/np.sqrt(2)]*l)
    a = np.diag([1.j/np.sqrt(2)]*l + [1.0] + [(-1)**m/np.sqrt(2) for m in range(1,l+1)])
    return a @ A + b @ B

class WignerGallery:
    '''
    Object generating all l-representation of the rotation group for a given gallery of rotations
    
    TODO under construction, perhaps use https://spherical.readthedocs.io/en/main/
    '''
    def __init__(self, n_spherical: int = 36, n_inplane: int = 36, l_max = 10):
        super().__init__()
        self.grid = array.from_euler_angles(np.array(list(self.gallery(n_spherical, n_inplane))))
        self.matrices = list(self.get_matrices(self.grid, l_max))

    @classmethod
    def gallery(cls, n_spherical: int = 36, n_inplane: int = 36) -> Generator[tuple[float,float,float],None,None]:
        '''
        Create Euler angle gallery in the convention of Mike Boyle:
        cfr. https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible

        alpha = psi   = inplane
        beta  = theta = tilt
        gamma = phi   = rot
        '''
        eps = 1.e-8
        for theta in np.linspace(0,np.pi/2,n_spherical//4+1):
            M = int(np.ceil(n_spherical * np.sin(theta+eps)))
            for phi in np.linspace(0,2*np.pi,M+1)[:-1]:
                for psi in np.linspace(0,2*np.pi,n_inplane+1)[:-1]:
                    yield psi,theta,phi

    def get_matrices(self, gallery: np.ndarray,l_max = 10) -> Generator[np.ndarray,None,None]:
        wigner = Wigner(l_max)
        entries = wigner.D(gallery)
        i = 0
        for l in range(l_max+1):
            Urs = to_realspace_Y(l) # convert Ylm to real space convention
            f = i + (2*l+1)**2
            M = Urs @ entries[:,i:f].reshape((-1,2*l+1,2*l+1)) @ np.conjugate(Urs.T)
            assert np.allclose(np.imag(M),0), "Wigner D's in real space convention should be real"
            yield np.real(M)
            i = f

    def __len__(self):
        return len(self.grid)

    def transformations(self) -> Generator[np.ndarray,None,None]:
        yield from self.matrices

    def __getitem__(self, item: int) -> dict[str, float]:
        '''return euler angles?'''
        angles = ["anglePsi","angleTilt","angleRot"]
        return {angle: rad.item() * 180 / np.pi for angle, rad in zip (angles, self.grid[item].to_euler_angles)}

