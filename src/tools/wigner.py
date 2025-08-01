from typing import Generator

import numpy as np
from spherical.wigner import Wigner
from quaternionic.arrays import array
from tools.utils import RealSph

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
            f = i + (2*l+1)**2
            M = RealSph.U(l) @ entries[:,i:f].reshape((-1,2*l+1,2*l+1)) @ np.conjugate(RealSph.U(l).T)
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

if __name__ == "__main__":

    def rotate_angles(theta, phi, rot):
        '''rotate angles theta, phi by rotation rot'''
        new_rot = rot.to_rotation_matrix.squeeze()
        vec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        new_vec = new_rot @ vec
        new_theta = np.arccos(new_vec[2])
        new_phi = np.arctan2(new_vec[1], new_vec[0])
        return new_theta, new_phi

    # pick some angles
    phi = 0.8 * np.pi
    theta = 0.3 * np.pi
    rot = array.from_euler_angles(np.array([[np.pi/2.13, np.pi/1.23,np.pi/3.21]]) )

    mplets = [np.array([RealSph.Ylm(l,m,theta,phi) for m in range(-l,l+1)]) for l in range(4)]

    # Rotate moments in 2 ways: through coordinates and wigner D matrices
    theta2,phi2 = rotate_angles(theta, phi, rot)
    matrices = WignerGallery.get_matrices(None,rot, 3)

    mplets_rotated_coord = [np.array([RealSph.Ylm(l,m,theta2, phi2) for m in range(-l,l+1)]) for l in range(4)]
    mplets_rotated_moments = [mat[0] @ mplet for mat, mplet in zip(matrices, mplets)]

    # Check if they match
    for l, (mplet_rot, mplet_wigner) in enumerate(zip(mplets_rotated_coord, mplets_rotated_moments)):
        if np.allclose(mplet_rot, mplet_wigner):
            print(f"Matching spherical moments for l={l}")
        else:
            print(np.stack((mplet_rot, mplet_wigner)))
