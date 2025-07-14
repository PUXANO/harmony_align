'''
Mini API to reproduce XMipp functionality
'''

from typing import Literal, Tuple, List, Self, Generator
from pathlib import Path
import subprocess
from os import getenv
from dataclasses import dataclass

import numpy as np
import mrcfile
import starfile
import pandas as pd

def check_xmipp_env():
    assert getenv('XMIPP_HOME'), "XMIPP environment not set for this shell"

def xmd_write(dataframe: pd.DataFrame, path: Path):
    '''
    Modified version of starfile-write, prepending the Xmipp specification
    '''
    starfile.write(dataframe, path)
    metadata = "# XMIPP_STAR_1 *\n# \n" + path.read_text()
    path.write_text(metadata)

@dataclass
class AtomLine:
    "Content of a single atom line in a pdb"
    id: int
    coordinates: np.ndarray
    name: Literal['CA', 'C', 'N', 'O'] = 'CA'
    element: Literal['C','N','O'] = 'C'
    residue: Literal['HIS','SER','GLN'] = 'MET'
    residue_id: int = 1
    chain: str = 'A'
    occupancy: float = 1.0
    temperature_factor: float = 16.0
    segment: str = ' '
    charge: str = ' '

    def __str__(self):
        parts = []
        parts.append(f"ATOM  {self.id:5d}  {self.name:<3} {self.residue}")
        parts.append(f" {self.chain}{self.residue_id:4d}    ")
        parts.append(''.join(f"{x:8.3f}" for x in self.coordinates))
        parts.append(f"{self.occupancy:6.2f}{self.temperature_factor:6.2f}      {self.segment:<4}{self.element:>2}{self.charge:>2}")
        assert tuple(map(len,parts)) == (20,10,24,26), f"Badly formatted PDB line: {''.join(parts)}"
        return ''.join(parts)

    @classmethod
    def parse(cls, line: str) -> Self | None:
        '''
        Parse pdb structure
        
        cfr. https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
        '''
        if line[:4] != 'ATOM':
            return None
        return cls(id = int(line[6:11]),
                   name = line[13:16],
                   residue = line[17:20],
                   chain = line[21:22],
                   residue_id = line[22:26],
                   coordinates = np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]),
                   occupancy = float(line[54:60]),
                   temperature_factor = float(line[60:66]),
                   segment = line[72:76],
                   element = line[76:78],
                   charge = line[78:80]
                   )
    
    @classmethod
    def concat(cls, coordinates: np.ndarray) -> str:
        return '\n'.join(str(cls(i,coord)) for i,coord in enumerate(coordinates,1))
    
    @classmethod
    def to_pdb(cls, coordinates: np.ndarray, path: Path) -> Path:
        path.write_text(cls.concat(coordinates)+'\nEND\n')
        return path

    @classmethod
    def from_pdb(cls, path: Path) -> Generator[Self,None,None]:
        for line in path.read_text().split('\n'):
            if (atom := cls.parse(line)) is not None:
                yield atom

def range_lines(key: str, N: int) -> List[str]:
    '''create range for projection metadata assuming full circle partitions'''
    stop = 180.0 if key == 'Tilt' else 360.0 * (N-1) / N
    return [
        f"_proj{key}Range    '{0.0:.06f} {stop:.06f} {N}",
        f"_proj{key}Randomness    even"
    ]

def projection_parameters(path: Path, n_rot: int = 10, n_tilt: int = 10, n_psi: int = 1, grid_size: int = 100, **kwargs) -> Path:
    '''create a projection parameters file for sampling'''
    if path.is_dir():
        path = path / "projection_parameters.xmd"
    lines = ['# XMIPP_STAR_! *','data_block1']
    lines.append(f"_dimensions2D    '{grid_size} {grid_size}'")
    for key, n_key in zip(['Rot','Tilt','Psi'],[n_rot,n_tilt,n_psi]):
        if n_key > 1:
            lines.extend(range_lines(key, n_key))
    lines.append("_noiseCoord '0.000000 0'")
    path.write_text('\n'.join(lines))
    return path

class Xmipp:
    '''Interface to some xmipp commands'''
    try:
        home = Path(getenv('XMIPP_HOME'))
    except:
        home = None
    available: dict[str,bool] = {}

    def __init__(self, out_dir: Path | str, label: str = "test"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True,exist_ok=True)
        self.label = label

        # params
        self.grid_size = 100
        self.l1 = 1
        self.l2 = 1

        if self.home is None:
            raise EnvironmentError('Xmipp not initialized')

    @classmethod
    def run(cls, cmd: str,*args):
        '''Call Xmipp using subprocess'''

        if cls.available.get(cmd) is None:
            cls.available[cmd] = cls.verify(cmd)
        if cls.available.get(cmd) is False:
            raise EnvironmentError(f"{cmd} not available for Xmipp")
        cmd = cls.home / "bin" / cmd
        
        try:
            subprocess.run(' '.join([str(cmd)] + list(args)),shell=True)
        except FileNotFoundError:
            raise EnvironmentError(f"Xmipp not found. Probably {cls.home} is not the Xmipp home folder.")
      
    @classmethod
    def verify(cls, cmd: str) -> bool:
        '''Call Xmipp using subprocess'''
        cmd = cls.home / "bin" / cmd
        try:
            process = subprocess.Popen(' '.join([str(cmd), "--help"]),shell=True,stdout=subprocess.PIPE)
            return any(iter(process.stdout.readline,b''))
        except FileNotFoundError:
            return False

    def volume_from_pdb(self, pdb: Path | np.ndarray, label: str = None) -> Path:
        '''
        Create a volume from a set of coordinates (as pdb/cif or plain numpy array) with the given grid_size.
        '''
        label = self.label if label is None else label
        if isinstance(pdb, np.ndarray):
            pdb = AtomLine.to_pdb(pdb, self.out_dir / f"{label}.pdb")
        volume_path = self.out_dir / f'{label}.vol'
        self.run('xmipp_volume_from_pdb', 
                f'-i {pdb}',
                '--sampling 1.000000',
                f"-o {volume_path.with_suffix('')}",
                f'--size {self.grid_size} {self.grid_size} {self.grid_size}',
                '--verbose 0')
        return volume_path
    
    def angular_project(self, volume_path: Path, angular_resolution: int = 10, experimental_images: Path = None) -> Tuple[Path,Path]:
        '''
        Project volume on a equilaterally distributed set of directions, optionally steered by experimental images.

        NOTE in_plane rotation sampling is omitted here. It is not required by the aligment, which does its own in plane sampling,
             and is buggy in the current xmipp release, cfr. GH issue #1008
        TODO not sure how the experimental_images influence the procedure, would be good to know.
        '''
        extra = ''
        if experimental_images is not None:
            images_meta = experimental_images.with_suffix('.xmd')
            self.run('xmipp_metadata_selfile_create',
                    f'-p "{experimental_images}"',
                    f'-o {images_meta}',
                    '-s')
            extra = f' --compute_neighbors --angular_distance -1 --experimental_images {images_meta}'
        self.run('xmipp_angular_project_library',
                  f"-i {volume_path.with_suffix('.vol')}",
                  f'-o {(mrcs_path := volume_path.with_suffix(".mrcs"))}',
                  f'--sampling_rate {angular_resolution}',
                  '--method real_space',
                  '--sym c1',
                  '--verbose 0',
                  extra)
        doc_path = mrcs_path.with_suffix('.doc')
        return mrcs_path, doc_path
    
    def angular_project_selection(self, volume_path: Path, angles: Path) -> Tuple[Path,Path]:
        '''
        Project volume on a selection of angles.
        '''
        if self.verify('xmipp_angular_project_selection'):
            # Custom command in modified Xmipp
            self.run('xmipp_angular_project_selection',
                    f"-i {volume_path.with_suffix('.vol')}",
                    f'-o {(mrcs_path := volume_path.with_suffix(".mrcs"))}',
                    f'--sel {angles}',
                    '--method real_space',
                    '--verbose 0')
        else:
            # Hack in regular Xmipp
            self.run('xmipp_cuda_angular_continuous_assign2',
                     f"-i {angles}",
                     "-o uselessOutput.mrcs",
                     "--sampling 1.000000",
                    "--Rmax 50",
                    "--padding 2",
                    f"--ref {volume_path}",
                    "--max_resolution 2.5",
                    "--nThreads 4",
                    f"--oprojections {mrcs_path}")  
            
        doc_path = mrcs_path.with_suffix('.doc') # TODO not sure this will be generated in continuous_assign2
        return mrcs_path, doc_path
    
    def phantom_project(self, volume_path: Path, angular_resolution: int = 10, in_plane: bool = False) -> Tuple[Path,Path]:
        '''
        Projection method (on CPU only) ensuring a more uniform distribution.
        '''
        mrcs_path = volume_path.with_suffix('.mrcs')
        xmd_path = volume_path.with_suffix('.xmd')
        n_psi = 1 if not in_plane else 360 // angular_resolution
        parameters_path = projection_parameters(self.out_dir,
                                                360//angular_resolution, 
                                                180 // angular_resolution + 1, 
                                                n_psi, 
                                                grid_size = self.grid_size)
        self.run('xmipp_phantom_project',
            f"-i {volume_path.with_suffix('.vol')}",
            f'-o {mrcs_path}',
            f'--params {parameters_path}',
            f'--method real_space',
            '--verbose 0')
        return mrcs_path, xmd_path
    
    def get_deformations(self, angles_path: Path, volume_path: Path) -> Path:
        deformation_path = self.out_dir / f'{volume_path.stem}_deformations.xmd'
        self.run('xmipp_cuda_angular_sph_alignment',
                 f'-i {angles_path}',
                 f'--ref {volume_path}',
                 f'-o {deformation_path}',
                 f'--odir {self.out_dir}', # BUG without specifying this, output is sometimes not written
                 f'--l1 {self.l1}',
                 f'--l2 {self.l2}',
                 f'--optimizeDeformation',
                 f'-v 0')
        return deformation_path
    
    @property
    def N_clnm(self) -> int:
        '''Compute nr of coefficients'''
        return 3 * sum(2 * l + 1 for n in range(self.l1+1) for l in range(min(self.l2,n)+1) if (n-l) % 2 == 0)
    
    def parse_clnm(self, clnm: str) -> np.ndarray:
        '''
        Create valid set of coefficients from a sphCoefficients entry:
        '''
        coef = np.array([float(c) for c in  clnm.split(' ') if c])
        assert len(coef) == self.N_clnm + 8, f"Found {len(coef)} coefficients, expected {self.N_clnm} + 8"
        return coef[:-8]
    
    def deformation_clnm(self, coefficients: str) -> Path:
        '''
        Create a deformations file structured like

        l1 l2 R
        znk_x_1 ... znk_x_N znk_y_1 ... znk_y_N znk_z_1 ... znk_z_N [+ 8 more coeff]

        with znk's the (white space separated) Zernike3D coefficients for the different coordinates and
          N = sum(2 * l +1 for n in range(l1+1) for l in range(min(n,l2+1)) if (n-l)%2 == 0)

        NOTE Typically 'coefficients' contains 8 additional numbers, not sure on their purpose or necessity here...
    
        '''
        _coef_path = self.out_dir / f"clnm.txt"
        basic_params = f'{self.l1} {self.l2} {self.grid_size//2}'
        clnm = ' '.join([str(c) for c in self.parse_clnm(coefficients)])
        with open(_coef_path,'w') as coef_file:
            coef_file.write('\n'.join([basic_params,clnm]))
        return _coef_path
    
    def apply_deformations(self,coordinates: Path | np.ndarray, coefficients: str | Path, label: str = 'deformed') -> Path:
        if isinstance(coordinates,np.ndarray):
            coordinates = AtomLine.to_pdb(coordinates, self.out_dir / "coordinates.pdb")
        if isinstance(coefficients, str):
            coefficients = self.deformation_clnm(coefficients)
        deformed_path = self.out_dir / f'{label}.pdb'
        self.run('xmipp_pdb_sph_deform',
                 f'--pdb {coordinates}',
                 f'-o {deformed_path}',
                 f'--clnm {coefficients}',
                 '--boxsize 100.0',
                 '--sr 1.0',
                 '--center_mass',
                 '-v 1')
        return deformed_path

    def align(self,experimental_images: Path, doc_path: Path) -> Path:
        '''
        Align the given 2d-classes in .mrcs with the gallery created by angular_project.
        The latter is summarized in a doc file, referring in turn to a (large) set of .mrcs projections.

        NOTE While phantom_project and angular_project produce similar output, in .xmd and .doc respectively,
             only the format of the latter is accepted here.
        '''
        images_meta = experimental_images.with_suffix('.xmd')
        angles_path = self.out_dir / f"{self.label}_angles.xmd"
        if not images_meta.exists():
            self.run('xmipp_metadata_selfile_create',
                    f'-p "{experimental_images}"',
                    f'-o {images_meta}',
                    '-s')
        self.run('xmipp_cuda_align_significant',
                  f'-i {images_meta}',
                  f'-r {doc_path}',
                  f'--odir {angles_path.parent}',
                  f'-o {angles_path.name}',
                  '--keepBestN 1.000000',
                  '--dev 0',
                  '--verbose 0')  
        return angles_path
    
    def trim_images(self, mrcs_path: Path, n_classes: int, meta_path: Path = None, target: Path = None) -> Tuple[Path, Path]:
        '''
        Picks a random subset of mrc images and, optionally, the corresponding metadata file.
        '''
        images = mrcfile.read(mrcs_path)
        selection = np.random.choice(len(images),n_classes, replace=False)
        target = self.out_dir / f'{self.label}_simulation' if target is None else target
        selected_mrcs_path = target.with_suffix('.mrcs')
        mrcfile.write(str(selected_mrcs_path),images[selection], overwrite=True)
        if meta_path is None:
            return selected_mrcs_path, None
        data = starfile.read(meta_path)
        selected = data.iloc[selection].assign(image=[f'{i:06d}@{selected_mrcs_path.absolute()}' for i in range(1,len(selection) + 1)])
        selected_meta_path = target.with_suffix(meta_path.suffix)
        xmd_write(selected,selected_meta_path)
        return selected_mrcs_path, selected_meta_path

    def simulate(self, pdb: Path | np.ndarray, n_classes: int, angular_resolution: int = 10, use_phantom: bool = False, target: Path = None) -> Tuple[Path, Path]:
        '''
        Simulate n_classes 2D classes from a pdb using phantom_project. Output written to xmipp folder unless target provided.
        '''
        volume = self.volume_from_pdb(pdb)
        if use_phantom:
            mrcs_path, meta_path = self.phantom_project(volume, angular_resolution)
        else:
            mrcs_path, meta_path = self.angular_project(volume, angular_resolution)
        return self.trim_images(mrcs_path, n_classes, meta_path, target)
    
    def align_pdb(self, pdb: Path | np.ndarray, classes: Path) -> Path:
        '''
        Execute the complete alignment pipeline, starting from a pdb and a set of 
        experimental 2d-classed, returning angular matches as .xmd file.
        '''
        volume = self.volume_from_pdb(pdb)
        _, all_angles = self.angular_project(volume, experimental_images=classes)
        return self.align(classes, all_angles)
