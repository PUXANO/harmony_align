'''
Module providing generating sequences, to ensure terms always come in the same order
'''
from typing import Generator

import numpy as np

class ZernikeSequence:
    '''Parent class fixing the order of Zernike coefficients'''
    def __init__(self, l1: int, l2: int):
        self.l1 = l1
        self.l2 = l2
    def lnm(self) -> Generator[tuple[int,int,int],None,None]:
        for l in range(self.l2 + 1):
            for n in range(l,self.l1 +1,2):
                for m in range(-l, l+1):
                    yield l,n,m
    def ln(self) -> Generator[tuple[int,int],None,None]:
        for l in range(self.l2 + 1):
            for n in range(l,self.l1 +1,2):
                    yield l,n

class SphericalSequence:
    def __init__(self, l_max: int = 3, allow_isotropic: bool = True):
        self.l_max = l_max
        self.allow_isotropic = allow_isotropic

    def start_l(self) -> int:
        return 0 if self.allow_isotropic else 1

    def lm(self) -> Generator[tuple[int,int],None,None]:
        for l in range(self.start_l(), self.l_max+1):
            for m in range(-l,l+1):
                yield l,m

    def l(self) -> Generator[int,None,None]:
        for l in range(self.start_l(), self.l_max+1):
            yield l
