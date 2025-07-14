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
    l_max = 3

    def lm(self) -> Generator[tuple[int,int],None,None]:
        for l in range(self.l_max+1):
            for m in range(-l,l+1):
                yield l,m

    @classmethod
    def collect(cls, gen: Generator, collector = np.stack):
        l = 0
        multiplet = []
        for value in gen:
            multiplet.append(value)
            if len(multiplet) == 2 * l + 1:
                yield collector(multiplet)
                l += 1
                multiplet = []