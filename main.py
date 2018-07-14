#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
import mpi4py # for information
from mpi4py import MPI
from fortran_source.hw import *

# from fortran_source.hw import * -> reference as hw1
# from fortran_source import hw -> reference as hw.hw1
# import fortran_source.hw -> reference as fortran_source.hw.hw1

print(mpi4py.get_include())
print(mpi4py.get_config())

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('rank: ', rank, 'size: ', size)
print()




print(hw1(0,0.1))
print(hw2(0,0.1))
print(hw3(1+1j, 1+0.5*1j))

print(array_addition.__doc__)

# a = np.ones((3,9), dtype=np.complex128, order='F')
a = np.arange(27, dtype=np.complex128).reshape((3,9), order='F')
b = np.full((3,9), fill_value=1.43+0.5j, dtype=np.complex128, order='F')
c = np.empty_like(a, dtype=np.complex128, order='F')

# array_addition([[1,2,3],[1,2,3]],[[2,3,4],[2,3,4]],[[0,0,0],[0,0,0]])
array_addition(a,b,c)
print(c)
