#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
import mpi4py # for information
from mpi4py import MPI
# if one deals with modules directly
from fortran_source.hw import functions as fct
# from folder.(*.so file) import module as namespace
import sys
import time


# if there are only subroutines and functions :

# from fortran_source.hw import * -> reference as hw1
# from fortran_source import hw -> reference as hw.hw1
# import fortran_source.hw -> reference as fortran_source.hw.hw1


################################################################################
## for future MPI stuff
################################################################################
# print(mpi4py.get_include())
# print(mpi4py.get_config())
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('rank: ', rank, 'size: ', size)
################################################################################



print(fct.hw1(0,0.1))
print(fct.hw2(0,0.1))
print(fct.hw3(1+1j, 1+0.5*1j))

# print(array_addition.__doc__)
a = np.arange(27, dtype=np.complex128).reshape((3,9), order='F')
b = np.full((3,9), fill_value=1.43+0.5j, dtype=np.complex128, order='F')
c = np.empty_like(a, dtype=np.complex128, order='F')


# array_addition([[1,2,3],[1,2,3]],[[2,3,4],[2,3,4]],[[0,0,0],[0,0,0]])
fct.array_addition(a,b,c)

print()
print()

print(fct.dlapack_mul.__doc__)
matrix_left = np.ones((10,1000), dtype=np.float64, order='F')
matrix_right = np.ones((1000,10), dtype=np.float64, order='F')
c = np.empty((10,10), dtype=np.float64, order='F')
fct.dlapack_mul(matrix_left, matrix_right, c)

matrix_left = np.ones((10,1000), dtype=np.complex128, order='F')
matrix_right = np.ones((1000,10), dtype=np.complex128, order='F')
c = np.zeros((10,10), dtype=np.complex128, order='F')
fct.zlapack_mul(matrix_left, matrix_right, c)


a = np.eye(5, dtype=np.complex128, order='F')*2
a[0,0] = 0

print('inverting....')
if (fct.inverse_matrix_z(a)):
  print('Singular matrix')
  sys.exit()
else:
  print('Good to go')

print(a)
