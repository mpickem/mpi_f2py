#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
import sys
import time

from mpi4py import MPI

from fortran_source.hw import functions as fct
from python_source.cli import parse_args

args = parse_args()

comm = MPI.COMM_WORLD
master = 0
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()


if (mpi_rank == master):
  print()
  print('Namespace data from argument parser')
  print(args) # namespace
  sys.stdout.flush()

comm.barrier()

data_interval = args.orbitals**2 * (2*args.frequencies+1) * args.qpoints

displ=[] # displacement
displ.append(0)
rct=[]   # receive count

# distribution of data_interval
for i in xrange(mpi_size-1):
  rct.append((data_interval-displ[i])//(mpi_size-i))
  displ.append(rct[i]+displ[i])
rct.append(data_interval - displ[mpi_size-1])

qstart = displ[mpi_rank]
qstop  = displ[mpi_rank] + rct[mpi_rank]

if (mpi_rank == master):
  print()
  print('datainterval size: ', data_interval)
  print('rct list:          ', rct)
  print('displ list:        ',displ)
  print()
  sys.stdout.flush()

comm.barrier()

for i in xrange(mpi_size):
  if (mpi_rank == i):
    print('rank: ', mpi_rank, qstart,':',qstop)
    sys.stdout.flush()
  else:
    time.sleep(0.05)

comm.barrier()


data_vec = np.zeros((30,30,qstop-qstart))
data_everyone = np.ones((10,10,data_interval), dtype=np.int64)


if (mpi_rank == master):
  comm.Reduce(MPI.IN_PLACE, [data_everyone, MPI.INT64_T], op=MPI.SUM, root=master)
else:
  comm.Reduce([data_everyone, MPI.INT64_T], None, op=MPI.SUM, root=master)

if (mpi_rank == master):
  print(data_everyone[...,0])
  sys.stdout.flush()

sys.exit()

# from now on forward we only let the master write to stdout

if (mpi_rank == master):
  print(fct.hw1(0,0.1))
  print(fct.hw2(0,0.1))
  print(fct.hw3(1+1j, 1+0.5*1j))

# print(array_addition.__doc__)
a = np.arange(27, dtype=np.complex128).reshape((3,9), order='F')
b = np.full((3,9), fill_value=1.43+0.5j, dtype=np.complex128, order='F')
c = np.empty_like(a, dtype=np.complex128, order='F')


# array_addition([[1,2,3],[1,2,3]],[[2,3,4],[2,3,4]],[[0,0,0],[0,0,0]])
fct.array_addition(a,b,c)

if (mpi_rank == master):
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
  print('Singular matrix ... aborting')
  comm.Abort()
else:
  print('Good to go')

print(a)
