#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
import sys
import time

import scipy.linalg
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

rct = np.array(rct)
displ = np.array(displ)

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




########################################
# TESTING REDUCE
########################################

if (mpi_rank == master):
  print()
  print('REDUCE TEST')
  print()
  sys.stdout.flush()

data_everyone = np.ones((10,10,data_interval), dtype=np.int64, order='F')
# Reducing stuff
if (mpi_rank == master):
  comm.Reduce(MPI.IN_PLACE, [data_everyone, MPI.INT64_T], op=MPI.SUM, root=master)
else:
  comm.Reduce([data_everyone, MPI.INT64_T], None, op=MPI.SUM, root=master)

if (mpi_rank == master):
  print(data_everyone[...,0])
  sys.stdout.flush()

if (mpi_rank == master):
  print()
  print('END REDUCE TEST')
  print()

comm.barrier()
########################################





########################################
# TESTING GATHERV
########################################

if (mpi_rank == master):
  print()
  print('GATHERV TEST')
  print()
  sys.stdout.flush()

data_vec = np.ones((2,2,qstop-qstart), dtype=np.float64, order='F') * (mpi_rank+1)

print(data_vec)
sys.stdout.flush()
comm.barrier()
print()

# Gathering stuff
if (mpi_rank == master): # so not every core has to allocate this memory
  # if Fortran order we have to put the parllelized index on the right
  data_gather = np.zeros((2,2,data_interval), dtype=np.float64, order='F')
else:
  # so the namespace is there for the Gatherv command
  data_gather = np.zeros((1,1,1), dtype=np.float64, order='F')

# this for is because of the 2,2 in front
comm.Gatherv([data_vec, rct[mpi_rank]*4, MPI.DOUBLE] , [data_gather, rct*4, displ*4, MPI.DOUBLE], root=master)

if (mpi_rank == master):
  print(data_gather)
  sys.stdout.flush()

comm.barrier()

if (mpi_rank == master):
  print()
  print('END GATHERV TEST')
  print()
  sys.stdout.flush()
########################################



########################################
# TESTING BCAST
########################################

if (mpi_rank == master):
  print()
  print('BCAST TEST')
  print()
  sys.stdout.flush()


data = np.zeros((3,3,data_interval), dtype=np.complex128, order='F')
if (mpi_rank == master):
  data = np.full_like(data, fill_value=2.3+1j, dtype=np.complex128, order='F')

if (mpi_rank == 1):
  print('data before bcast', data)
  sys.stdout.flush()
comm.barrier()
comm.Bcast([data, MPI.COMPLEX16], root=master) # capital B -> in place (i.e. same name
# small b -> returns mpi4py data object ... no
if (mpi_rank == 1):
  print('data after bcast', data)
  sys.stdout.flush()
  print(data.flags)
comm.barrier()

if (mpi_rank == master):
  print('END BCAST TEST') # never forget about rescaling rct and displ 1!!!!
  sys.stdout.flush()

########################################




########################################
# TESTING SCATTER
########################################

if (mpi_rank == master):
  print()
  print('SCATTER TEST')
  print()
  sys.stdout.flush()

comm.barrier()

data = np.zeros((2,2,qstop-qstart), dtype=np.complex128, order='F')
if (mpi_rank == master):
  data_to_scatter = np.arange(2*2*data_interval, dtype=np.complex128).reshape((2,2,data_interval), order='F')
else:
  data_to_scatter = None


if (mpi_rank == 0):
  print(data_to_scatter)
  print(data)
  print()
  print()
  sys.stdout.flush()

comm.barrier()

comm.Scatterv([data_to_scatter, rct*4, displ*4, MPI.COMPLEX16], [data, 4*(qstop-qstart), MPI.COMPLEX16], root=master)

if (mpi_rank == 0):
  print(data_to_scatter[:,:,displ[0]:displ[0]+rct[0]])
  print()
  print(data)
  print()
  sys.stdout.flush()
comm.barrier()

if (mpi_rank == master):
  print('END SCATTER TEST') # never forget about rescaling rct and displ 1!!!!
  sys.stdout.flush()

sys.exit()

########################################
# TESTING SEND RECV
########################################

########################################
# TESTING PYTHON OBJECTS (lists etc. and not np arrays)
########################################

if (mpi_rank == master):
  print('ISEND PYTHON TEST') # never forget about rescaling rct and displ 1!!!!
  sys.stdout.flush()

comm.barrier()

if (mpi_rank == master):
  data = [123, 'what', 3.0+1j*25]
else:
  data = None

if (mpi_rank == master):
  comm.isend(data, dest=1, tag=11) # small i; general python objects
  # large i -> buffer like objects which have to be provided in a list form
  # normal -> data, size, type
  # vector -> data, rct, displ, type
elif (mpi_rank == 1):
  print('data before: ', data)
  data = comm.recv(source=0, tag=11)
  print('data after: ', data)


comm.barrier()

if (mpi_rank == master):
  print('END ISEND PYTHON TEST') # never forget about rescaling rct and displ 1!!!!
  sys.stdout.flush()

comm.barrier()




# sys.exit()

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


# a = np.eye(5, dtype=np.complex128, order='F')*2
a = np.random.random((200,200))
b = np.asfortranarray(a).astype(np.complex128)
# a[0,0] = 0

print('inverting....')

if False:
  t0 = time.time()
  for i in xrange(200):
    a = np.random.random((2000,2000))
    b = np.asfortranarray(a).astype(np.complex128)
    fct.inverse_matrix_z(b)
  print((time.time()-t0)/200, ' s')

  t0 = time.time()
  for i in xrange(200):
    a = np.random.random((2000,2000))
    b = np.asfortranarray(a).astype(np.complex128)
    scipy.linalg.inv(b)
  print((time.time()-t0)/200, ' s')

# if (fct.inverse_matrix_z(b)):
#   print('Singular matrix ... aborting')
#   comm.Abort()
# else:
#   print('Good to go')

# print(a)
