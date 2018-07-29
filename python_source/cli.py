#! /usr/bin/env python

import argparse

# with can use the args argument for debugging purposes
# e.g. parse_args('2 1 3'.split())
def parse_args(args=None):
  parser = argparse.ArgumentParser(
    description='here is a description',
    epilog="That's the end of the help")
  # optional keywords start with a -
  # mandatory simply without
  parser.add_argument('orbitals',    help='Number of bands', type=int)
  parser.add_argument('frequencies', help='Number of bosonic frequencies', type=int)
  parser.add_argument('qpoints',     help='Number of "bosonic" q-points', type=int)
  parser.add_argument('--type',      help='which type we want (0,1,2)', default=0, type=int, dest='abc')
  return parser.parse_args(args)

if __name__ == '__main__':
  args = parse_args()
