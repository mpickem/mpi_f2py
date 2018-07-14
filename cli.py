#! /usr/bin/env python

import argparse

def parse_args(args=None):
  parser = argparse.ArgumentParser(
    description='here is a description',
    epilog="That's the end of the help")
  # optional keywords start with a -
  # mandatory simply without
  parser.add_argument('number', help='how many whatever', type=int)
  parser.add_argument('--filename', help='Name of data file', type=str)
  parser.add_argument('--month', help='Month to filter on', type=int)
  parser.add_argument('--type', help='which type we want (0,1,2)', default=0, type=int, dest='abc')
  return parser.parse_args(args)

def main(args=None):
  # args = parse_args(args)
  return parse_args(args)

if __name__ == '__main__':
  # main(['abc', 'def']) # for testing purposes ... if this is empty -> CommandLine
  args = main()

  print(args) # prints the Namespace # prints the Namespace
