SHELL = /bin/bash
F2PY = f2py
export CC=gcc
export CXX=g++

hw.so: hw.f90
	$(F2PY) -c -m hw hw.f90 --fcompiler=gnu95

.PHONY: clean
clean:
	rm -f *.so
