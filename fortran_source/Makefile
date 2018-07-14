SHELL = /bin/bash
F2PY = f2py
export CC=gcc
export CXX=g++

all: hw.so

%.so: %.f90
	$(F2PY) -c -m hw $< --fcompiler=gnu95 -DF2PY_REPORT_ON_ARRAY_COPY=1

.PHONY: clean
clean:
	rm -f *.so
