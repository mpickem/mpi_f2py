SHELL = /bin/bash
F2PY  = f2py
FC    = gfortran
LLINK = /opt/lapack-3.8.0/
LIBS  = -L${LLINK} -llapack

all: hw.so

# convenction F90 -> directly for f2py
# f90 -> fortran compiler and only for linking
#     -> therefore not callable directly by python
hw.so: hw.F90 writeout.o
	$(F2PY) -c -m hw $^ --fcompiler=gnu95 $(LIBS) -DF2PY_REPORT_ON_ARRAY_COPY=1

writeout.o: writeout.f90
	$(FC) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.so *.o *.mod
