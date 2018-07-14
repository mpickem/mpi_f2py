SHELL = /bin/bash
F2PY = f2py
LLINK = /opt/lapack-3.8.0/
LIBS = -L${LLINK} -llapack

all: hw.so

%.so: %.f90
	$(F2PY) -c -m hw $< --fcompiler=gnu95 $(LIBS) -DF2PY_REPORT_ON_ARRAY_COPY=1

.PHONY: clean
clean:
	rm -f *.so
