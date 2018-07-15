all: sources

sources:
	cd fortran_source/; make
	cd python_source/; python -m compileall .
	python -m compileall main.py

clean:
	cd fortran_source/; make clean
	cd python_source/; make clean
	rm -rf *.pyc
