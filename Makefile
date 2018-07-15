all: sources

sources:
	cd fortran_source/; make all
	cd python_source/; make all
	python -m compileall main.py

clean:
	cd fortran_source/; make clean
	cd python_source/; make clean
	rm -rf *.pyc
