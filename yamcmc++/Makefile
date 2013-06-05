all: yamcmcpp

yamcmcpp: force
	python setup.py build_ext --inplace

clean:
	rm -rf build src/*.pyc src/yamcmcpp.so

force:
	true
