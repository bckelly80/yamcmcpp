all: yamcmcpp

yamcmcpp: force
	CFLAGS="-O3" python setup.py build_ext --inplace

clean:
	rm -rf build src/*.pyc src/yamcmcpp.so

force:
	true
