all: yamcmcppLib

yamcmcppLib: force
	CFLAGS="-O3" python setup.py build_ext --inplace

clean:
	rm -rf build lib/yamcmcpp/*.pyc lib/yamcmcpp/yamcmcppLib.so

force:
	true
