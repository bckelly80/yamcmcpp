all: yamcmcppLib

yamcmcppLib: force
	CFLAGS="-O3" python setup.py build_ext --inplace

clean:
	rm -rf build 

force:
	true
