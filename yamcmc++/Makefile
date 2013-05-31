all: yamcmcpp

yamcmcpp: force
	python setup.py build_ext --inplace

clean:
	rm -rf build src/*.pyc src/_yamcmcpp.so

force:
	true
