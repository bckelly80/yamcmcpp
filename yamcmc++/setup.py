from distutils.core import setup, Extension
import numpy.distutils.misc_util
import os

desc = open("README.rst").read()
required = ["numpy"]

# define the name of the extension to use
extension_name    = "_yamcmcpp"
extension_version = "0.1.0"
extension_url     = "https://github.com/bckelly80/yamcmcpp"

BOOST_DIR         = os.environ["BOOST_DIR"]
ARMADILLO_DIR     = os.environ["ARMADILLO_DIR"]
include_dirs      = ["/usr/include/", BOOST_DIR+"/include", ARMADILLO_DIR+"/include", numpy.distutils.misc_util.get_numpy_include_dirs()]
library_dirs      = ["/usr/lib64/", "/usr/lib/", BOOST_DIR+"/lib", ARMADILLO_DIR+"/lib"]

# define the libraries to link with the boost python library
libraries = [ "boost_python", "lapack", "blas"]

# define the source files for the extension
source_files = [ "boost_python_wrapper.cpp", "random.cpp" ]
 
# create the extension and add it to the python distribution
setup( name=extension_name, 
       version=extension_version, 
       author="Brandon Kelly and Andrew Becker",
       author_email="acbecker@gmail.com",
       packages=[extension_name],
       package_dir = { "": "src" },
       url=extension_url,
       description="C++ version of Yet Another Markov Chain Monte Carlo sampler",
       long_description=desc,
       install_requires=required,
       ext_modules=[Extension( extension_name, source_files, 
                               include_dirs=include_dirs, 
                               library_dirs=library_dirs, 
                               libraries=libraries )] )
