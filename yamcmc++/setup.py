from distutils.core import setup
import numpy.distutils.misc_util
import os

desc              = open("README.rst").read()
extension_version = "0.1.0"
extension_url     = "https://github.com/bckelly80/yamcmcpp"
CFLAGS            = "-O3"
BOOST_DIR         = os.environ["BOOST_DIR"]
ARMADILLO_DIR     = os.environ["ARMADILLO_DIR"]
include_dirs      = [BOOST_DIR+"/include", ARMADILLO_DIR+"/include", "/usr/include/"]
for include_dir in numpy.distutils.misc_util.get_numpy_include_dirs():
    include_dirs.append(include_dir)
library_dirs      = [BOOST_DIR+"/lib", ARMADILLO_DIR+"/lib", "/usr/lib64/", "/usr/lib/"]

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration("yamcmcpp", parent_package, top_path)
    config.version = extension_version
    config.add_data_dir((".", "yamcmcpp"))
    config.add_installed_library(
        "yamcmcpp",
        sources=["proposals.cpp", "random.cpp", "samplers.cpp", "steps.cpp"],
        install_dir="../../")

    # Currently there is nothing in the wrapper, do not build unless we add anything
    # NOTE: we will also add the following to yamcmcpp/__init__.py
    # from _yamcmcpp import *
    if False:
        config.add_extension(
            "_yamcmcpp",
            sources=["boost_python_wrapper.cpp", "samplers.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["boost_python", "boost_filesystem", "boost_system", "armadillo", "yamcmcpp"]
            )

    config.add_data_dir(("../../../../include", "include"))
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
