#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include <boost/python.hpp>

// include the headers for the wrapped functions and classes
#include "include/samplers.hpp" 

BOOST_PYTHON_MODULE(yamcmcppLib){
    using namespace boost::python;

    class_<MCMCOptions>("MCMCOptions")
        .def("getSampleSize", &MCMCOptions::getSampleSize)
        .def("setSampleSize", &MCMCOptions::setSampleSize)
        .def("getThin", &MCMCOptions::getThin)
        .def("setThin", &MCMCOptions::setThin)
        .def("getBurnin", &MCMCOptions::getBurnin)
        .def("setBurnin", &MCMCOptions::setBurnin)
        .def("getChains", &MCMCOptions::getChains)
        .def("setChains", &MCMCOptions::setChains)
        .def("getDataFileName ", &MCMCOptions::getDataFileName)
        .def("setDataFileName", &MCMCOptions::setDataFileName)
        .def("getOutFileName ", &MCMCOptions::getOutFileName)
        .def("setOutFileName", &MCMCOptions::setOutFileName)
        ;
};
