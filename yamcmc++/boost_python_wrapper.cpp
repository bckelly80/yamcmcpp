#include<boost/python.hpp>

// include the headers for the wrapped functions and classes
#include "random.hpp"

using namespace boost::python;

// generate wrapper code for the classes
BOOST_PYTHON_MODULE(Yamcmcpp){
    // declare the classes to wrap. remember to exposure all
    // constructors!
    class_<RandomGenerator>("RandomGenerator",init<>())
        .def("SetSeed", &RandomGenerator::SetSeed )
        ;
}
