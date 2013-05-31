#include<boost/python.hpp>

// include the headers for the wrapped functions and classes
#include "parameters.hpp" 
#include "proposals.hpp"
#include "random.hpp"
#include "samplers.hpp" 
#include "steps.hpp"

using namespace boost::python;

// generate wrapper code for the classes
BOOST_PYTHON_MODULE(Yamcmcpp){
    // declare the classes to wrap. remember to exposure all
    // constructors!

    class_<NormalProposal>("NormalProposal",init<>())
        .def(init<double>())
        .def("Draw", &NormalProposal::Draw )
        .def("LogDensity", &NormalProposal::LogDensity )
        ;
    
    class_<RandomGenerator>("RandomGenerator",init<>())
        .def("SetSeed", &RandomGenerator::SetSeed )
        .def("SaveSeed", &RandomGenerator::SaveSeed )
        .def("RecoverSeed", &RandomGenerator::RecoverSeed )
        .def("exp", &RandomGenerator::exp )
        .def("lognormal", &RandomGenerator::lognormal )
        .def("powerlaw", &RandomGenerator::powerlaw )
        .def("tdist", &RandomGenerator::tdist )
        .def("chisqr", &RandomGenerator::chisqr )
        .def("scaled_inverse_chisqr", &RandomGenerator::scaled_inverse_chisqr )
        .def("gamma", &RandomGenerator::gamma )
        .def("invgamma", &RandomGenerator::invgamma )
        .def("uniform_integer", &RandomGenerator::uniform_integer )
        ;
}
