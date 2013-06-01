#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include <boost/python.hpp>
#include <armadillo>
#include <string>

// include the headers for the wrapped functions and classes
#include "parameters.hpp" 
#include "proposals.hpp"
#include "random.hpp"
#include "samplers.hpp" 
#include "steps.hpp"


using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(univariateNormal, RandomGenerator::normal, 0, 2)

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(uniformDouble, RandomGenerator::uniform, 0, 2)

// generate wrapper code for the classes
BOOST_PYTHON_MODULE(_yamcmcpp){
    // declare the classes to wrap. remember to exposure all
    // constructors!

    // Parameters:

    // Parameter is templated.  I believe we need explicit
    // instantiations.  For now use use arma::vec.
    // If you want to get serious, look at:
    // http://boost.2283326.n4.nabble.com/C-sig-Problem-with-Templated-class-and-Boost-python-td2697861.html
    struct ParameterWrap : Parameter<arma::vec>, wrapper<Parameter<arma::vec> >
    {
        // Pure virtual functions
        arma::vec StartingValue() { return this->get_override("StartingValue")(); }
        
        // Virtual functions with default implementations
        double LogDensity(arma::vec value)
        {
            if (override LogDensity = this->get_override("LogDensity"))
                return LogDensity(value);
            return Parameter<arma::vec>::LogDensity(value);
        }
        double default_LogDensity(arma::vec value) { return this->Parameter<arma::vec>::LogDensity(value); }        

        arma::vec RandomPosterior()
        {
            if (override RandomPosterior = this->get_override("RandomPosterior"))
                return RandomPosterior();
            return Parameter<arma::vec>::RandomPosterior();
        }
        arma::vec default_RandomPosterior() { return this->Parameter<arma::vec>::RandomPosterior(); }        

        void Save(arma::vec new_value)
        {
            if (override Save = this->get_override("Save"))
                Save(new_value);
            Parameter<arma::vec>::Save(new_value);
        }
        void default_Save(arma::vec new_value) { this->Parameter<arma::vec>::Save(new_value); }        
    };

    class_<ParameterWrap, bases<BaseParameter>, boost::noncopyable>("Parameter")
        .def("StartingValue", pure_virtual(&Parameter<arma::vec>::StartingValue))        
        ;

        //.def("LogDensity", &Parameter<arma::vec>::LogDensity, &ParameterWrap::default_LogDensity)
        //.def("RandomPosterior", &Parameter<arma::vec>::RandomPosterior, &ParameterWrap::default_RandomPosterior)
        //.def("Save", &Parameter<arma::vec>::Save, &ParameterWrap::default_Save)
        //;
    /*
    class_<Parameter, bases<BaseParameter> >("Parameter", no_init)
        .def(init<bool,std::string,double>())
        .def("StartingValue", &Parameter::StartingValue )
        .def("LogDensity", &Parameter::LogDensity )
        .def("RandomPosterior", &Parameter::RandomPosterior )
        .def("Value", &Parameter::Value )
        .def("Save", &Parameter::Save )
        .def("SetSampleSize", &Parameter::SetSampleSize )
        .def("AddToSample", &Parameter::AddToSample )
        .def("GetSamples", &Parameter::GetSamples )
        ;                      
    */

    // Proposals:
    /*
    class_<NormalProposal, bases<Proposal<double> > >("NormalProposal",init<>())
        .def(init<double>())
        .def("Draw", &NormalProposal::Draw )
        .def("LogDensity", &NormalProposal::LogDensity )
        ;
    class_<StudentProposal, bases<Proposal<double> > >("StudentProposal",init<>())
        .def(init<double,double>())
        .def("Draw", &StudentProposal::Draw )
        .def("LogDensity", &StudentProposal::LogDensity )
        ;
    class_<MultiNormalProposal, bases<Proposal<arma::vec> > >("MultiNormalProposal",init<>())
        .def(init<arma::mat>())
        .def("Draw", &MultiNormalProposal::Draw )
        .def("LogDensity", &MultiNormalProposal::LogDensity )
        ;
    class_<LogNormalProposal, bases<Proposal<double> > >("LogNormalProposal",init<>())
        .def(init<double>())
        .def("Draw", &LogNormalProposal::Draw )
        .def("LogDensity", &LogNormalProposal::LogDensity )
        ;
    */

    /* PROBLEM WITH TEMPLATED CLASSES
    class_<StretchProposal<Parameter>, bases<EnsembleProposal<arma::vec,Parameter> > >("StretchProposal<Parameter>",init<>())
        .def(init<Ensemble<Parameter>,int,double>())
        .def("SetScalingSupport", &StretchProposal<Parameter>::SetScalingSupport )
        .def("GrabParameter", &StretchProposal<Parameter>::GrabParameter )
        .def("Draw", &StretchProposal<Parameter>::Draw )
        .def("LogDensity", &StretchProposal<Parameter>::LogDensity )
        ;
    */

    // Random
    class_<RandomGenerator>("RandomGenerator",init<>())
        .def("SetSeed", &RandomGenerator::SetSeed )
        .def("SaveSeed", &RandomGenerator::SaveSeed )
        .def("RecoverSeed", &RandomGenerator::RecoverSeed )
        .def("exp", &RandomGenerator::exp )
        .def("normal", static_cast< double(RandomGenerator::*)(double mu, double sigma)>
             (&RandomGenerator::normal), univariateNormal())
        .def("normal", static_cast< arma::vec(RandomGenerator::*)(arma::mat covar)>
             (&RandomGenerator::normal))
        .def("lognormal", &RandomGenerator::lognormal )
        .def("uniform", static_cast< double(RandomGenerator::*)(double lowbound, double upbound)>
             (&RandomGenerator::uniform), uniformDouble())
        .def("uniform", static_cast< int(RandomGenerator::*)(int lowbound, int upbound)>
             (&RandomGenerator::uniform))
        .def("powerlaw", &RandomGenerator::powerlaw )
        .def("tdist", &RandomGenerator::tdist )
        .def("chisqr", &RandomGenerator::chisqr )
        .def("scaled_inverse_chisqr", &RandomGenerator::scaled_inverse_chisqr )
        .def("gamma", &RandomGenerator::gamma )
        .def("invgamma", &RandomGenerator::invgamma )
        ;

    // Samplers

    // Steps
}
