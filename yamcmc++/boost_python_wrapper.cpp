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
    // Needed for virtual functions of this base class
    /*
    struct BaseParameterWrap : BaseParameter, wrapper<BaseParameter>
    {
        std::string StringValue()
        {
            if (override StringValue = this->get_override("StringValue"))
                return StringValue();
            return BaseParameter::StringValue();
        }
        std::string default_StringValue() { return this->BaseParameter::StringValue(); }

        void SetSampleSize() 
        {
            this->get_override("SetSampleSize")();
        }

        void AddToSample() 
        {
            this->get_override("AddToSample")();
        }
    };
    class_<BaseParameterWrap, boost::noncopyable>("BaseParameter")
        .def("StringValue", &BaseParameter::StringValue, &BaseParameterWrap::default_StringValue)
        .def("SetSampleSize", pure_virtual(&BaseParameter::SetSampleSize))
        .def("AddToSample", pure_virtual(&BaseParameter::AddToSample))
        //.def(init<bool,std::string>())
        //.def(init<bool,std::string,double>())
        //.def("GetLogDensity", &BaseParameter::GetLogDensity )
        //.def("SetLogDensity", &BaseParameter::SetLogDensity )
        //.def("GetTemperature", &BaseParameter::GetTemperature )
        //.def("Track", &BaseParameter::Track )
        //.def("SetTracking", &BaseParameter::SetTracking )
        //.def("Label", &BaseParameter::Label )
        ;
    */

    /*
    class_<Parameter>("Parameter",init<bool,std::string,double>())
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
