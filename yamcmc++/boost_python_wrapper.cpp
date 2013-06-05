#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include <boost/python.hpp>
#include <armadillo>
#include <string>

// include the headers for the wrapped functions and classes
#include "include/parameters.hpp" 
#include "include/proposals.hpp"
#include "include/random.hpp"
#include "include/samplers.hpp" 
#include "include/steps.hpp"


using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(univariateNormal, RandomGenerator::normal, 0, 2)

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(uniformDouble, RandomGenerator::uniform, 0, 2)


// Parameter is templated.  I believe we need explicit
// instantiations.  For now use use arma::vec.
// If you want to get serious, look at:
// http://boost.2283326.n4.nabble.com/C-sig-Problem-with-Templated-class-and-Boost-python-td2697861.html
struct ParameterWrap : Parameter<arma::vec>, wrapper<Parameter<arma::vec> >
{
    ParameterWrap(PyObject *p, bool track, std::string label, double temperature=1.0)
        : Parameter<arma::vec>(track, label, temperature), self(p) {}

    ParameterWrap(PyObject *p, const Parameter<arma::vec>& x)
        : Parameter<arma::vec>(x), self(p) {}

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

private:
    PyObject* self;
};


struct StepWrap : Step, wrapper<Step>
{
    StepWrap(PyObject *p)
        : Step(), self(p) {}

    StepWrap(PyObject *p, const Step& x)
        : Step(x), self(p) {}

    // Pure virtual functions
    void DoStep() { this->get_override("DoStep")(); }
    void Start() { this->get_override("Start")(); }
    BaseParameter* GetParPointer() { return this->get_override("GetParPointer")(); }
    
    // Virtual functions with default implementations
    std::string ParameterLabel()
    {
        if (override ParameterLabel = this->get_override("ParameterLabel"))
            return ParameterLabel();
        return Step::ParameterLabel();
    }
    std::string default_ParameterLabel() { return this->Step::ParameterLabel(); }        

    std::string ParameterValue()
    {
        if (override ParameterValue = this->get_override("ParameterValue"))
            return ParameterValue();
        return Step::ParameterValue();
    }
    std::string default_ParameterValue() { return this->Step::ParameterValue(); }        

    bool ParameterTrack()
    {
        if (override ParameterTrack = this->get_override("ParameterTrack"))
            return ParameterTrack();
        return Step::ParameterTrack();
    }
    bool default_ParameterTrack() { return this->Step::ParameterTrack(); }        

private:
    PyObject* self;
};


// generate wrapper code for the classes
BOOST_PYTHON_MODULE(_yamcmcpp){
    // declare the classes to wrap. remember to exposure all
    // constructors!

    // Parameters:
    class_<Parameter<arma::vec>, ParameterWrap>("Parameter", init<bool,std::string,double>())
        // Virtual methods
        .def("StartingValue", &ParameterWrap::StartingValue)
        .def("LogDensity", &Parameter<arma::vec>::LogDensity, &ParameterWrap::default_LogDensity)
        .def("RandomPosterior", &Parameter<arma::vec>::RandomPosterior, &ParameterWrap::default_RandomPosterior)
        .def("Save", &Parameter<arma::vec>::Save, &ParameterWrap::default_Save)
        // Class methods
        .def("Value", &Parameter<arma::vec>::Value)
        .def("SetSampleSize", &Parameter<arma::vec>::SetSampleSize)
        .def("AddToSample", &Parameter<arma::vec>::AddToSample)
        .def("GetSamples", &Parameter<arma::vec>::GetSamples)
        // Base class methods (some of these are virtual but I don't want to deal with that for now)
        .def("GetLogDensity", &Parameter<arma::vec>::GetLogDensity)
        .def("SetLogDensity", &Parameter<arma::vec>::SetLogDensity)
        .def("GetTemperature", &Parameter<arma::vec>::GetTemperature)
        .def("StringValue", &Parameter<arma::vec>::StringValue)
        .def("Track", &Parameter<arma::vec>::Track)
        .def("SetTracking", &Parameter<arma::vec>::SetTracking)
        .def("Label", &Parameter<arma::vec>::Label)
        .def("SetSampleSize", &Parameter<arma::vec>::SetSampleSize)
        .def("AddToSample", &Parameter<arma::vec>::AddToSample)
        ;

    // Proposals:
    class_<NormalProposal>("NormalProposal",init<>())
        .def(init<double>())
        .def("Draw", &NormalProposal::Draw )
        .def("LogDensity", &NormalProposal::LogDensity )
        ;
    class_<StudentProposal>("StudentProposal",init<>())
        .def(init<double,double>())
        .def("Draw", &StudentProposal::Draw )
        .def("LogDensity", &StudentProposal::LogDensity )
        ;
    class_<MultiNormalProposal>("MultiNormalProposal",init<>())
        .def(init<arma::mat>())
        .def("Draw", &MultiNormalProposal::Draw )
        .def("LogDensity", &MultiNormalProposal::LogDensity )
        ;
    class_<LogNormalProposal>("LogNormalProposal",init<>())
        .def(init<double>())
        .def("Draw", &LogNormalProposal::Draw )
        .def("LogDensity", &LogNormalProposal::LogDensity )
        ;

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
    class_<Sampler>("Sampler",init<int,int,int>())
        .def("AddStep", &Sampler::AddStep)
        .def("Iterate", &Sampler::Iterate)
        .def("Run", &Sampler::Run)
        .def("NumberOfSteps", &Sampler::NumberOfSteps)
        .def("NumberOfTrackedSteps", &Sampler::NumberOfTrackedSteps)
        .def("GetTrackedNames", &Sampler::GetTrackedNames)
        .def("GetTrackedParams", &Sampler::GetTrackedParams)
        ;

    // Steps
    /*
    class_<Step, StepWrap>("Step", no_init)
        // Virtual methods
        .def("DoStep", &StepWrap::DoStep)
        .def("Start", &StepWrap::Start)
        //.def("GetParPointer", &StepWrap::GetParPointer)
        .def("ParameterLabel", &Step::ParameterLabel, &StepWrap::default_ParameterLabel)
        .def("ParameterValue", &Step::ParameterValue, &StepWrap::default_ParameterValue)
        .def("ParameterTrack", &Step::ParameterTrack, &StepWrap::default_ParameterTrack)
        ;
    */

    /*
    class_<AdaptiveMetro>("AdaptiveMetro", init<boost::shared_ptr<Parameter<arma::vec> >,
                          boost::shared_ptr<Proposal<double> >,arma::mat,double,int>())
        .def("Start", &AdaptiveMetro::Start)
        ;
    */       
   
}
