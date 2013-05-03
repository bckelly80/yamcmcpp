//
//  parameters.hpp
//  yamcmc++
//
//  Created by Brandon Kelly on 4/15/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef yamcmc___parameters_hpp
#define yamcmc___parameters_hpp

// Standard includes
#include <iostream>
#include <string>
// Boost includes
#include <boost/ptr_container/ptr_vector.hpp>
// Local includes
#include "random.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Global object for generating random variables from various distributions,
// instantiated in steps.cpp
extern RandomGenerator RandGen;

/******************************************************************
 PARAMETER CLASSES
 *****************************************************************/

// This is the base Parameter class. It is abstract, so it should
// never be instantiated directly. Instead, users must subclass it when
// building their own parameter classes.

template<class ParValueType>
class Parameter {
public:
	/// Default class constructor.
	Parameter() {}
    
    // track: True if parameter should be tracked and saved.
    // label: Label of parameter for tracking purposes.
    // temperature: Temperature of parameter object. This is used by ensemble samplers. If you don't
    //              know what this is just keep it at the default of 1.0.
	Parameter(bool track, std::string label, double temperature=1.0) : track_(track), label_(label),
    temperature_(temperature) {}
	
	// Method to return the starting value for the parameter.
	virtual ParValueType StartingValue() = 0;
	
	// Method to return the log of the probability density (plus constant).
	// value: Value of parameter to evaluate density at.
	virtual double LogDensity(ParValueType value) {
		return 0.0;
    }
	
	// Return the current value of the log-posterior. Useful for
	// Metropolis steps so we don't have to compute the log-posterior
	// for the current value of the parameter more than once
	virtual double GetLogDensity() {
		return log_posterior_;
	}
	
    // Method to directly set the log-posterior of the parameter. Useful for certain steps
    // when we do not need to recalculate the posterior. Used in the exchange step.
    void SetLogDensity(double logpost) {
        log_posterior_ = logpost;
    }
    
    // Return the value of the temperature used. Primarily used in tempered transitions.
    double GetTemperature() {
        return temperature_;
    }
    
	// Return a random draw from the posterior.
	// Random draw from posterior is called by GibbsStep.
	virtual ParValueType RandomPosterior() {
        ParValueType p;
		return p;
	}
	
	// Return the current value of the parameter.
	ParValueType Value() {
        return value_;
    }
	
	// Return a string representation of the parameter value.
	virtual std::string StringValue() {
		return " ";
	}
	
	// Save a new value of the parameter.
	// new_value: New value to save.
	void Save(ParValueType new_value) {
        value_ = new_value;
        log_posterior_ = LogDensity(new_value);
    }
	
	// Parameter is tracked / saved. Return True if parameter is tracked.
	bool Track() {
		return track_;
	}
	
    // Set whether a parameter is tracked or not.
    void SetTracking(bool track) {
        track_ = track;
    }
    
	// Return the a string containing the parameter label. This is used to identify the parameter.
	std::string Label() {
		return label_;
	}
    
protected:
    double log_posterior_; // The log of the posterior distribution
    ParValueType value_; // The current value of the parameter
    // Temperature value, used when doing tempered steps. By default this is one.
    double temperature_;
    
	/// Should this variable be tracked?
	bool track_;
	/// Name of variable for tracking purposes.
	std::string label_;
};

// This is the Ensemble class. It is basically a class
// containing a pointer container (boost::ptr_vector) that allows
// one to collect an ensemble of objects. This is the basis for
// the Ensemble MCMC samplers, which require both an ensemble of
// parameter and proposal objects.

template <class EnsembleType>
class Ensemble {
public:
    // Empty constructor
    Ensemble() {}
    
    // Add parameter object to the parameter ensemble
    void AddObject(EnsembleType* pObject) {
        the_objects_.push_back(pObject);
    }
    
    // Return the number of parameters in the ensemble
    int size() {
        return the_objects_.size();
    }
    
    // Access elements of the parameter ensemble. This done by overloading
    // the [] operator so that one can obtain a reference to a parameter
    // as parameter = ensemble[i].
    EnsembleType& operator [] (const int index)
    {
        return the_objects_[index];
    }
    
    EnsembleType const & operator [] (const int index) const
    {
        return the_objects_[index];
    }
    
    //private:
    boost::ptr_vector<EnsembleType> the_objects_;
};

#endif