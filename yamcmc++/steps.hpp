//
//  steps.h
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __yamcmc____steps__
#define __yamcmc____steps__

#include <iostream>
/*
 *  steps.hpp
 *  mymcmc
 *
 *  Created by Brandon Kelly on 11/24/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __MCMC_STEPS__
#define __MCMC_STEPS__

// Standard includes
#include <string>
#include <sstream>
#include <cmath>
// Boost includes
#include <boost/ptr_container/ptr_vector.hpp>
// Local includes
#include "random.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Global object for generating random variables from various distributions,
// instantiated in steps.cpp
extern RandomGenerator RandGen;

/*! \brief Abstract parameter class.
 *
 * The model is defined as many instances of parameter subclasses.  Different Step types require different methods
 * to be implemented in each Parameter subclass.
 *
 * \section overwrite Methods that should be implemented in subclass:
 * Required for all step types:
 *   - Constructor that sets, at least, Parameter::track_ and Parameter::name_
 *   - Parameter::Value
 *   - Parameter::Save
 *
 * GibbsStep:
 *   - Parameter::StartingValue
 *   - Parameter::RandomPosterior
 *
 * Metropolis-Hastings:
 *   - Parameter::StartingValue
 *   - Parameter::LogDensity
 *
 * Slice:
 *   - Parameter::StartingValue
 *   - Parameter::LogDensity
 *
 * FunctionStep (deterministic):
 *   - Parameter::Function
 *
 */

////////// FUNCTION DEFINITIONS AND TEMPLATE FUNCTIONS /////////////

// Method to perform the rank-1 Cholesky update, needed for updating the
// proposal covariance matrix
void CholUpdateR1(arma::mat& S, arma::vec& v, bool downdate);

// Function to convert any streaming type to string
template <class T>
std::string to_string (const T& t) {
    std::stringstream ss;
	ss << t;
	std::string s = ss.str();
	return s;
}

/******************************************************************
 PARAMETER CLASSES
 *****************************************************************/

// This is the base Parameter class. It is abstract, so it should
// never be instantiated.

template<class ParValueType>
class Parameter {
public:
	/// Default class constructor.
	Parameter() {}
	/*! \brief Base class constructor
	 *
	 * \param track True if parameter should be tracked and saved.
	 * \param label Label of parameter for tracking purposes.
	 */
	Parameter(bool track, std::string label, double temperature=1.0) : track_(track), label_(label),
    temperature_(temperature) {}
	
	/// Function for deterministic nodes.
	/// \return Function value.
	virtual ParValueType Function() {
        ParValueType p;
        return p;
	}
	
	/// Starting value.
	/// \return Starting value.
	virtual ParValueType StartingValue() = 0;
	
	/// Log of the probability density (plus constant)
	/// \param value Value of parameter to evaluate density at.
	/// \return Log of probability density (plus constant)
	virtual double LogDensity(ParValueType value) {
		return 0.0;
    }
	
	// Return the current value of the log-posterior. Useful for
	// Metropolis steps so we don't have to compute the log-posterior
	// for the current value of the parameter more than once
	virtual double GetLogDensity() {
		return 0.0;
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
    
	/// Return a random draw from the posterior.
	/// Random draw from posterior is called by GibbsStep.
	/// \return Random draw from posterior.
	virtual ParValueType RandomPosterior() {
        ParValueType p;
		return p;
	}
	
	/// Value of parameter.
	/// \return Parameter value
	virtual ParValueType Value() = 0;
	
	// Return a string representation of parameter value
	virtual std::string StringValue() {
		return " ";
	}
	
	/// Save a new value of the parameter.
	/// \param new_value New value to save.
	virtual void Save(ParValueType new_value) = 0;
	
	/// Parameter is tracked / saved.
	/// \return True if parameter is tracked.
	bool Track() {
		return track_;
	}
	
    // Set whether a parameter is tracked or not
    void SetTracking(bool track) {
        track_ = track;
    }
    
	/// String label of parameter.
	/// \return Label of parameter, for purposes of saving output.
	std::string Label() {
		return label_;
	}
    
protected:
    double log_posterior_; // The log of the posterior distribution
    
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

/*! \defgroup steps Implemented MCMC Step Types
 *
 * MCMC algorithms typically consist of many steps.  While users will often want
 * to implement their own step types, a few that arise over and over are included here.
 * MetropStep and SliceStep in particular only require specification of
 * a log density function (minus an unknown constant).
 *
 */

/*! \brief Abstract base step class.
 *
 * \ingroup steps
 *
 * The step (Gibbs, Metropolis-Hastings, Slice, etc) describes how the sampler
 * calculates the next value of each parameter.
 */

class Step {
public:
	/*! \brief Take step.
	 *
	 * Executes the main step function, such as taking a MH step or deterministic step.
	 */
	virtual void DoStep() = 0;
	/*! \brief Set starting value.
	 *
	 *  Calls the parameter's Parameter::Starting function and then saves the result using Parameter::Save.
	 */
	virtual void Start() = 0;
	
	/*! \brief Return parameter label
	 *
	 * \return The label of the parameter associated with the step instance.
	 * \see Parameter::Label
	 */
	virtual std::string ParameterLabel() {
		return " ";
	}
	
	// Return a string representation of the parameter value
	virtual std::string ParameterValue() {
		return " ";
	}
	/*! Return the tracking status
	 *
	 * \return The tracking status of the parameter associated with the step instance.
	 * \see Parameter::Track
	 */
	virtual bool ParameterTrack() {
		return true;
	}
};

/*! \brief Gibbs step.
 *
 * \ingroup steps
 *
 * A Gibbs step draws from the Parameter::RandomPosterior function of a Parameter and then saves the result
 * using Parameter:Save.
 */
template <class ParValueType>
class GibbsStep: public Step {
public:
	/// Default constructor, for copy assignments, etc.
	GibbsStep() {}
	/*! \brief Main constructor taking a pointer to a Parameter object.
	 *  \param parameter A Parameter that implements Parameter::RandomPosterior.
	 */
	GibbsStep(Parameter<ParValueType>& parameter) : parameter_(parameter) {}
	
	/*! \brief Take a Gibbs step for the parameter.
	 *
	 *  Draws from the parameter's Parameter::RandomPosterior function and then
	 *	saves the result using Parameter::Save.
	 */
	void DoStep() {
		parameter_.Save(parameter_.RandomPosterior());
	}
	
	void Start() {
		parameter_.Save(parameter_.StartingValue());
	}
	
	// Return string of parameter label
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	// Return string representation of parameter value
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
	
private:
	/// Reference to parameter associated with step instance.
	Parameter<ParValueType>& parameter_;
};

/****************************************************************************
 
 Metropolis-Hastings steps. These will require both a Parameter and Proposal
 object.
 
 ****************************************************************************/


// Metropolis Hastings step. Requires both a Parameter and Proposal instance.

template <class ParValueType>
class MetropStep: public Step {
public:
	// \brief Main constructor taking a reference to a Parameter and Proposal object.
	MetropStep(Parameter<ParValueType>& parameter, Proposal<ParValueType>& proposal,
			   int report_iter=-1) :
	parameter_(parameter), proposal_(proposal), report_iter_(report_iter)
	{
		naccept_ = 0;
		niter_ = 0;
	}
	
	void Start() {
		parameter_.Save(parameter_.StartingValue());
	}
    
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	// Return string representation of parameter value
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
	
	bool Accept(ParValueType new_value, ParValueType old_value) {
		// MH accept/reject criteria
		double alpha_ = parameter_.LogDensity(new_value) - parameter_.GetLogDensity()
		+ proposal_.LogDensity(old_value, new_value) - proposal_.LogDensity(new_value, old_value);
		
		if (!arma::is_finite(alpha_)) {
			// New value of the log-posterior is not finite, so reject this
			// proposal
			return false;
		}
		
		double unif = uniform_(rng);
		alpha_ = std::min(exp(alpha_), 1.0);
		if (unif < alpha_) {
			return true;
		} else {
			return false;
		}
	}
	
	// Do Metropolis-Hastings Update
	void DoStep() {
		// Draw a new parameter
		ParValueType new_value = proposal_.Draw(parameter_.Value());
		ParValueType old_value = parameter_.Value();
		
		// MH accept/reject criteria
		if (Accept(new_value, old_value)) {
			naccept_++;
			parameter_.Save(new_value);
		}
		niter_++;
		
		if (niter_ == report_iter_) {
			// Give report on average acceptance rate
			Report();
		}
	}
    
	// Report on acceptance rates since last report
	void Report() {
		double arate = ((double)(naccept_)) / ((double)(niter_));
		std::cout << "Average Acceptance Rate Since Last Report: " << arate << std::endl;
		niter_ = 0;
		naccept_ = 0;
	}
    
    // Return if parameter is tracked.
    bool ParameterTrack() {
        return parameter_.Track();
    }
	
private:
	// References to parameter and proposal associated with step instance.
	Parameter<ParValueType>& parameter_;
	Proposal<ParValueType>& proposal_;
	boost::random::uniform_real_distribution<> uniform_;
	double alpha_; // Acceptance probability
	int naccept_; // The number of accepted steps
	int niter_; // The number of iterations performed
	int report_iter_; // The number of iterations until we print out the average acceptance rate
};

// Robust Adaptive Multivariate Normal proposal for Metropolis-Hastings (RAM).
//
// Reference: Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate,
//			  M. Vihola, 2012, Statistics & Computing, 22, 997-1008

class AdaptiveMetro : public Step
{
public:
	// Constructor
	AdaptiveMetro(Parameter<arma::vec>& parameter, Proposal<double>& proposal,
				  arma::mat proposal_covar, double target_rate, int maxiter);
    
	void Start() {
		parameter_.Save(parameter_.StartingValue());
	}
	
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	// Return string representation of parameter value
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
	
	// Method to set the target acceptance rate
	void SetTargetRate(double target_rate) {
		target_rate_ = target_rate;
	}
	
	// Method to set the rate at which the step size sequence decays.
	void SetDecayRate(double gamma) {
		gamma_ = gamma;
	}
	
	// Method to determine whether a proposal is accepted
	bool Accept(arma::vec new_value, arma::vec old_value);
	
	// Method to perform the RAM step.
	void DoStep();
    
    // Return if parameter is tracked.
    bool ParameterTrack() {
        return parameter_.Track();
    }
	
private:
	/// References to parameter and proposal associated with step instance.
	Parameter<arma::vec>& parameter_;
	Proposal<double>& proposal_;
	boost::random::uniform_real_distribution<> uniform_;
	arma::mat chol_factor_; // Cholesky factor of proposal scale matrix
	double gamma_; // Rate of decay for step size update
	double target_rate_; // Target acceptance rate
	int niter_; // Number of iterations performed
	int naccept_; // Number of MHA proposals accepted
	int maxiter_; // Maximum number of iterations to update proposal scale matrix
	double alpha_; // Acceptance probability
};

// Class performing the exchange step used in Parallel Tempering
template <class ParValueType, class ParameterType>
class ExchangeStep : public Step
{
public:
    // Constructor
    ExchangeStep(Parameter<ParValueType>& parameter, int parameter_index, Ensemble<ParameterType>& ensemble,
                 int report_iter=-1) :
    parameter_(parameter), parameter_index_(parameter_index), ensemble_(ensemble), report_iter_(report_iter)
    {
        // Always propose swap between ensemble_(parameter_index) and ensemble(parameter_index-1), so
        // make sure parameter_index > 0.
        BOOST_ASSERT(parameter_index_ > 0);
        naccept_ = 0;
        niter_ = 0;
    }
    
    // Method to initialize the parameter value
	void Start() {
		parameter_.Save(parameter_.StartingValue());
	}
	
	// Return string of parameter label
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	// Return string representation of parameter value
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
	
    // Do the exchange step.
    void DoStep() {
        
        // Get the log-posterior and the temperature for this parameter
        double this_logpost = parameter_.GetLogDensity();
        // Warmer temperature : this_temperature > other_temperature
        double this_temperature = parameter_.GetTemperature();
        
        // Do the same thing, but for the parameter we are trying to swap with
        double other_logpost = ensemble_[parameter_index_-1].GetLogDensity();
        double other_temperature = ensemble_[parameter_index_-1].GetTemperature();
        
        // Return the log-posterior values to the original Temp=1 scale
        this_logpost = this_logpost * this_temperature;
        other_logpost = other_logpost * other_temperature;
        
        // Calculate the logarithm of the metropolis-hastings ratio
        double alpha = 1.0 / this_temperature * (other_logpost - this_logpost) +
        1.0 / other_temperature * (this_logpost - other_logpost);
        
        // Perform metropolis-hastings update
		double unif = uniform_(rng);
		alpha = std::min(exp(alpha), 1.0);
		if (unif < alpha) {
            // Swap the parameter values
            ParValueType this_theta = parameter_.Value();
            parameter_.Save(ensemble_[parameter_index_-1].Value());
            ensemble_[parameter_index_-1].Save(this_theta);
            
            // Update the log-posterior values
            double this_logpost_new = other_logpost / this_temperature;
            parameter_.SetLogDensity(this_logpost_new);
            double other_logpost_new = this_logpost / other_temperature;
            ensemble_[parameter_index_-1].SetLogDensity(other_logpost_new);
            
            naccept_++;
		}
        
        niter_++;
        
		if (niter_ == report_iter_) {
			// Give report on average acceptance rate
			Report();
		}
    }
    
	// Report on acceptance rates since last report
	void Report() {
		double arate = ((double)(naccept_)) / ((double)(niter_));
		std::cout << "Average Exchange Acceptance Rate Since Last Report: " << arate << std::endl;
		niter_ = 0;
		naccept_ = 0;
	}
    
    // Return if parameter is tracked.
    bool ParameterTrack() {
        return parameter_.Track();
    }
	
private:
    Parameter<ParValueType>& parameter_; // The parameter associated with this step
    int parameter_index_; // The index of parameter_ in the ensemble
    Ensemble<ParameterType>& ensemble_; // The parameter ensemble
    int report_iter_; // Report on acceptance rates after this many iterations
    boost::random::uniform_real_distribution<> uniform_;
    int niter_; // The number of iterations performed since last report
    int naccept_; // The number of accepted exchanges since last report
};

#endif

#endif /* defined(__yamcmc____steps__) */
