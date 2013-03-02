//
//  samplers.h
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __yamcmc____samplers__
#define __yamcmc____samplers__

#include <iostream>
/*
 *  mymcmc.hpp
 *  mymcmc
 *
 *  Created on 11/18/12 by
 *
 *     Dr. Brandon C. Kelly
 *     Department of Physics
 *     University of California, Santa Barbara
 *     (bckelly80@gmail.com)
 *
 *  Routines to perform Markov Chain Monte Carlo (MCMC) adopted from
 *
 *  	Scythe MCMC - A Scythe Markov Chain Monte Carlo C++ Framework
 *		by Tristan Zajonc (tristanz@gmail.com)
 *
 *	I have adopted most of the classes and functions from Scythe MCMC, but
 *  have modified them to remove the dependency on the Scythe Statistical
 *  library. My set of MCMC classes and functions depends on the BOOST
 *	libraries. In addition, I have introduced a multivariate Metropolis
 *	step that updates a vector of parameters.
 */
#ifndef __MY_MCMC__
#define __MY_MCMC__

// Standard includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <fstream>
#include <limits>
// Boost includes
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
// Local includes
#include "random.hpp"
#include "steps.hpp"

/*! \brief Basic MCMC options.
 *
 * These options are generally specified at the command line and are generic to all
 * MCMC samplers.  Other options are passed through an .ini config file.
 */
struct MCMCOptions {
	/// Sample size. (iterations - burnin)/(thin + 1).
	int sample_size;
	/// Thinning interval
	int thin;
	/// Burn in period.
	int burnin;
	/// Chains.
	int chains;
	/// Data file
	std::string data_file;
	/// Out file to save parameter chains.
	std::string out_file;
};

/********************
 FUNCTION PROTOTYPES
 ********************/

// Function to get working directory (the directory that the executable
// was called from)
std::string get_initial_directory();

// Function to prompt user for MCMC parameters and return a structure
// containing those parameters.
MCMCOptions mcmc_options_prompt(std::string idirectory);

// Function to test if we can write to a file
bool write_to_file(std::string filename);

// Function to test if we can read a file
bool read_from_file(std::string filename);

// Function to test if two doubles are within the machine precision of each other.
bool approx_equal(double a, double b);

/*! MCMC sampler.
 *
 *  The sampler is the main MCMC object that holds all the MCMC steps for each parameter.  Running the sampler
 *  performs MCMC sampling for the model, saving results, and displaying progress.  In the language of the Command Pattern, the
 *  sampler is the Invoker or Command Manager.
 *
 *  After instantiating the sampler, users should add all the required steps using the Sampler::AddStep method, which places
 *  each step onto a stack. The entire sampling process is run using Sampler::Run.
 */
class Sampler {
public:
    // Constructor to initialize sampler. Takes a MCMCOptions struct as input.
    Sampler(MCMCOptions& options);
	
    // Method to add Step to Sampler execution stack.
    void AddStep(Step* step);
	
    // Run sampler for a specific number of iterations.
    void Iterate(int number_of_iterations, bool progress = false);
	
	// Run MCMC sampler.
    void Run();
	
    // Return number of steps in one sampler iteration.
    int NumberOfSteps() {
        return steps_.size();
    }
    
    // Return number of tracked steps in one sampler iteration.
    int NumberOfTrackedSteps() {
        return tracks_.size();
    }
    
    // Save the parameter values after a iteration to a file
    virtual void SaveValues(std::ofstream& outfile);
    
protected:
    int sample_size_;
    int burnin_;
    int thin_;
	std::string out_file_;
    boost::ptr_vector<Step> steps_;
	std::vector<int> tracks_;
};

/*
 Ensemble MCMC sampler. This is basically the same as the normal MCMC sampler class, except
 we need to modify the SaveValues method to print out the values for each walker on a
 new line for a single iteration.
 */
class EnsembleSampler : public Sampler
{
public:
	// Constructor to initialize sampler. Takes a MCMCOptions struct as input.
	EnsembleSampler(MCMCOptions& options) : Sampler(options) {};
    
	// Method to print out the parameter values to a files
    void SaveValues(std::ofstream& outfile);
};

#endif

#endif /* defined(__yamcmc____samplers__) */
