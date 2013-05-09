//
//  yamcmc_test.cpp
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

/*
 Code to test mymcmc routines.
 
 Testing History:
 
 11/24/2012: Tested univariate random number generator object, uRand, with default inputs: PASS
 11/24/2012: Tested univariate random number generator object with non-default inputs: PASS
 12/01/2012: Tested gibbs sampler for normal distribution with unknown mean and variance: PASS
 12/01/2012: Testing metropolis sampler for normal distribution with unknown mean and variance: PASS
 12/06/2012: Test Gibbs sampler and MHA using SAME data set: PASS
 12/13/2012: Testing multivariate normal random number generator: PASS
 12/15/2012: Add in multivariate metropolis-step: PASS
 12/17/2012: Tested rank-1 update for cholesky decomposition: PASS
 12/17/2012: Test adaptive metropolis step: PASS
 12/18/2012: Need to figure out how to optimize
 
 */

#include <iostream>
#include <fstream>
#include <string>
#include "samplers.hpp"

// Externally-defined random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Externally-defined object containing methods to generate random numbers from
// several common univariate distributions.
extern RandomGenerator RandGen;

/*
 First setup parameter class definitions
 */

// Define class for Gaussian mean parameter
class Mu : public Parameter<double> {
public:
    Mu() : Parameter<double>() {}
	// Overidden constructor
	Mu(bool track, std::string name, double temperature=1.0) : Parameter<double>(track, name, temperature) {}
	double RandomPosterior();
	double StartingValue();
	double LogDensity(double mu_value);
	// Prior is normal
	void SetPrior(double prior_mean, double prior_var);
	void SetData(double* data, int ndata);
    void SetVariancePointer(Parameter<double>* p_var);
    double GetVariance() const;
private:
    double mu_;
	double prior_mean_;
	double prior_var_;
    Parameter<double>* p_variance_; // Pointer to variance parameter
	double* data_; // Pointer to data array
	int ndata_;
};

// Define class for Gaussian variance parameter
class SigmaSqr : public Parameter<double> {
public:
	SigmaSqr() : Parameter<double>() {}
	SigmaSqr(bool track, std::string name, double temperature=1.0) : Parameter<double>(track, name, temperature) {}
	double RandomPosterior();
	double StartingValue();
	double LogDensity(double sigsqr_value);
	// Prior is inverse-gamma
	void SetPrior(double alpha, double beta);
	void SetData(double* data, int ndata);
	void SetMeanPointer(Parameter<double>* p_mu);
    double GetMean() const;
private:
    double var_;
	double alpha_;
	double beta_;
	Parameter<double>* p_mu_; // Reference to mean parameter
	double* data_; // Pointer to data array
	int ndata_;
};

class Theta : public Parameter<arma::vec> {
public:
	Theta() : Parameter<arma::vec>() {}
	Theta(bool track, std::string name, double temperature=1.0) : Parameter<arma::vec>(track, name, temperature) {}
	arma::vec StartingValue();
	std::string StringValue();
	double LogDensity(arma::vec theta_value);
 	// Prior is independent for theta = (mu,sigsqr). Prior for mu is
	// normal, while prior for sigsqr is inverse-gamma
	void SetPrior(double prior_mean, double prior_var, double alpha, double beta);
	void SetData(double* data, int ndata);
private:
	// Prior parameters
	double prior_mean_;
	double prior_var_;
	double alpha_;
	double beta_;
	double* data_; // Pointer to data array
	int ndata_;
};

// Function Prototypes
double variance(double* values, int size);
double mean(double* values, int size);
void test_mcmc();
void test_random();
void test_ensemble_mcmc();

int main(int argc, const char * argv[])
{
    // Run some tests...
    test_ensemble_mcmc();
    return 0;
}

// Function to test the random number generators

void test_random() {
	
	arma::mat Covar(3,3);
	arma::mat Corr(3,3);
	arma::vec sigma(3);
	
	sigma(0) = 2.0;
	sigma(1) = 3.4;
	sigma(2) = 0.43;
	
	std::stringstream ss;
	(sigma.t()).raw_print(ss);
	std::string sigstr = ss.str();
	
	std::cout << "Sigma String: " << sigstr << std::endl;
}

// Function to test the MCMC sampler

void test_mcmc () {
	
	// First find out which directory we are in
	//std::string idirectory = get_initial_directory();
    std::string idirectory = "./";
    std::cout << idirectory << std::endl;
    
    // Generate fake data
    double true_mean = 3.4;
    double true_var = 2.3;
    
    int nsample = 1000;
    double* simulated_data;
    simulated_data = new double [nsample];
    
    for (int i=0; i<nsample; i++) {
        simulated_data[i] = RandGen.normal(true_mean, sqrt(true_var));
    }
    
    // Save simulated data to file
    std::ofstream outfile("normal.dat");
    for (int i=0; i<nsample; i++) {
        outfile << simulated_data[i] << std::endl;
    }
    
	// Prompt user for MCMC parameters
	MCMCOptions mcmc_options = mcmc_options_prompt(idirectory);
	
	// Initialize parameter objects
	Theta NormalTheta(true, "theta = (mu,sigsqr)");
	
	// Mu NormalMean(true, "mu");
	// SigmaSqr NormalVariance(true, "variance");
	
    // Add in pointers so the two parameters know about eachother
    // NormalMean.SetVariancePointer(&NormalVariance);
    // NormalVariance.SetMeanPointer(&NormalMean);
    
	// Grab Data
	double* data;
	data = new double [nsample];
	
	std::ifstream datafile("normal.dat");
	for (int i=0; i<nsample; i++) {
		datafile >> data[i];
	}
    
	NormalTheta.SetData(simulated_data, nsample);
	// NormalMean.SetData(data, nsample);
	// NormalVariance.SetData(data, nsample);
    
    
	// Set prior parameters
	double prior_mean = 0.0, prior_variance = 1.0e6;
	double prior_alpha = 0.0, prior_beta = 0.0;
	
	NormalTheta.SetPrior(prior_mean, prior_variance, prior_alpha, prior_beta);
	
	// NormalMean.SetPrior(prior_mean, prior_variance);
	// NormalVariance.SetPrior(prior_alpha, prior_beta);
    
	// Instantiate Metropolis-Hastings proposal objects
	
	arma::mat prop_covar(2,2);
	prop_covar.eye();
    prop_covar = prop_covar * 1e-2;
	
	NormalProposal ThetaProp(1.0);
    
	// double mean_prop_sigma = 0.05;
	// NormalProposal MeanProp(mean_prop_sigma);
	// double var_prop_sigma = 0.05;
	// NormalProposal VarProp(var_prop_sigma);
	
    // Instantiate MCMC Sampler and add pointers to step objects
	Sampler normal_model(mcmc_options);
	
	normal_model.AddStep(new AdaptiveMetro(NormalTheta, ThetaProp,
										   prop_covar, 0.25, mcmc_options.burnin));
	
    // 	  normal_model.AddStep(new MetropStep<double>(NormalMean, MeanProp));
    //	  normal_model.AddStep(new MetropStep<double>(NormalVariance, VarProp));
    //    normal_model.AddStep(new GibbsStep<double>(NormalMean));
    //    normal_model.AddStep(new GibbsStep<double>(NormalVariance));
    
    // Now run the MCMC sampler. The samples will be dumped in the
    // output file provided by the user.
    normal_model.Run();
 
    delete [] simulated_data;
    delete [] data;
}

void test_ensemble_mcmc () {
	
	// First find out which directory we are in
	//std::string idirectory = get_initial_directory();
    std::string idirectory = ".";
    std::cout << idirectory << std::endl;
    
    // Generate fake data
    double true_mean = 3.4;
    double true_var = 2.3;
    
    int nsample = 1000;
    double* simulated_data;
    simulated_data = new double [nsample];
    
    for (int i=0; i<nsample; i++) {
        simulated_data[i] = RandGen.normal(true_mean, sqrt(true_var));
    }
    
    // Save simulated data to file
    std::ofstream outfile("normal.dat");
    for (int i=0; i<nsample; i++) {
        outfile << simulated_data[i] << std::endl;
    }
    
	// Prompt user for MCMC parameters
	MCMCOptions mcmc_options = mcmc_options_prompt(idirectory);
    
	// Grab Data
	double* data;
	data = new double [nsample];
	
	std::ifstream datafile("normal.dat");
	for (int i=0; i<nsample; i++) {
		datafile >> data[i];
	}
    
    // Set prior parameters
	double prior_mean = 0.0, prior_variance = 1.0e6;
	double prior_alpha = 0.0, prior_beta = 0.0;
    
    // Construct the parameter ensemble
    Ensemble<Theta> ThetaEnsemble;
    int nwalkers = 10;
    
    // Set the temperature ladder. The logarithms of the temperatures are on a linear grid
    double max_temperature = 100.0;
    
    arma::vec temp_ladder = arma::linspace<arma::vec>(0.0, log(max_temperature), nwalkers);
    temp_ladder = arma::exp(temp_ladder);
    
    // Add the parameters to the ensemble, starting with the coolest chain
	for (int i=0; i<nwalkers; i++) {
		// Add this walker to the ensemble
        ThetaEnsemble.AddObject(new Theta(false, "theta", temp_ladder(i)));
		// Set the prior parameters
        ThetaEnsemble[i].SetPrior(prior_mean, prior_variance, prior_alpha, prior_beta);
        ThetaEnsemble[i].SetData(simulated_data, nsample);
	}
    
	// Instantiate Metropolis-Hastings proposal objects
	
	arma::mat prop_covar(2,2);
	prop_covar.eye();
    prop_covar = prop_covar * 1e-2;
	
	NormalProposal ThetaProp(1.0);
	
    // Instantiate MCMC Sampler and add pointers to step objects
	Sampler normal_model(mcmc_options);
	
    int report_iter = mcmc_options.burnin + mcmc_options.thin * mcmc_options.sample_size;
    
    for (int i=nwalkers-1; i>0; i--) {
        normal_model.AddStep(new AdaptiveMetro(ThetaEnsemble[i], ThetaProp, prop_covar, 0.4,
                                               mcmc_options.burnin));
        normal_model.AddStep(new ExchangeStep<arma::vec, Theta>(ThetaEnsemble[i], i, ThetaEnsemble, report_iter));
    }

    ThetaEnsemble[0].SetTracking(true);
    normal_model.AddStep(new AdaptiveMetro(ThetaEnsemble[0], ThetaProp, prop_covar, 0.4,
                                           mcmc_options.burnin));
	
    // 	  normal_model.AddStep(new MetropStep<double>(NormalMean, MeanProp));
    //	  normal_model.AddStep(new MetropStep<double>(NormalVariance, VarProp));
    //    normal_model.AddStep(new GibbsStep<double>(NormalMean));
    //    normal_model.AddStep(new GibbsStep<double>(NormalVariance));
    
    // Now run the MCMC sampler. The samples will be dumped in the
    // output file provided by the user.
    normal_model.Run();
    delete [] simulated_data;
    delete [] data;
}

//
/* Here define the methods for the parameters, first for Mu then SigmaSqr */
//

// Method to perform the Gibbs update. Returns a random draw of the mean
// conditional on the data and the current value of the variance
double Mu::RandomPosterior()
{
	double data_mean = mean(data_, ndata_);
    double sigsqr = GetVariance();
	double condvar = 1.0 / (1.0 / prior_var_ + ndata_ / sigsqr);
	double condmean = condvar * (prior_mean_ / prior_var_ + ndata_ * data_mean / sigsqr);
	double cond_sigma = sqrt(condvar);
	return RandGen.normal(condmean, cond_sigma);
}

// Method to generate the starting value of the mean by setting it equal
// to the data average.
double Mu::StartingValue()
{
	double data_mean = mean(data_, ndata_);
	return data_mean;
}

// Method to calculate the log-posterior given some value of mu
double Mu::LogDensity(double mu_value)
{
	double log_posterior;
	double data_mean = mean(data_, ndata_);
    double sigsqr = GetVariance();
	log_posterior = -0.5 * (data_mean - mu_value) *
	(data_mean - mu_value) * ((double)(ndata_)) / sigsqr -
	(mu_value - prior_mean_) * (mu_value - prior_mean_) / prior_var_;
	
	return log_posterior;
}

// Method to set the prior mean and variance for mu
void Mu::SetPrior(double prior_mean, double prior_var)
{
	prior_mean_ = prior_mean;
	prior_var_ = prior_var;
}

// Method to set the data parameters: pointer to the data array and number of data points
void Mu::SetData(double* data, int ndata)
{
	data_ = data;
	ndata_ = ndata;
}

// Method to set the pointer to the variance parameter object
void Mu::SetVariancePointer(Parameter<double>* p_var)
{
    p_variance_ = p_var;
}

// Method to return the current value of the variance
double Mu::GetVariance() const
{
    return p_variance_->Value();
}

/* Method definitions for SigmaSqr */

// Method to perform the Gibbs update. Returns a random draw of the variance
// conditional on the data and the current value of the mean
double SigmaSqr::RandomPosterior()
{
	double alpha1 = alpha_ + ((double)(ndata_)) / 2.0;
    double mu = GetMean();
	double data_rss = 0.0;
	for (int i=0; i<ndata_; i++) {
		data_rss += (data_[i] - mu) * (data_[i] - mu);
	}
	double beta1 = beta_ + 0.5 * data_rss;
	
	return RandGen.invgamma(alpha1, beta1);
}

// Method to generate the starting value of the variance by setting it equal
// to the data variance.
double SigmaSqr::StartingValue()
{
	double sigsqr = variance(data_, ndata_);
	return sigsqr;
}

// Method to calculate the log-posterior given some value of mu
double SigmaSqr::LogDensity(double sigsqr_value)
{
	double log_posterior;
	double alpha1 = alpha_ + ((double)(ndata_)) / 2.0;
    double mu = GetMean();
	double data_rss = 0.0;
	for (int i=0; i<ndata_; i++) {
		data_rss += (data_[i] - mu) * (data_[i] - mu);
	}
	double beta1 = beta_ + 0.5 * data_rss;
	
	log_posterior = -(alpha1 + 1.0) * log(sigsqr_value) - beta1 / sigsqr_value;
	
	if (sigsqr_value < 0) {
		log_posterior = -1e300;
	}
	
	return log_posterior;
}

// Method to set the values for the inverse-gamma prior on SigmaSqr
void SigmaSqr::SetPrior(double alpha, double beta)
{
	alpha_ = alpha;
	beta_ = beta;
}

// Method to set the data parameters: a pointer to the data array and number of data points
void SigmaSqr::SetData(double* data, int ndata)
{
	data_ = data;
	ndata_ = ndata;
}

// Method to set the pointer to the Mean parameter object
void SigmaSqr::SetMeanPointer(Parameter<double>* p_mu)
{
    p_mu_ = p_mu;
}

// Method to return the value of the mean parameter
double SigmaSqr::GetMean() const
{
    return p_mu_->Value();
}

// Method to set the prior for theta = (mu,sigsqr)
void Theta::SetPrior(double prior_mean, double prior_var, double alpha, double beta) {
	prior_mean_ = prior_mean;
	prior_var_ = prior_var;
	alpha_ = alpha;
	beta_ = beta;
}

// Method to set the point to the data array
void Theta::SetData(double* data, int ndata)
{
	data_ = data;
	ndata_ = ndata;
}

// Method to generate the starting value of theta by setting it equal
// to the data average and variance.
arma::vec Theta::StartingValue()
{
	double data_mean = mean(data_, ndata_);
	double data_var = variance(data_, ndata_);
	
	arma::vec itheta(2);
	itheta(0) = data_mean;
	itheta(1) = data_var;
    
	return itheta;
}

// Method to return a string representation of the parameter vector theta
std::string Theta::StringValue()
{
	std::stringstream ss;
	(value_.t()).raw_print(ss);
	std::string theta_str = ss.str();
	return theta_str;
}

// Method to calculate the log-posterior given some value of mu
double Theta::LogDensity(arma::vec theta_value)
{
	double log_posterior, log_likhood, log_prior_mu, log_prior_sigsqr;
    
	log_likhood = 0.0;
	for (int i=0; i<ndata_; i++) {
		log_likhood -= 0.5 * (data_[i] - theta_value(0)) * (data_[i] - theta_value(0)) / theta_value(1);
	}
	
	log_likhood += -0.5 * ((double)(ndata_)) * log(theta_value(1));
	
	log_prior_mu = -0.5 * (theta_value(0) - prior_mean_) * (theta_value(0) - prior_mean_) / prior_var_;
	
	log_prior_sigsqr = -(alpha_ + 1) * log(theta_value(1)) - beta_ / theta_value(1);
	
	log_posterior = log_likhood + log_prior_mu + log_prior_sigsqr;

	log_posterior = log_posterior;
    
	if (theta_value(1) < 0) {
		log_posterior = -1e300;
	}
	
	return log_posterior;
}

//
// FUNCTION DEFINITIONS
//


// Function to return the mean value of an array
double mean(double* values, int size)
{
	double avg = 0.0;
	for (int i=0; i<size; i++) {
		avg += values[i];
	}
	avg = avg / ((double)(size));
	return avg;
}

// Function to return the variance of an array
double variance(double* values, int size)
{
	double mean_sqr = 0.0, mean_val = 0.0;
	for (int i=0; i<size; i++) {
		mean_val += values[i];
		mean_sqr += values[i] * values[i];
	}
	mean_sqr = mean_sqr / ((double)(size));
	mean_val = mean_val / ((double)(size));
	double sigsqr = mean_sqr - mean_val * mean_val;
	
	return sigsqr;
}


