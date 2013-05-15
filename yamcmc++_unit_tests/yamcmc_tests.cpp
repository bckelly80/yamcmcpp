//
//  main.cpp
//  yamcmc++_unit_tests
//
//  Created by Brandon Kelly on 5/14/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "samplers.hpp"

// Externally-defined random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Externally-defined object containing methods to generate random numbers from
// several common univariate distributions.
extern RandomGenerator RandGen;


/*******************************************************************************
 *                                                                             *
 *                      TEST FOR PARAMETER CLASS                               *
 *                                                                             *
 *******************************************************************************/

// Define class for univariate normal mean parameter
class Mu : public Parameter<double> {
public:
	// Constructor
	Mu(bool track, std::string name, double sigma, arma::vec& data, double temperature=1.0) :
        Parameter<double>(track, name, temperature), sigma_(sigma), data_(data)
        {
            sigsqr_ = sigma * sigma;
            data_mean_ = arma::mean(data_);
        }
    
    // Generate a random draw of mu from its posterior
	double RandomPosterior() {
        double condvar = 1.0 / (1.0 / prior_var_ + data_.size() / sigsqr_);
        double condmean = condvar * (prior_mean_ / prior_var_ + data_.size() * data_mean_ / sigsqr_);
        double cond_sigma = sqrt(condvar);
        return RandGen.normal(condmean, cond_sigma);
    }
    // Generate the initial value for mu by drawing from its posterior
	double StartingValue() {
        double initial_mu = data_mean_ + sigma_ / sqrt(data_.size()) * RandGen.normal();
        return initial_mu;
    }
    // Return the log-posterior
	double LogDensity(double mu_value) {
        double log_posterior;
        log_posterior = -0.5 * (data_mean_ - mu_value) * (data_mean_ - mu_value) * data_.size() / sigsqr_ -
        0.5 * (mu_value - prior_mean_) * (mu_value - prior_mean_) / prior_var_;
        return log_posterior;
    }
	// Prior is normal
	void SetPrior(double prior_mean, double prior_var) {
        prior_mean_ = prior_mean;
        prior_var_ = prior_var;
    }
    
private:
    double sigma_;
    double data_mean_;
    double sigsqr_;
	double prior_mean_;
	double prior_var_;
    arma::vec& data_;
};

// Define class for bivariate normal mean parameter
class BiMu : public Parameter<arma::vec> {
public:
	// Constructor
	BiMu(bool track, std::string name, arma::mat covar, arma::mat& data, double temperature=1.0) :
    Parameter<arma::vec>(track, name, temperature), data_(data)
    {
        covar_ = arma::symmatl(covar); // make sure covar is symmetric
        data_mean_ = arma::trans(arma::mean(data_,0));
    }
    
    // Generate a random draw of mu from its posterior
    arma::vec RandomPosterior() {
        arma::mat condvar = arma::inv(prior_var_.i() + data_.size() * covar_.i());
        arma::vec condmean = condvar * (prior_var_.i() * prior_mean_ + data_.size() * covar_.i() * data_mean_);
        return condmean + RandGen.normal(condvar);
    }
    
    // Generate the initial value for mu by drawing from its posterior
    arma::vec StartingValue() {
        arma::mat icovar = covar_ / sqrt(data_.size());
        arma::vec initial_mu = data_mean_ + RandGen.normal(icovar);
        return initial_mu;
    }
    // Return the log-posterior
	double LogDensity(arma::vec mu_value) {
        double log_posterior;
        arma::vec zcent = data_mean_ - mu_value;
        arma::vec prior_cent = mu_value - prior_mean_;
        log_posterior = -0.5 * data_.size() * arma::as_scalar(zcent.t() * covar_.i() * zcent) -
        0.5 * arma::as_scalar(prior_cent.t() * prior_var_.i() * prior_cent);
        return log_posterior;
    }
	// Prior is normal
	void SetPrior(arma::vec prior_mean, arma::mat prior_var) {
        prior_mean_ = prior_mean;
        prior_var_ = prior_var;
    }
    
private:
    arma::mat covar_;
    arma::vec data_mean_;
    arma::vec prior_mean_;
    arma::mat prior_var_;
    arma::mat& data_;
};

TEST_CASE("parameters/normal_mean", "Test the parameter class for a normal mean.") {
    double sigma = 2.3;
    double mu0 = 6.7;
    
    // Generate some data
    unsigned int ndata = 1000;
    arma::vec data(ndata);
    data.randn();
    data *= sigma;
    data += mu0;
    
    Mu NormalMean(true, "mu", sigma, data);
    
    double prior_mu = -1.0;
    double prior_var = 20.0;
    NormalMean.SetPrior(prior_mu, prior_var);
    
    // Run some simple tests for base parameter class
    REQUIRE(NormalMean.GetTemperature() == 1.0); // Make sure temperature is set correctly
    double initial_mu = NormalMean.StartingValue();
    NormalMean.Save(initial_mu);
    REQUIRE(NormalMean.Value() == initial_mu); // Make sure value is saved
    double initial_logpost = NormalMean.LogDensity(initial_mu);
    REQUIRE(NormalMean.GetLogDensity() == initial_logpost); // Make sure log-posterior for new value is saved
    
    // Run some tests specific to the normal mean parameter class
    unsigned int ndraws = 100000;
    double postvar = 1.0 / (1.0 / prior_var + ndata / sigma / sigma);
    double postmean = postvar * (prior_mu / prior_var + ndata * arma::mean(data) / sigma / sigma);
    
    double chisqr = 0.0;
    for (int i=0; i<ndraws; i++) {
        double mu_draw = NormalMean.RandomPosterior();
        chisqr += (mu_draw - postmean) * (mu_draw - postmean) / postvar;
    }
    
    double chisqr_zscr = abs(chisqr - ndraws) / sqrt(2.0 * ndraws);
    
    REQUIRE(chisqr_zscr < 4.0); // Make sure RandomPosterior is working
    
    double mu1 = 0.81 * mu0;
    double mu2 = 1.13 * mu0;
    double logdens1 = NormalMean.LogDensity(mu1);
    double logdens2 = NormalMean.LogDensity(mu2);
    double logratio = logdens1 - logdens2;
    
    logdens1 = -0.5 * log(postvar) - 0.5 * (mu1 - postmean) * (mu1 - postmean) / postvar;
    logdens2 = -0.5 * log(postvar) - 0.5 * (mu2 - postmean) * (mu2 - postmean) / postvar;
    double logratio0 = logdens1 - logdens2;
    
    double fracdiff = abs((logratio0 - logratio) / logratio0);
    REQUIRE(fracdiff < 1e-6);
}

/*******************************************************************************
 *                                                                             *
 *                      TESTS FOR PROPOSAL CLASSES                             *
 *                                                                             *
 *******************************************************************************/

TEST_CASE("proposals/normal_proposal", "Test the univariate normal proposal object.") {
    double sigma = 2.3;
    double mu = 6.7;
    
    NormalProposal UnivNormProp(2.3);
    
    unsigned int ndraws = 100000;
    double chisqr = 0.0;
    arma::vec x(ndraws);
    for (int i=0; i<ndraws; i++) {
        double xdraw = UnivNormProp.Draw(mu);
        x(i) = xdraw;
        chisqr += (xdraw - mu) * (xdraw - mu) / sigma / sigma;
    }
    double xmean = arma::mean(x);
    
    // Make sure sample mean and chi-square is within 4-sigma of their true values
    double xmean_zscr = abs(xmean - mu) / (sigma / sqrt(ndraws));
    REQUIRE(xmean_zscr < 4.0);
    double chisqr_zscr = abs(chisqr - ndraws) / sqrt(2.0 * ndraws);
    REQUIRE(chisqr_zscr < 4.0);
}

TEST_CASE("proposals/multinorm_proposal", "Test the multivariate normal proposal object.") {
    
    arma::vec mu(3);
    mu << 0.3 << 12.0 << -6.5;
    
    arma::mat corr(3,3);
    corr << 1.0 << 0.3 << -0.5 << arma::endr
    << 0.3 << 1.0 << 0.54 << arma::endr
    << -0.5 << 0.54 << 1.0;
    
    arma::mat sigma(3,3);
    sigma << 2.3 << 0.0 << 0.0 << arma::endr
    << 0.0 << 0.45 << 0.0 << arma::endr
    << 0.0 << 0.0 << 13.4;
    
    arma::mat covar = sigma * corr * sigma;
    arma::mat covar_inv = covar.i();
    
    MultiNormalProposal MultiNormProp(covar);
    
    unsigned int ndraws = 100000;
    double chisqr = 0.0;
    for (int i=0; i<ndraws; i++) {
        arma::vec xdraw = MultiNormProp.Draw(mu);
        xdraw -= mu;
        chisqr += arma::as_scalar(xdraw.t() * covar_inv * xdraw);
    }
    
    // Test by comparing squared standardized residuals with chi-square distribution
    double chisqr_zscr = abs(chisqr - 3 * ndraws) / sqrt(2.0 * 3 * ndraws);
    REQUIRE(chisqr_zscr < 4.0);
}

TEST_CASE("proposals/student_proposal", "Test the student's t proposal object.") {
    double sigma = 2.3;
    double mu = 6.7;
    int dof = 8;
    double tvar = sigma * sigma * dof / (dof - 2.0);
    unsigned int ndraws = 1000000;
    
    StudentProposal tProp(dof, sigma);
    
    arma::vec xdraws(ndraws);
    for (int i=0; i<ndraws; i++) {
        xdraws(i) = tProp.Draw(mu);
    }
    
    // Make sure sample mean is within 4-sigma of the true mean, assuming the central limit theorem
    double xmean_zscr = abs(arma::mean(xdraws) - mu) / sqrt(tvar / ndraws);
    REQUIRE(xmean_zscr < 4.0);
}

TEST_CASE("proposals/lognormal", "Test the log-normal proposal class.") {
    double mu = 2.3;
    double sigma = 4.5;
    unsigned int ndraws = 100000;
    
    LogNormalProposal LgNormProp(sigma);
    double chisqr = 0.0;
    for (int i=0; i<ndraws; i++) {
        double xdraw = LgNormProp.Draw(mu);
        chisqr += (log(xdraw) - log(mu)) * (log(xdraw) - log(mu)) / sigma / sigma;
    }
    
    // Test by comparing squared standardized residuals with chi-square distribution
    double chisqr_zscr = abs(chisqr - ndraws) / sqrt(2.0 * ndraws);
    REQUIRE(chisqr_zscr < 4.0);
    
    // Test LogDensity method of LogNormalProposal
    double old_value = 4.5;
    double new_value = 2.0;
    double log_ratio = LgNormProp.LogDensity(new_value, old_value) - LgNormProp.LogDensity(old_value, new_value);
    
    double var = sigma * sigma;
    double log_density1 = -0.5 * log(var) - log(new_value) -
    0.5 * (log(new_value) - log(old_value)) * (log(new_value) - log(old_value)) / var;
    double log_density2 = -0.5 * log(var) - log(old_value) -
    0.5 * (log(new_value) - log(old_value)) * (log(new_value) - log(old_value)) / var;
    double log_ratio0 = log_density1 - log_density2;
    double fracdiff = abs((log_ratio - log_ratio0) / log_ratio0);
    REQUIRE(fracdiff < 1e-6);
}

TEST_CASE("proposals/stretch", "Test the stretch proposal.") {
    arma::mat covar(2,2);
    covar << 2.3 << -0.6 << arma::endr << -0.6 << 0.8 << arma::endr;
    arma::vec mu0(2);
    mu0 << 4.5 << -9.0;
    
    // Generate some data
    unsigned int ndata = 1000;
    arma::mat data(ndata,2);
    for (int i=0; i<ndata; i++) {
        data.row(i) = arma::trans(mu0 + RandGen.normal(covar));
    }
    
    unsigned int nwalkers = 3;
    Ensemble<BiMu> MuEnsemble;
    arma::vec prior_mean(2);
    prior_mean << -1.0 << 5.0;
    arma::mat prior_var(2,2);
    prior_var << 5.0 << 0.0 << arma::endr << 0.0 << 5.0 << arma::endr;

    // Add parameters to the ensemble
    for (int j=0; j<nwalkers; j++) {
        MuEnsemble.AddObject(new BiMu(true, "mu", covar, data));
        MuEnsemble[j].SetPrior(prior_mean, prior_var);
        MuEnsemble[j].Save(MuEnsemble[j].StartingValue());
    }
    
    StretchProposal<BiMu> MuStretch(MuEnsemble, 0);
    
    // Test GrabParameter method
    int ntrials = 1000;
    int nthis_walker = 0;
    int nother_walkers = 0;
    arma::vec value0 = MuEnsemble[0].Value();
    arma::vec value1 = MuEnsemble[1].Value();
    arma::vec value2 = MuEnsemble[2].Value();
    for (int j=0; j<ntrials; j++) {
        arma::vec grabbed_value = MuStretch.GrabParameter();
        if ((grabbed_value[0] == value0[0]) && (grabbed_value[1] == value0[1])) {
            nthis_walker++;
        }
        else if ((grabbed_value[0] == value1[0]) && (grabbed_value[1] == value1[1])) {
            nother_walkers++;
        }
        else if ((grabbed_value[0] == value2[0]) && (grabbed_value[1] == value2[1])) {
            nother_walkers++;
        }
    }
    REQUIRE(nthis_walker == 0); // Make sure we never grab the same walker
    REQUIRE(nother_walkers == ntrials); // Make sure we always grab the other walkers
    
    // Test the log-density method
    double logdensity = MuStretch.LogDensity(MuEnsemble[0].Value(), MuEnsemble[1].Value());
    double logdensity0 = (MuEnsemble[0].Value().size() - 1.0) * log(2.0);
    double fracdiff = abs((logdensity - logdensity0) / logdensity0);
    REQUIRE(fracdiff < 1e-8);
    logdensity = MuStretch.LogDensity(MuEnsemble[1].Value(), MuEnsemble[0].Value());
    REQUIRE(logdensity == 0.0);
}

/*******************************************************************************
 *                                                                             *
 *                      TESTS FOR STEP CLASSES                                 *
 *                                                                             *
 *******************************************************************************/

TEST_CASE("steps/metropolis", "Test the Metropolis-Hastings step") {
    double sigma = 2.3;
    double mu0 = 6.7;
    
    // Generate some data
    unsigned int ndata = 1000;
    arma::vec data(ndata);
    data.randn();
    data *= sigma;
    data += mu0;
    
    // Create the parameter object
    Mu NormalMean(true, "mu", sigma, data);
    
    double prior_mu = -1.0;
    double prior_var = 10.0;
    NormalMean.SetPrior(prior_mu, prior_var);
    
    // Create the proposal object
    double prop_scale = sigma / sqrt(ndata);
    NormalProposal NormProp(prop_scale);
    
    // Create the step object
    MetropStep<double> MHStep(NormalMean, NormProp);
    MHStep.Start();
    
    double new_value = NormalMean.Value();
    bool accepted = MHStep.Accept(new_value, NormalMean.Value());
    REQUIRE(accepted); // Make sure we accept when proposed value is the same as current value
    double MH_ratio = MHStep.GetMetroRatio();
    REQUIRE(abs(MH_ratio - 1.0) < 1e-8); // Make sure MH ratio is unity when using the same value
    
    // Run same tests, but for a tempered parameter.
    double temperature = 25.0;
    Mu TemperedMean(true, "hot mu", sigma, data, temperature);
    TemperedMean.SetPrior(prior_mu, prior_var);
    MetropStep<double> TemperedMetro(TemperedMean, NormProp);
    TemperedMetro.Start();
    
    new_value = TemperedMean.Value();
    REQUIRE(TemperedMean.GetTemperature() == temperature);
    accepted = TemperedMetro.Accept(new_value, TemperedMean.Value());
    REQUIRE(accepted);
    MH_ratio = TemperedMetro.GetMetroRatio();
    REQUIRE(abs(MH_ratio - 1.0) < 1e-8);
    
    // Test Metro step with asymmetric proposal
    LogNormalProposal LogNormProp(prop_scale);
    MetropStep<double> Hastings(NormalMean, LogNormProp);
    Hastings.Start();
    new_value = NormalMean.Value();
    accepted = Hastings.Accept(new_value, NormalMean.Value());
    REQUIRE(accepted);
    MH_ratio = Hastings.GetMetroRatio();
    REQUIRE(abs(MH_ratio - 1.0) < 1e-8);
}

