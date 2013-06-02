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
#include <boost/lexical_cast.hpp>

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

TEST_CASE("steps/rank1_cholesky_update", "Test the rank-1 Cholesky update and downdate.") {
    arma::mat corr(3,3);
    corr << 1.0 << 0.3 << -0.5 << arma::endr
    << 0.3 << 1.0 << 0.54 << arma::endr
    << -0.5 << 0.54 << 1.0;
    
    arma::mat sigma(3,3);
    sigma << 2.3 << 0.0 << 0.0 << arma::endr
    << 0.0 << 0.45 << 0.0 << arma::endr
    << 0.0 << 0.0 << 13.4;
    
    arma::mat covar = sigma * corr * sigma;
    
    arma::mat chol_factor = arma::chol(covar);
    arma::vec v(3);
    v.randn();
    v = v / arma::norm(v, 2.0);
    v = sqrt(0.5) * chol_factor.t() * v;
    
    arma::mat covar_update = covar + v * v.t();
    arma::mat covar_downdate = covar - v * v.t();
    
    // First get cholesky factors from updated and downdated matrices using the slow way
    arma::mat Lup0 = arma::chol(covar_update);
    arma::mat Ldown0 = arma::chol(covar_downdate);
    
    // Now get rank-1 updated and downdated factors using the fast method.
    arma::mat Lup = chol_factor;
    arma::vec v0 = v;
    CholUpdateR1(Lup, v, false);
    arma::mat Ldown = chol_factor;
    v0 = v;
    CholUpdateR1(Ldown, v, true);
    
    // Make sure the cholesky factors from the two methods agree.
    double max_frac_diff = 100.0;
    for (int i=0; i<Lup.n_rows; i++) {
        // Cholesky factor is upper triangular, so only loop over upper triangle
        for (int j=i; j<Lup.n_cols; j++) {
            double frac_diff = abs((Lup(i,j) - Lup0(i,j)) / Lup0(i,j));
            max_frac_diff = std::min(frac_diff, max_frac_diff);
        }
    }
    REQUIRE(max_frac_diff < 1e-8);
    
    max_frac_diff = 100.0;
    for (int i=0; i<Ldown.n_rows; i++) {
        // Cholesky factor is upper triangular, so only loop over upper triangle
        for (int j=i; j<Ldown.n_cols; j++) {
            double frac_diff = abs((Ldown(i,j) - Ldown0(i,j)) / Ldown0(i,j));
            max_frac_diff = std::min(frac_diff, max_frac_diff);
        }
    }
    REQUIRE(max_frac_diff < 1e-8);
}

TEST_CASE("steps/adaptive_mha", "Test the Robust Adaptive Metropolis step class") {
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

    BiMu NormMean(true, "mu", covar, data);
    
    arma::vec prior_mean(2);
    prior_mean << -1.0 << 5.0;
    arma::mat prior_var(2,2);
    prior_var << 5.0 << 0.0 << arma::endr << 0.0 << 5.0 << arma::endr;
    
    NormMean.SetPrior(prior_mean, prior_var);

    NormalProposal UnitProp(1.0);
    
    double target_rate = 0.4;
    arma::mat prop_covar(2,2);
    prop_covar.eye();
    int niter = 100000;
    int maxiter = niter + 1;
    AdaptiveMetro RAM(NormMean, UnitProp, prop_covar, target_rate, maxiter);
    RAM.Start();
    
    // Make sure we always accept proposed values that are the same as the current values
    arma::vec new_value = NormMean.Value();
    bool accepted = RAM.Accept(new_value, NormMean.Value());
    REQUIRE(accepted);
    double MH_ratio = RAM.GetMetroRatio();
    REQUIRE(abs(MH_ratio - 1.0) < 1e-8); // Make sure MH ratio is unity when using the same value
    
    // Make sure we achieve the requested acceptance rate
    for (int i=0; i<niter; i++) {
        RAM.DoStep();
    }
    double arate = RAM.GetAcceptRate();
    double frac_diff = abs(arate - target_rate) / target_rate;
    REQUIRE(frac_diff < 0.02); // Make sure acceptance rate is within 2% of target
    
    // Test if covariance matrix of proposals has converged to stable point
    prop_covar = RAM.GetCovariance();
    arma::vec prop_eval(2);
    arma::mat prop_evect(2,2);
    arma::eig_sym(prop_eval, prop_evect, prop_covar);
    arma::mat prop_covroot = prop_evect.t() * arma::diagmat(arma::sqrt(prop_eval)) * prop_evect;
    
    arma::mat post_covar = arma::inv(prior_var.i() + ndata * covar.i());
    arma::vec post_eval(2);
    arma::mat post_evect(2,2);
    arma::eig_sym(post_eval, post_evect, post_covar);
    arma::mat postinv_covroot = post_evect.t() * arma::diagmat(1.0 / arma::sqrt(post_eval)) * post_evect;
    
    // Compute the sub-optimality factor. This should be unity if two matrices are proportional.
    arma::mat ratio_matrix = prop_covroot * postinv_covroot;
    arma::vec evals(2);
    arma::mat evect(2,2);
    arma::eig_sym(evals, evect, ratio_matrix);
    double inv_eval_sum = arma::sum(1.0 / evals);
    double inv_eval_sqr_sum = arma::sum(1.0 / (evals % evals));
    double subopt_fact = evals.n_elem * inv_eval_sqr_sum / (inv_eval_sum * inv_eval_sum);
    REQUIRE(subopt_fact < 1.01); // make sure sub-optimality factor is within 1% of theoretical value
}

TEST_CASE("steps/exchange", "Test the exchange step from parallel tempering.") {
    double sigma = 2.3;
    double mu0 = 6.7;    
    double prior_mu = -1.0;
    double prior_var = 10.0;
    
    // Generate some data
    unsigned int ndata = 1000;
    arma::vec data(ndata);
    data.randn();
    data *= sigma;
    data += mu0;

    // Create the parameter ensemble
    Ensemble<Mu> MuEnsemble;
    double temp_ladder[3] = {1.0, 2.0, 5.0};
    for (int j=0; j<3; j++) {
        MuEnsemble.AddObject(new Mu(true, "mu", sigma, data, temp_ladder[j]));
        MuEnsemble[j].SetPrior(prior_mu, prior_var);
        MuEnsemble[j].Save(MuEnsemble[j].StartingValue());
    }
    
    // Create the exchange steps
    ExchangeStep<double, Mu> Exchange21(MuEnsemble[2], 2, MuEnsemble);
    ExchangeStep<double, Mu> Exchange10(MuEnsemble[1], 1, MuEnsemble);
    
    // First make sure that we always accept cases when the parameter value is unchanged
    MuEnsemble[2].Save(MuEnsemble[1].Value());
    Exchange21.DoStep();
    double MH_ratio = Exchange21.GetMetroRatio();
    REQUIRE(abs(MH_ratio - 1.0) < 1e-8);
    
    // Now test to make sure that the parameter values and logdensities are exchanged correctly
    double bad_value = -10.0;
    MuEnsemble[0].Save(bad_value); // Choose really bad value to make sure we accept the exchange
    double old_value0 = bad_value;
    double old_logpost0 = MuEnsemble[0].GetLogDensity();
    double old_value1 = MuEnsemble[1].Value();
    double old_logpost1 = MuEnsemble[1].GetLogDensity();
    Exchange10.DoStep();
    double new_value0 = MuEnsemble[0].Value();
    double new_logpost0 = MuEnsemble[0].GetLogDensity();
    double new_value1 = MuEnsemble[1].Value();
    double new_logpost1 = MuEnsemble[1].GetLogDensity();
    REQUIRE(old_value1 == new_value0); // Were the parameter values exchanged?
    REQUIRE(old_value0 == new_value1);
    REQUIRE(new_logpost0 == old_logpost1); // Were the log-densities exchanged?
    REQUIRE(new_logpost1 == old_logpost0);
}

/*******************************************************************************
 *                                                                             *
 *                      TESTS FOR MCMC SAMPLER CLASSES                         *
 *                                                                             *
 *******************************************************************************/


TEST_CASE("samplers/metropolis_uni_sampler", "Test the MCMC sampler for a univariate normal model using a Metropolis algorithm.")
{
    double sigma = 2.3;
    double mu0 = 6.7;
    double prior_mu = -1.0;
    double prior_var = 10.0;
    
    // Generate some data
    unsigned int ndata = 1000;
    arma::vec data(ndata);
    data.randn();
    data *= sigma;
    data += mu0;
    
    // setup MCMC options
    int sample_size = 100000;
    int nthin = 1;
    int burnin = 10000;
    
    // Instantiate MCMC objects needed for MCMC Sampler
    Mu NormMean(true, "mu", sigma, data);
    NormMean.SetPrior(prior_mu, prior_var);
    StudentProposal MuProp(8.0, sigma / sqrt(ndata));
	Sampler normal_model(sample_size, burnin, nthin);
    int report_iter = sample_size;
	normal_model.AddStep(new MetropStep<double>(NormMean, MuProp, report_iter));
    
    // Make sure the parameter named "mu" is tracked
    std::set<std::string> tracked_names = normal_model.GetTrackedNames();
    std::set<std::string>::iterator mu_it;
    mu_it = tracked_names.find(NormMean.Label());
    REQUIRE(mu_it != tracked_names.end());
    
    // Make sure the map of pointers to the parameter objects points to the correct object
    std::map<std::string, BaseParameter*> p_tracked_params = normal_model.GetTrackedParams();
    REQUIRE(p_tracked_params.size() == 1);
    std::map<std::string, BaseParameter*>::iterator mu_it2 = p_tracked_params.find(NormMean.Label());
    REQUIRE(mu_it2 != p_tracked_params.end());
    REQUIRE(p_tracked_params[NormMean.Label()]->Label() == "mu");
    REQUIRE(p_tracked_params[NormMean.Label()]->GetTemperature() == 1.0);
    
    // Run the sampler
    normal_model.Run();
    
    // Grab the sampled parameter values
    std::vector<double> samples0 = NormMean.GetSamples();
    arma::vec samples(samples0); // convert to armadillo library vector
    double post_mean = arma::mean(samples);
    double post_var = arma::var(samples);
    double zscore = std::abs(post_mean - mu0) / sqrt(post_var);
    REQUIRE(zscore < 3.0);
}

TEST_CASE("samplers/RAM_bivariate_sampler", "Test the MCMC sampler for a bivariate normal model using the Robust Adaptive Metropolis algorithm.")
{
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
    
    BiMu NormMean(true, "mu", covar, data);
    
    arma::vec prior_mean(2);
    prior_mean << 0.0 << 0.0;
    arma::mat prior_var(2,2);
    prior_var << 100.0 << 0.0 << arma::endr << 0.0 << 100.0 << arma::endr;
    
    NormMean.SetPrior(prior_mean, prior_var);
    
    NormalProposal UnitProp(1.0);
    
    // setup MCMC parameters
    int sample_size = 100000;
    int nthin = 1;
    int burnin = 10000;

    // Instantiate MCMC sampler object
    Sampler normal_model(sample_size, burnin, nthin);

    // RAM step parameters
    double target_rate = 0.4;
    arma::mat prop_covar(2,2);
    prop_covar.eye();
    int maxiter = burnin;
    
    // Add RAM step
    normal_model.AddStep(new AdaptiveMetro(NormMean, UnitProp, prop_covar, target_rate, maxiter));
    
    // Make sure the parameter named "mu" is tracked
    std::set<std::string> tracked_names = normal_model.GetTrackedNames();
    std::set<std::string>::iterator mu_it;
    mu_it = tracked_names.find(NormMean.Label());
    REQUIRE(mu_it != tracked_names.end());
    
    // Make sure the map of pointers to the parameter objects points to the correct object
    std::map<std::string, BaseParameter*> p_tracked_params = normal_model.GetTrackedParams();
    REQUIRE(p_tracked_params.size() == 1);
    std::map<std::string, BaseParameter*>::iterator mu_it2 = p_tracked_params.find(NormMean.Label());
    REQUIRE(mu_it2 != p_tracked_params.end());
    REQUIRE(p_tracked_params[NormMean.Label()]->Label() == "mu");
    REQUIRE(p_tracked_params[NormMean.Label()]->GetTemperature() == 1.0);
    
    // Run the sampler
    normal_model.Run();
    
    // Grab the sampled parameter values
    std::vector<arma::vec> samples0 = NormMean.GetSamples();
    
    // convert to armadillo library vector
    arma::vec mu1_samples(sample_size);
    arma::vec mu2_samples(sample_size);
    for (int i=0; i<sample_size; i++) {
        mu1_samples[i] = samples0[i](0);
        mu2_samples[i] = samples0[i](1);
    }
    
    arma::vec post_mean(2);
    post_mean(0) = arma::mean(mu1_samples);
    post_mean(1) = arma::mean(mu2_samples);
    
    arma::mat post_covar(2,2);
    post_covar(0,0) = arma::var(mu1_samples);
    post_covar(1,1) = arma::var(mu2_samples);
    post_covar(0,1) = arma::as_scalar(arma::cov(mu1_samples, mu2_samples));
    post_covar(1,0) = post_covar(0,1);
    
    // make sure true values are contained within 99% credibility region
    arma::vec mu_centered = post_mean - mu0;
    double zsqr = arma::as_scalar(mu_centered.t() * arma::inv(post_covar) * mu_centered);
    REQUIRE(zsqr < 9.21);
}

TEST_CASE("samplers/RAM_exchange_bivariate_sampler", "Test the MCMC sampler for a bivariate normal model using the Robust Adaptive Metropolis algorithm and Parallel tempering.")
{
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
    
    // prior parameters
    arma::vec prior_mean(2);
    prior_mean << 0.0 << 0.0;
    arma::mat prior_var(2,2);
    prior_var << 100.0 << 0.0 << arma::endr << 0.0 << 100.0 << arma::endr;
    
    // ensemble parameters
    int nchains = 10;
    Ensemble<BiMu> BiMuEnsemble;
    double max_temperature = 100.0;
    arma::vec temp_ladder = arma::linspace<arma::vec>(0.0, log(max_temperature), nchains);
    temp_ladder = arma::exp(temp_ladder);
    
    // build the ensemble of parameter objects
    for (int j=0; j<nchains; j++) {
        std::string parname = "mu_" + boost::lexical_cast<std::string>(j);
        BiMuEnsemble.AddObject(new BiMu(false, parname, covar, data, temp_ladder(j)));
        BiMuEnsemble[j].SetPrior(prior_mean, prior_var);
    }
    
    StudentProposal UnitProp(8.0, 1.0);
    
    // setup MCMC parameters
    int sample_size = 100000;
    int nthin = 1;
    int burnin = 10000;
    int report_iter = burnin + nthin * sample_size;

    // Instantiate MCMC sampler object
    Sampler normal_model(sample_size, burnin, nthin);
    
    // RAM step parameters
    double target_rate = 0.4;
    arma::mat prop_covar(2,2);
    prop_covar.eye();
    int maxiter = burnin;
    
    // Add the steps to the sampler, starting with the hottest chain first
    for (int j=nchains-1; j>0; j--) {
        // First add Robust Adaptive Metropolis Step
        normal_model.AddStep( new AdaptiveMetro(BiMuEnsemble[j], UnitProp, prop_covar,
                                                target_rate, maxiter) );
        // Now add Exchange steps
        normal_model.AddStep( new ExchangeStep<arma::vec, BiMu>(BiMuEnsemble[j], j, BiMuEnsemble, report_iter) );
    }
    
    // Make sure we set this parameter in the coolest chain (the one corresponding to the posterior) to be tracked
    BiMuEnsemble[0].SetTracking(true);
    // Add in coolest chain. This is the chain that is actually moving in the posterior.
    normal_model.AddStep( new AdaptiveMetro(BiMuEnsemble[0], UnitProp, prop_covar, target_rate, maxiter) );
    
    // Make sure the parameter named "mu_0" is tracked)
    std::set<std::string> tracked_names = normal_model.GetTrackedNames();
    REQUIRE(tracked_names.size() == 1);
    std::set<std::string>::iterator mu_it;
    mu_it = tracked_names.find(BiMuEnsemble[0].Label());
    REQUIRE(mu_it != tracked_names.end());
    REQUIRE(*mu_it == "mu_0");
    
    // Make sure the map of pointers to the parameter objects points to the correct object
    std::map<std::string, BaseParameter*> p_tracked_params = normal_model.GetTrackedParams();
    REQUIRE(p_tracked_params.size() == 1);
    std::map<std::string, BaseParameter*>::iterator mu_it2 = p_tracked_params.find(BiMuEnsemble[0].Label());
    REQUIRE(mu_it2 != p_tracked_params.end());
    REQUIRE(p_tracked_params[BiMuEnsemble[0].Label()]->Label() == "mu_0");
    REQUIRE(p_tracked_params[BiMuEnsemble[0].Label()]->GetTemperature() == 1.0);
    
    // Run the sampler
    normal_model.Run();
    
    // Grab the sampled parameter values
    std::vector<arma::vec> samples0 = BiMuEnsemble[0].GetSamples();
    
    // convert to armadillo library vector
    arma::vec mu1_samples(sample_size);
    arma::vec mu2_samples(sample_size);
    for (int i=0; i<sample_size; i++) {
        mu1_samples[i] = samples0[i](0);
        mu2_samples[i] = samples0[i](1);
    }
    
    arma::vec post_mean(2);
    post_mean(0) = arma::mean(mu1_samples);
    post_mean(1) = arma::mean(mu2_samples);
    
    arma::mat post_covar(2,2);
    post_covar(0,0) = arma::var(mu1_samples);
    post_covar(1,1) = arma::var(mu2_samples);
    post_covar(0,1) = arma::as_scalar(arma::cov(mu1_samples, mu2_samples));
    post_covar(1,0) = post_covar(0,1);
    
    // make sure true values are contained within 99% credibility region
    arma::vec mu_centered = post_mean - mu0;
    double zsqr = arma::as_scalar(mu_centered.t() * arma::inv(post_covar) * mu_centered);
    REQUIRE(zsqr < 9.21);
}

// Define class for univariate normal mean parameter
class NormalMean : public Parameter<double> {
public:
	// Constructor
	NormalMean(bool track, std::string name, arma::vec& data, double temperature=1.0) :
    Parameter<double>(track, name, temperature), data_(data)
    {
        data_mean_ = arma::mean(data_);
    }
    
    // Generate a random draw of the mean from its posterior
	double RandomPosterior() {
        double condvar = 1.0 / (1.0 / prior_var_ + data_.size() / SigSqr_->Value());
        double condmean = condvar * (prior_mean_ / prior_var_ + data_.size() * data_mean_ / SigSqr_->Value());
        double cond_sigma = sqrt(condvar);
        return RandGen.normal(condmean, cond_sigma);
    }
    // Generate the initial value for mu by drawing from its posterior
	double StartingValue() {
        double initial_mu = data_mean_ + arma::stddev(data_) / sqrt(data_.size()) * RandGen.normal();
        return initial_mu;
    }
 	// Prior is normal
	void SetPrior(double prior_mean, double prior_var) {
        prior_mean_ = prior_mean;
        prior_var_ = prior_var;
    }
    // Set the pointer to the normal variance parameter object
    void SetNormVar(Parameter<double>* SigSqr) {
        SigSqr_ = SigSqr;
    }
    
private:
    Parameter<double>* SigSqr_;
    double data_mean_;
    double sigsqr_;
	double prior_mean_;
	double prior_var_;
    arma::vec& data_;
};

// Define class for univariate normal mean parameter
class NormalVar : public Parameter<double> {
public:
	// Constructor
	NormalVar(bool track, std::string name, arma::vec& data, double temperature=1.0) :
    Parameter<double>(track, name, temperature), data_(data) {}
    
    // Generate a random draw of the mean from its posterior
	double RandomPosterior() {
        double dof = prior_dof_ + data_.n_elem;
        arma::vec data_cent = data_ - Mu_->Value();
        double data_ssqr = arma::mean(data_cent % data_cent);
        double ssqr = (prior_dof_ * prior_ssqr_ + data_.n_elem * data_ssqr) / dof;
        return RandGen.scaled_inverse_chisqr(dof, ssqr);
    }
    
    // Generate the initial value for mu by drawing from its posterior
	double StartingValue() {
        double initial_var = RandGen.scaled_inverse_chisqr(data_.n_elem, arma::var(data_));
        return initial_var;
    }

	// Prior is normal
	void SetPrior(double prior_dof, double prior_ssqr) {
        prior_dof_ = prior_dof;
        prior_ssqr_ = prior_ssqr;
    }
    // Set the pointer to the normal variance parameter object
    void SetNormMean(Parameter<double>* NormMean) {
        Mu_ = NormMean;
    }
    
private:
    Parameter<double>* Mu_;
	double prior_dof_;
	double prior_ssqr_;
    arma::vec& data_;
};


TEST_CASE("samplers/gibbs_sampler", "Test the Gibbs sampler for a normal model.") {
    double sigma0 = 2.3;
    double mu0 = 6.7;
    double prior_mu = -1.0;
    double prior_var = 10.0;
    double prior_dof = 10.0;
    double prior_ssqr = 10.0;
    
    // Generate some data
    unsigned int ndata = 1000;
    arma::vec data(ndata);
    data.randn();
    data *= sigma0;
    data += mu0;
    
    // setup MCMC options
    int sample_size = 100000;
    int burnin = 10000;
    Sampler normal_model(sample_size, burnin);

    // Instantiate MCMC objects needed for MCMC Sampler
    NormalMean Mu(true, "mu", data);
    NormalVar SigSqr(true, "sigsqr", data);
    Mu.SetPrior(prior_mu, prior_var);
    Mu.SetNormVar(&SigSqr);
    SigSqr.SetPrior(prior_dof, prior_ssqr);
    SigSqr.SetNormMean(&Mu);
    
    // Add the Gibbs steps to the MCMC sampler
    normal_model.AddStep(new GibbsStep<double>(Mu));
    normal_model.AddStep(new GibbsStep<double>(SigSqr));
    
    // Make sure the parameters named "mu" and "sigsqr" are tracked
    std::set<std::string> tracked_names = normal_model.GetTrackedNames();
    REQUIRE(tracked_names.size() == 2);
    std::set<std::string>::iterator parameter_it;
    parameter_it = tracked_names.find(Mu.Label());
    REQUIRE(parameter_it != tracked_names.end());
    REQUIRE(*parameter_it == Mu.Label());
    parameter_it = tracked_names.find(SigSqr.Label());
    REQUIRE(parameter_it != tracked_names.end());
    REQUIRE(*parameter_it == SigSqr.Label());
    
    // Make sure the map of pointers to the parameter objects points to the correct object
    std::map<std::string, BaseParameter*> p_tracked_params = normal_model.GetTrackedParams();
    REQUIRE(p_tracked_params.size() == 2);
    std::map<std::string, BaseParameter*>::iterator parameter_it2 = p_tracked_params.find(Mu.Label());
    REQUIRE(parameter_it2 != p_tracked_params.end());
    REQUIRE(p_tracked_params[Mu.Label()]->Label() == "mu");
    REQUIRE(p_tracked_params[Mu.Label()]->GetTemperature() == 1.0);
    parameter_it2 = p_tracked_params.find(SigSqr.Label());
    REQUIRE(parameter_it2 != p_tracked_params.end());
    REQUIRE(p_tracked_params[SigSqr.Label()]->Label() == "sigsqr");
    REQUIRE(p_tracked_params[SigSqr.Label()]->GetTemperature() == 1.0);
    
    // Run the sampler
    normal_model.Run();
    
    // Grab the sampled parameter values
    std::vector<double> mu_samples0 = Mu.GetSamples();
    std::vector<double> var_samples0 = SigSqr.GetSamples();
    arma::vec mu_samples(mu_samples0); // convert to armadillo library vector
    arma::vec var_samples(var_samples0);
    
    // Make sure they are within 3sigma of their true values
    double mu_post_mean = arma::mean(mu_samples);
    double mu_post_var = arma::var(mu_samples);
    double zscore = std::abs(mu_post_mean - mu0) / sqrt(mu_post_var);
    REQUIRE(zscore < 3.0);
    
    double var_post_mean = arma::mean(var_samples);
    double var_post_var = arma::var(var_samples);
    zscore = std::abs(var_post_mean - sigma0 * sigma0) / sqrt(var_post_var);
    REQUIRE(zscore < 3.0);
}
