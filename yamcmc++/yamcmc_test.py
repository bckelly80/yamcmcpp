import numpy as np
import sys
sys.path.append("src")
import _yamcmcpp

def test_ensemble_mcmc():

    RandGen   = _yamcmcpp.RandomGenerator()

    true_mean = 3.4
    true_var  = 2.3
    nsample   = 1000
    simulated_data = []
    for i in range(nsample):
        simulated_data.append(RandGen.normal(true_mean, np.sqrt(true_var)))
    simulated_data = np.array(simulated_data)
    print simulated_data

    theta = _yamcmcpp.Theta(True, "theta = (mu,sigsqr)")
    theta.SetData(simulated_data, nsample);
    prior_mean     = 0.0
    prior_variance = 1.0e6
    prior_alpha    = 0.0
    prior_beta     = 0.0
    theta.SetPrior(prior_mean, prior_variance, prior_alpha, prior_beta)

if __name__ == "__main__":
    test_ensemble_mcmc()
