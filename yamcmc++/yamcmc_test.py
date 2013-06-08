import numpy as np
import sys
sys.path.append("src")
import yamcmcpp

class Theta(yamcmcpp.Parameter):
    def __init__(self, track, name, temperature=1.0):
        yamcmcpp.Parameter.__init__(self, track, name, temperature)
        
        self._data = None
        self._ndata = 0

    def setData(self, data):
        self._data = data
        self._ndata = len(data)

    def setPrior(self, pmean, pvar, palpha, pbeta):
        self._pmean = pmean
        self._pvar = pvar
        self._palpha = palpha
        self._pbeta = pbeta

    def StartingValue(self):
        return np.array(np.mean(self._data), np.variance(self._variance))

def test_ensemble_mcmc():

    RandGen   = yamcmcpp.RandomGenerator()

    true_mean = 3.4
    true_var  = 2.3
    nsample   = 1000
    simulated_data = []
    for i in range(nsample):
        simulated_data.append(RandGen.normal(true_mean, np.sqrt(true_var)))
    simulated_data = np.array(simulated_data)
    print simulated_data

    theta = Theta(True, "theta = (mu,sigsqr)")
    theta.setData(simulated_data);
    prior_mean     = 0.0
    prior_variance = 1.0e6
    prior_alpha    = 0.0
    prior_beta     = 0.0
    theta.setPrior(prior_mean, prior_variance, prior_alpha, prior_beta)
    print theta

    covariance = np.identity((2))
    covariance *= 1e-2
    print covariance

    thetaProp = yamcmcpp.NormalProposal(1.0)
    print thetaProp

    burnin = 100
    thin   = 1
    normalSampler = yamcmcpp.Sampler(nsample, burnin, thin)
    print normalSampler

if __name__ == "__main__":
    test_ensemble_mcmc()
