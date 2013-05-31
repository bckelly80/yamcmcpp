import numpy as np
import sys
sys.path.append("src")
import _yamcmcpp

def test_ensemble_mcmc():

    RandGen   = RandomGenerator()

    true_mean = 3.4
    true_var  = 2.3
    nsample   = 1000
    simulated_data = []
    for i in range(sample):
        simulated_data[i] = RandGen.normal(true_mean, np.sqrt(true_var))
    simulated_data = np.array(simulated_data)

if __name__ == "__main__":
    test_ensemble_mcmc()
