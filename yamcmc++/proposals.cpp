//
//  proposals.cpp
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#include "proposals.hpp"

// Method of NormalProposal class to generate a normally-distributed
// proposal, centered at starting_value.
double NormalProposal::Draw(double starting_value) {
    return RandGen.normal(starting_value, standard_deviation_);
}

// Method of StudentProposal class to generate a t-distributed
// proposal, centered at starting_value.
double StudentProposal::Draw(double starting_value) {
	return RandGen.tdist(dof_, starting_value, scale_);
}

// Method of MultiNormalProposal class to generate a multivariate
// normally-distributed proposal, centered at starting value.
arma::vec MultiNormalProposal::Draw(arma::vec starting_value) {
	return starting_value + RandGen.normal(covar_);
}

// Method of LogNormalProposal class to generate a lognormally-distributed
// proposal.
double LogNormalProposal::Draw(double starting_value) {
	double logmean = log(starting_value);
	return RandGen.lognormal(logmean, logsd_);
}
