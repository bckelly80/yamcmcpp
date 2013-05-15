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

