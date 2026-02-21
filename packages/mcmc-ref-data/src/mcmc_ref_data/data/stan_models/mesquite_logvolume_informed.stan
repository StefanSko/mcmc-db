// Informed-prior version of mesquite-logmesquite_logvolume
// Uses standardized data and weakly informative priors
data {
  int<lower=0> N;
  int<lower=0> K;
  vector[N] log_canopy_volume_std;
  vector[N] log_weight_std;
}

parameters {
  vector[K] beta;
  real<lower=0> sigma;
}

model {
  // Weakly informative priors (after standardization)
  beta ~ normal(0, 2.5);
  sigma ~ normal(0, 1);  // half-normal via constraint

  // Likelihood
  log_weight_std ~ normal(beta[1] + beta[2] * log_canopy_volume_std, sigma);
}
