// Informed-prior version of radon_all-radon_pooled
// Uses standardized data and weakly informative priors
data {
  int<lower=0> N;
  vector[N] floor_measure_std;
  vector[N] log_radon_std;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma_y;
}

model {
  // Weakly informative priors (after standardization)
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  sigma_y ~ normal(0, 1);  // half-normal via constraint

  // Likelihood
  log_radon_std ~ normal(alpha + beta * floor_measure_std, sigma_y);
}
