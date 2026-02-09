// Informed-prior version of earnings-logearn_height
// Uses standardized data and weakly informative priors
data {
  int<lower=0> N;
  vector[N] height_std;
  vector[N] log_earn_std;
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma;
}

model {
  // Weakly informative priors (after standardization)
  beta_0 ~ normal(0, 2.5);
  beta_1 ~ normal(0, 2.5);
  sigma ~ normal(0, 1);  // half-normal via constraint

  // Likelihood
  log_earn_std ~ normal(beta_0 + beta_1 * height_std, sigma);
}
