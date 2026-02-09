// Informed-prior version of sblrc-blr
// Uses standardized data and weakly informative priors
data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N, D] X_std;
  vector[N] y_std;
}

parameters {
  vector[D] beta;
  real<lower=0> sigma;
}

model {
  // Weakly informative priors (after standardization)
  beta ~ normal(0, 2.5);
  sigma ~ normal(0, 1);  // half-normal via constraint

  // Likelihood
  y_std ~ normal(X_std * beta, sigma);
}
