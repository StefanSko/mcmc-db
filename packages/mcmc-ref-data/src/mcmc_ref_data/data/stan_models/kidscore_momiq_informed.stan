// Informed-prior version of kidiq-kidscore_momiq
// Uses standardized data and weakly informative priors
data {
  int<lower=0> N;
  vector[N] mom_iq_std;
  vector[N] kid_score_std;
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
  kid_score_std ~ normal(beta_0 + beta_1 * mom_iq_std, sigma);
}
