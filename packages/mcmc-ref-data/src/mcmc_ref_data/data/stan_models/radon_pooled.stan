
data {
  int<lower=1> N;
  int<lower=1> N_county;
  array[N] int<lower=1, upper=N_county> county;
  array[N] int<lower=0, upper=1> floor_measure;
  array[N] real log_radon;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma;
}
model {
  beta_0 ~ normal(0, 2);
  beta_1 ~ normal(0, 1);
  sigma ~ lognormal(-1, 0.5);
  for (n in 1:N) {
    log_radon[n] ~ normal(beta_0 + beta_1 * floor_measure[n], sigma);
  }
}
