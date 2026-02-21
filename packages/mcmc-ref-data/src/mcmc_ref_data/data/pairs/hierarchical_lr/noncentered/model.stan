
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> group;
  array[N] real x;
  array[N] real y;
}
parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[J] alpha_raw;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[J] alpha = mu_alpha + sigma_alpha * alpha_raw;
}
model {
  mu_alpha ~ normal(0, 1);
  sigma_alpha ~ normal(0, 1);
  alpha_raw ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(alpha[group[n]] + beta * x[n], sigma);
  }
}
