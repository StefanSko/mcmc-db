
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
  vector[J] alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  mu_alpha ~ normal(0, 1);
  sigma_alpha ~ normal(0, 1);
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(alpha[group[n]] + beta * x[n], sigma);
  }
}
