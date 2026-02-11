data {
  int<lower=1> N;
  int<lower=1> K;
  array[N] int<lower=1,upper=K> group;
  vector[N] x;
  vector[N] y;
}
parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[K] alpha;
  real beta;
  real<lower=0> sigma_y;
}
model {
  mu_alpha ~ normal(0, 10);
  sigma_alpha ~ normal(0, 5);
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(0, 5);
  sigma_y ~ normal(0, 5);
  y ~ normal(alpha[group] + beta * x, sigma_y);
}
