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
  vector[K] alpha_raw;
  real beta;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[K] alpha = mu_alpha + sigma_alpha * alpha_raw;
}
model {
  mu_alpha ~ normal(0, 10);
  sigma_alpha ~ normal(0, 5);
  alpha_raw ~ normal(0, 1);
  beta ~ normal(0, 5);
  sigma_y ~ normal(0, 5);
  y ~ normal(alpha[group] + beta * x, sigma_y);
}
