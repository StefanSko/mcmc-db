
data {
  int<lower=1> N;
  array[N] real y;
  array[N] real sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[N] theta_raw;
}
transformed parameters {
  vector[N] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  theta_raw ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
