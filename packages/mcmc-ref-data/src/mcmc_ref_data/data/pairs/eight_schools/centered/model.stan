
data {
  int<lower=1> N;
  array[N] real y;
  array[N] real sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[N] theta;
}
model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
