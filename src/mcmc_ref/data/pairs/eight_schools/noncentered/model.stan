data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta_raw ~ normal(0, 1);
  y ~ normal(theta, to_vector(sigma));
}
