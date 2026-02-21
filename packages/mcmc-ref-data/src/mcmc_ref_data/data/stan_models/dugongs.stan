
data {
  int<lower=1> N;
  array[N] real x;
  array[N] real y;
}
parameters {
  real<lower=0> U3;
  real alpha;
  real beta;
  real<lower=0> lambda;
  real<lower=0> sigma;
  real tau;
}
transformed parameters {
  array[N] real mu;
  for (n in 1:N) {
    mu[n] = U3 - alpha * exp(-lambda * x[n]) + beta;
  }
}
model {
  U3 ~ lognormal(2, 0.3);
  alpha ~ normal(3.0, 1.0);
  beta ~ normal(0.0, 1.0);
  lambda ~ lognormal(-2.0, 0.4);
  tau ~ normal(alpha, 0.5);
  sigma ~ lognormal(-2.0, 0.3);
  y ~ normal(mu, sigma + 0.05 * abs(tau));
}
