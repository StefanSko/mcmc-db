data {
  int<lower=0> N;
  array[N] real<lower=0, upper=1> y;
  vector[N] x;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> kappa;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  kappa ~ exponential(0.1);
  {
    vector[N] mu = inv_logit(alpha + beta * x);
    vector[N] a = mu * kappa;
    vector[N] b = (1 - mu) * kappa;
    y ~ beta(a, b);
  }
}
