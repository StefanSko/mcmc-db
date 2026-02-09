data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
  vector[N] x;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  y ~ bernoulli_logit(alpha + beta * x);
}
