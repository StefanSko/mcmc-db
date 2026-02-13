data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
  real<lower=1> nu;
}
model {
  alpha ~ normal(0, 100);
  beta ~ normal(0, 100);
  sigma ~ exponential(0.01);
  nu ~ gamma(2, 0.1);
  earn ~ student_t(nu, alpha + beta * height, sigma);
}
