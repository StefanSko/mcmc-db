data {
  int<lower=0> N;
  int<lower=0> N_tank;
  array[N] int<lower=1, upper=N_tank> tank;
  array[N] int<lower=0> surv;
  array[N] int<lower=0> density;
}
parameters {
  real alpha_bar;
  real<lower=0> sigma;
  vector[N_tank] alpha;
}
model {
  alpha_bar ~ normal(0, 1.5);
  sigma ~ exponential(1);
  alpha ~ normal(alpha_bar, sigma);
  surv ~ binomial_logit(density, alpha[tank]);
}
