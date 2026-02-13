data {
  int<lower=0> N;
  int<lower=0> N_actor;
  int<lower=0> N_block;
  array[N] int<lower=1, upper=N_actor> actor;
  array[N] int<lower=1, upper=N_block> block;
  array[N] int<lower=0, upper=1> pulled_left;
  vector[N] treatment;
}
parameters {
  real alpha_bar;
  real beta_bar;
  vector[N_actor] alpha;
  vector[N_actor] beta;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  vector[N_block] gamma;
  real<lower=0> sigma_gamma;
}
model {
  alpha_bar ~ normal(0, 1.5);
  beta_bar ~ normal(0, 0.5);
  sigma_alpha ~ exponential(1);
  sigma_beta ~ exponential(1);
  sigma_gamma ~ exponential(1);
  alpha ~ normal(alpha_bar, sigma_alpha);
  beta ~ normal(beta_bar, sigma_beta);
  gamma ~ normal(0, sigma_gamma);
  pulled_left ~ bernoulli_logit(alpha[actor] + beta[actor] .* treatment
                                + gamma[block]);
}
