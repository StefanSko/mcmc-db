data {
  int<lower=0> N;
  array[N] int<lower=0> total_tools;
  vector[N] log_pop;
  vector[N] contact_high;
}
parameters {
  real alpha;
  real beta_pop;
  real beta_contact;
  real<lower=0> phi;
}
model {
  alpha ~ normal(3, 0.5);
  beta_pop ~ normal(0.26, 0.1);
  beta_contact ~ normal(0, 0.3);
  phi ~ exponential(1);
  {
    vector[N] lambda = exp(alpha + beta_pop * log_pop
                           + beta_contact * contact_high);
    total_tools ~ neg_binomial_2(lambda, phi);
  }
}
