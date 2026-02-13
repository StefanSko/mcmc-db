data {
  int<lower=2> K;
  int<lower=0> N;
  array[N] int<lower=1, upper=K> y;
  vector[N] action;
  vector[N] intention;
  vector[N] contact;
}
parameters {
  ordered[K - 1] cutpoints;
  real beta_action;
  real beta_intention;
  real beta_contact;
}
model {
  cutpoints ~ normal(0, 1.5);
  beta_action ~ normal(0, 1);
  beta_intention ~ normal(0, 1);
  beta_contact ~ normal(0, 1);
  {
    vector[N] phi = beta_action * action
                    + beta_intention * intention
                    + beta_contact * contact;
    for (n in 1:N) {
      y[n] ~ ordered_logistic(phi[n], cutpoints);
    }
  }
}
