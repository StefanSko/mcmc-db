data {
  int<lower=0> N_obs;
  int<lower=0> N_cens;
  vector<lower=0>[N_obs] t_obs;
  vector<lower=0>[N_cens] t_cens;
  vector[N_obs] x_obs;
  vector[N_cens] x_cens;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> k;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  k ~ lognormal(0, 1);
  // Observed events
  {
    vector[N_obs] lambda_obs = exp(alpha + beta * x_obs);
    t_obs ~ weibull(k, 1.0 ./ lambda_obs);
  }
  // Right-censored observations
  {
    vector[N_cens] lambda_cens = exp(alpha + beta * x_cens);
    for (n in 1:N_cens) {
      target += weibull_lccdf(t_cens[n] | k, 1.0 / lambda_cens[n]);
    }
  }
}
