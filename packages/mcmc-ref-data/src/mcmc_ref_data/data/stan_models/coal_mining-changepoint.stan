data {
  int<lower=0> N;
  array[N] int<lower=0> disasters;
  vector[N] year;
  real year_min;
  real year_max;
}
parameters {
  real<lower=year_min, upper=year_max> switchpoint;
  real<lower=0> early_rate;
  real<lower=0> late_rate;
}
model {
  switchpoint ~ uniform(year_min, year_max);
  early_rate ~ exponential(1);
  late_rate ~ exponential(1);
  for (n in 1:N) {
    // Soft assignment via log_mix for smooth gradients
    real w = inv_logit(2 * (year[n] - switchpoint));
    target += log_mix(w,
                      poisson_lpmf(disasters[n] | late_rate),
                      poisson_lpmf(disasters[n] | early_rate));
  }
}
