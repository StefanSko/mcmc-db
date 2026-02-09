data {
  int<lower=0> N;
  array[N] int<lower=0> y;
}
parameters {
  real<lower=0, upper=1> theta;
  real alpha;
}
model {
  theta ~ beta(2, 6);
  alpha ~ normal(1, 0.5);
  for (n in 1:N) {
    if (y[n] == 0) {
      target += log_sum_exp(log(theta),
                            log1m(theta) + poisson_log_lpmf(0 | alpha));
    } else {
      target += log1m(theta) + poisson_log_lpmf(y[n] | alpha);
    }
  }
}
