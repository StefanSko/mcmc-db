data {
  int<lower=1> N;
}
parameters {
  real v;
  vector[N] x_raw;
}
transformed parameters {
  vector[N] x = x_raw * exp(v / 2);
}
model {
  v ~ normal(0, 3);
  x_raw ~ normal(0, 1);
}
