data {
  int<lower=1> N;
}
parameters {
  real v;
  vector[N] x;
}
model {
  v ~ normal(0, 3);
  x ~ normal(0, exp(v / 2));
}
