
data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] int<lower=1, upper=D> district;
  array[N] int<lower=0, upper=1> urban;
  array[N] int<lower=0, upper=1> use;
}
parameters {
  real mu_a;
  real mu_b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  vector[D] a;
  vector[D] b;
}
model {
  a ~ normal(mu_a, sigma_a);
  b ~ normal(mu_b, sigma_b);
  mu_a ~ normal(0, 1);
  mu_b ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  for (n in 1:N) {
    use[n] ~ bernoulli_logit(a[district[n]] + b[district[n]] * urban[n]);
  }
}
