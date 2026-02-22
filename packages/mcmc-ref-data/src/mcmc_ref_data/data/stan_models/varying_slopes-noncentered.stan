
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> group;
  array[N] real x;
  array[N] real y;
}
parameters {
  vector[2] mu;
  vector<lower=0>[2] sigma_group;
  matrix[J, 2] z_group;
  real<lower=0> sigma;
}
transformed parameters {
  matrix[J, 2] beta_group;
  for (j in 1:J) {
    beta_group[j, 1] = mu[1] + sigma_group[1] * z_group[j, 1];
    beta_group[j, 2] = mu[2] + sigma_group[2] * z_group[j, 2];
  }
}
model {
  to_vector(z_group) ~ normal(0, 1);
  sigma_group ~ normal(0, 1);
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (n in 1:N) {
    y[n] ~ normal(beta_group[group[n], 1] + beta_group[group[n], 2] * x[n], sigma);
  }
}
