data {
  int<lower=1> N;
  int<lower=1> K;
  array[N] int<lower=1,upper=K> group;
  vector[N] x;
  vector[N] y;
}
parameters {
  vector[2] mu;
  vector<lower=0>[2] sigma;
  cholesky_factor_corr[2] L_Rho;
  matrix[K, 2] z_raw;
  real<lower=0> sigma_y;
}
transformed parameters {
  matrix[K, 2] z;
  for (k in 1:K)
    z[k] = (mu + diag_pre_multiply(sigma, L_Rho) * z_raw[k]')';
}
model {
  mu ~ normal(0, 5);
  sigma ~ normal(0, 5);
  L_Rho ~ lkj_corr_cholesky(2);
  sigma_y ~ normal(0, 5);
  to_vector(z_raw) ~ normal(0, 1);
  y ~ normal(z[group, 1] + z[group, 2] .* x, sigma_y);
}
