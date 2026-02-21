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
  matrix[K, 2] z;
  real<lower=0> sigma_y;
}
model {
  mu ~ normal(0, 5);
  sigma ~ normal(0, 5);
  L_Rho ~ lkj_corr_cholesky(2);
  sigma_y ~ normal(0, 5);
  for (k in 1:K)
    z[k] ~ multi_normal_cholesky(mu, diag_pre_multiply(sigma, L_Rho));
  y ~ normal(z[group, 1] + z[group, 2] .* x, sigma_y);
}
