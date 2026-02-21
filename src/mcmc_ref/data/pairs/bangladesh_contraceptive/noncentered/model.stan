data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] int<lower=1,upper=D> district;
  array[N] int<lower=0,upper=1> urban;
  array[N] int<lower=0,upper=1> y;
}
parameters {
  vector[2] mu;
  vector<lower=0>[2] sigma;
  cholesky_factor_corr[2] L_Rho;
  matrix[D, 2] z_raw;
}
transformed parameters {
  matrix[D, 2] z;
  for (d in 1:D)
    z[d] = (mu + diag_pre_multiply(sigma, L_Rho) * z_raw[d]')';
}
model {
  mu ~ normal(0, 1);
  sigma ~ exponential(1);
  L_Rho ~ lkj_corr_cholesky(2);
  to_vector(z_raw) ~ normal(0, 1);
  y ~ bernoulli_logit(z[district, 1] + z[district, 2] .* to_vector(urban));
}
