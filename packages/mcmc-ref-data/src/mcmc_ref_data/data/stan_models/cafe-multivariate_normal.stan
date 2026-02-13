data {
  int<lower=0> N;
  int<lower=0> N_cafe;
  array[N] int<lower=1, upper=N_cafe> cafe;
  vector[N] afternoon;
  vector[N] wait;
}
parameters {
  real alpha_bar;
  real beta_bar;
  vector<lower=0>[2] sigma_cafe;
  cholesky_factor_corr[2] L_Rho;
  matrix[N_cafe, 2] z;
  real<lower=0> sigma;
}
transformed parameters {
  matrix[N_cafe, 2] v;
  v = z * diag_pre_multiply(sigma_cafe, L_Rho)';
}
model {
  alpha_bar ~ normal(5, 2);
  beta_bar ~ normal(-1, 0.5);
  sigma_cafe ~ exponential(1);
  L_Rho ~ lkj_corr_cholesky(2);
  to_vector(z) ~ normal(0, 1);
  sigma ~ exponential(1);
  {
    vector[N] mu;
    for (n in 1:N) {
      mu[n] = (alpha_bar + v[cafe[n], 1])
              + (beta_bar + v[cafe[n], 2]) * afternoon[n];
    }
    wait ~ normal(mu, sigma);
  }
}
generated quantities {
  matrix[2, 2] Rho;
  Rho = multiply_lower_tri_self_transpose(L_Rho);
}
