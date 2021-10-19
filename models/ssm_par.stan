
functions {
  real partial_sum(real[] y_slice,
                   int start, int end,
                   real r,
                   real q,
                   real a,
                   real b,
                   vector u,
                   vector x,
                   int N){
    int tt;
    int tt2;
    if (end+1 > N){
        tt = N;
        tt2 = end-start;
    }else {
        tt = end+1;
        tt2 = end-start+1;
    }

    vector[end-start+1] mu = a * x[start:end] + b * u[start:end];
    real target_ = normal_lpdf(y_slice | x[start:end], r);
    target_ += normal_lpdf(x[start+1:tt] | mu[1:tt2], q);
    return target_;
   }
}

data {
    int<lower=0> N;
    vector[N] u;
    real y[N];
    int grainsize;
}

parameters {
    real<lower=0.,upper=1.> a;
    real b;
    vector[N] x;
    real<lower=1e-8> q;
    real<lower=1e-8> r;
}

model {
    vector[N] mu = a * x + b * u;

    // priors
    q ~ cauchy(0., 1.);
    r ~ cauchy(0., 1.);

    // likelihoods
//    x[2:N] ~ normal(mu[1:N-1], q);
//    y ~ normal(x, r);
  target += reduce_sum(partial_sum, y, grainsize, r, q, a, b, u, x, N);

}

generated quantities {
    vector[N] yhat = x + normal_rng(0, r);
}