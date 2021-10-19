// make models/ssm

// to build using new compiled stanc3
// ./_build/default/src/stanc/stanc.exe ssm.stan

data {
    int<lower=0> N;
    vector[N] u;
    vector[N] y;
}

parameters {
    real<lower=0.,upper=1.> a;
    real b;
    vector[N] x;
    real<lower=1e-8> q;
    real<lower=1e-8> r;
}

transformed parameters {
    vector[N] mu = a * x + b * u;
}

model {
    // priors
    q ~ cauchy(0., 1.);
    r ~ cauchy(0., 1.);

    // likelihoods
//    x[2:N] ~ normal(mu[1:N-1], q);        // standard distribution
//    x[2:N] ~ new_normal(mu[1:N-1], q);      // use to test adding new distribution
    x[2:N] ~ new_normal_blas(mu[1:N-1], q);      // use to test blas
//    target += new_normal_lpdf(x[2:N] | mu[1:N-1], q);
    y ~ normal(x, r);

}

generated quantities {
    vector[N] yhat = x + normal_rng(0, r);
}
