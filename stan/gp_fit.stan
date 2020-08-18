data {
	int<lower=1> N;
	matrix[11,N] x;
	vector[N] y;
}

transformed data {
	vector[N] mu;
	for (i in 1:N) mu[i] = 0;
}

parameters { 
	real log_eta_sq;
	vector[11] log_rho_sq;
}



model {
	vector[11] diff;
	matrix[N,N] Sigma;
	matrix[11,11] hypers_diag;
  real p;
  
	hypers_diag = diag_matrix(exp(log_rho_sq));
	// off-diagonal elements
	for (i in 1:(N-1)) {
		for (j in (i+1):N) {
		          diff = x[,i]-x[,j];				
				      Sigma[i,j] = exp(log_eta_sq) * exp(-1*(quad_form(hypers_diag,diff)));
							Sigma[j,i] = Sigma[i,j];
		}
	}


	// diagonal elements
	for (k in 1:N){ 
		p = (1000.5*exp(y[k]) - 0.5)/(1000*(1+exp(y[k])));
		Sigma[k,k] = exp(log_eta_sq) + 1/(1000*p*(1-p));
	}

	log_eta_sq ~ normal(0,10);
	log_rho_sq ~ normal(0,10);
	y ~ multi_normal(mu,Sigma);
}

