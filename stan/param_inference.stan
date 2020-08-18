data {
	int<lower=1> Nd;
	vector[11] x[Nd];
	vector[Nd] y[24];
	vector[12] hypers[24];
	vector[12] coeffs[24];
	int<lower=1> nobs;
	vector[nobs] yobs;

	vector[13] prior_mean; 
	matrix[13,13] prior_var;
}


transformed data{
		
	cov_matrix[Nd] Sigma[24];
	vector[11] diff;
	matrix[11,11] hypers_diag;
	matrix[Nd, Nd] L_K[24];
	real p1;
	for(a in 1:24){
		hypers_diag = diag_matrix(exp(hypers[a][2:12]));	
		for (i in 1:(Nd-1)){
			for (j in (i+1):Nd){
				diff = x[i]-x[j];				
				Sigma[a][i,j] = exp(hypers[a][1])*exp(-1*(quad_form(hypers_diag,diff)));
			  Sigma[a][j,i] = Sigma[a][i,j];
			}
		}

		for (k in 1:Nd){

			p1 = (1000.5*exp(y[a][k]) - 0.5)/(1000*(1+exp(y[a][k])));	
			if(p1 > 0.9999)
       p1 = 0.9999;
	    if(p1 < 0.0001)
       p1 = 0.0001;   	

			Sigma[a][k,k] = exp(hypers[a][1]) + 1/(1000*p1*(1-p1));
		}
	L_K[a] = cholesky_decompose(Sigma[a]);
	}
}


parameters {
	vector[13] myparams;
}

model {
  
	vector[24] meanfun;  
	vector[48] pred_mean;
	vector[48] pred_var;
	real p;
  real nugget;
	matrix[11,11] hypers_diag2;	
	vector[Nd] K[24];
	vector[Nd] v_pred[24];
	vector[11] diff2;
	matrix[1,Nd] K_transpose_div_Sigma[24];

	vector[1] mu[24];
	
	
	vector[Nd] K_div_y1[24];
	vector[24] f2_mu;
	vector[24] cov_f2;

	for(a in 1:24){
		//print(a);
		meanfun[a] = coeffs[a][1] + dot_product(coeffs[a][2:12],myparams[1:11]);

		hypers_diag2 = diag_matrix(exp(hypers[a][2:12]));

  	for (k in 1:Nd){
     diff2 = x[k]-myparams[1:11];								
	 	 K[a][k] = exp(hypers[a][1])*exp(-1*(quad_form(hypers_diag2,diff2)));
	 	}

		K_div_y1[a] = mdivide_left_tri_low(L_K[a], y[a]);
		K_div_y1[a] = mdivide_right_tri_low(K_div_y1[a]',L_K[a])';
		f2_mu[a] = (K[a]' * K_div_y1[a]);	
		v_pred[a] = mdivide_left_tri_low(L_K[a], K[a]);
		p = ((exp(meanfun[a]) * 1000.5) - 0.5 )/(1000 * (1 + exp(meanfun[a])));
  	if(p > 0.9999)
       p = 0.9999;
    if(p < 0.0001)
       p = 0.0001;   	

	nugget = 1/(1000*p*(1-p));

	
	cov_f2[a] = (exp(hypers[a][1]) + nugget) - v_pred[a]' * v_pred[a];
		pred_mean[a] = meanfun[a] + f2_mu[a];
	}	

	for(a in 1:9){
		pred_var[a] = cov_f2[a] + exp(myparams[12])^2; 
	}

	for(a in 10:24){
		pred_var[a] = cov_f2[a] + exp(myparams[13])^2; 
	}

	myparams[1:11] ~ multi_normal(prior_mean[1:11], prior_var[1:11,1:11]);

	pow(exp(myparams[12]), -2) ~ gamma(2, 0.12);
	 target +=  log(4) -2*myparams[12];
  
	 pow(exp(myparams[13]), -2) ~ gamma(0.75, 0.05);
	  target += log(4) -2*myparams[13];
  
	yobs[1:24] ~ multi_normal(pred_mean[1:24],diag_matrix(pred_var[1:24]));
	yobs[25:48] ~ multi_normal(pred_mean[1:24],diag_matrix(pred_var[1:24]));
}

