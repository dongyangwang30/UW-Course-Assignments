#include "final.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

//prints the elements of a matrix in a file
void printmatrix(char* filename,gsl_matrix* m)
{
	int i,j;
	double s;
	FILE* out = fopen(filename,"w");
	
	if(NULL==out)
	{
		printf("Cannot open output file [%s]\n",filename);
		exit(1);
	}
	for(i=0;i<m->size1;i++)
	{
	        fprintf(out,"%.3lf",gsl_matrix_get(m,i,0));
		for(j=1;j<m->size2;j++)
		{
			fprintf(out,"\t%.3lf",
				gsl_matrix_get(m,i,j));
		}
		fprintf(out,"\n");
	}
	fclose(out);
	return;
}

//creates the transpose of the matrix m
gsl_matrix* transposematrix(gsl_matrix* m)
{
	int i,j;
	
	gsl_matrix* tm = gsl_matrix_alloc(m->size2,m->size1);
	
	for(i=0;i<tm->size1;i++)
	{
		for(j=0;j<tm->size2;j++)
		{
		  gsl_matrix_set(tm,i,j,gsl_matrix_get(m,j,i));
		}
	}	
	
	return(tm);
}

//calculates the product of a nxp matrix m1 with a pxl matrix m2
//returns a nxl matrix m
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m)
{
	int i,j,k;
	double s;
	
	for(i=0;i<m->size1;i++)
	{
	  for(k=0;k<m->size2;k++)
	  {
	    s = 0;
	    for(j=0;j<m1->size2;j++)
	    {
	      s += gsl_matrix_get(m1,i,j)*gsl_matrix_get(m2,j,k);
	    }
	    gsl_matrix_set(m,i,k,s);
	  }
	}
	return;
}


//computes the inverse of a positive definite matrix
//the function returns a new matrix which contains the inverse
//the matrix that gets inverted is not modified
gsl_matrix* inverse(gsl_matrix* K)
{
	int j;
	
	gsl_matrix* copyK = gsl_matrix_alloc(K->size1,K->size1);
	if(GSL_SUCCESS!=gsl_matrix_memcpy(copyK,K))
	{
		printf("GSL failed to copy a matrix.\n");
		exit(1);
	}
	
	gsl_matrix* inverse = gsl_matrix_alloc(K->size1,K->size1);
	gsl_permutation *myperm = gsl_permutation_alloc(K->size1);
	
	if(GSL_SUCCESS!=gsl_linalg_LU_decomp(copyK,myperm,&j))
	{
		printf("GSL failed LU decomposition.\n");
		exit(1);
	}
	if(GSL_SUCCESS!=gsl_linalg_LU_invert(copyK,myperm,inverse))
	{
		printf("GSL failed matrix inversion.\n");
		exit(1);
	}
	gsl_permutation_free(myperm);
	gsl_matrix_free(copyK);
	
	return(inverse);
}

//creates a submatrix of matrix M
//the indices of the rows and columns to be selected are
//specified in the last four arguments of this function
gsl_matrix* MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn)
{
	int i,j;
	gsl_matrix* subM = gsl_matrix_alloc(lenIndRow,lenIndColumn);
	
	for(i=0;i<lenIndRow;i++)
	{
		for(j=0;j<lenIndColumn;j++)
		{
			gsl_matrix_set(subM,i,j,
                                       gsl_matrix_get(M,IndRow[i],IndColumn[j]));
		}
	}
	
	return(subM);
}


//computes the log of the determinant of a symmetric positive definite matrix
double logdet(gsl_matrix* K)
{
        int i;

	gsl_matrix* CopyOfK = gsl_matrix_alloc(K->size1,K->size2);
	gsl_matrix_memcpy(CopyOfK,K);
	gsl_permutation *myperm = gsl_permutation_alloc(K->size1);
	if(GSL_SUCCESS!=gsl_linalg_LU_decomp(CopyOfK,myperm,&i))
	{
		printf("GSL failed LU decomposition.\n");
		exit(1);
	}
	double logdet = gsl_linalg_LU_lndet(CopyOfK);
	gsl_permutation_free(myperm);
	gsl_matrix_free(CopyOfK);
	return(logdet);
}

// From HW6, Regression Models

//tests if two vectors are equal
//we assume that the two vectors are sorted
int sameregression(int lenA1,int* A1,int lenA2,int* A2)
{
  int i;

  if(lenA1!=lenA2)
  {
    return 0;
  }

  //the two vectors have the same length
  //are their elements equal?
  for(i=0;i<lenA1;i++)
  {
     if(A1[i]!=A2[i])
     {
       return 0;
     }
  }

  return 1;
}

//this function adds a new regression with predictors A
//to the list of regressions. Here "regressions" represents
//the head of the list, "lenA" is the number of predictors
//and "logmarglikA" is the marginal likelihood of the regression
//with predictors A

// Changed to suit printing needs of the result
void AddRegression(int nMaxRegs, LPRegression regressions, int lenA,int* A,gsl_matrix* beta, double mc_likelihood, double laplace_likelihood)
{
  int i; 
  LPRegression p = regressions;
  LPRegression pnext = p->Next;

  while(NULL!=pnext)
  {
     //return if we have previously found this regression
     if(sameregression(lenA,A,pnext->lenA,pnext->A))
     {
        return;
     }

     //go to the next element in the list if the current
     //regression has a larger log marginal likelihood than
     //the new regression A
     if(pnext->mc_likelihood>mc_likelihood)
     {
        p = pnext;
        pnext = p->Next;
     }
     else //otherwise stop; this is where we insert the new regression
     {
        break;
     }
  }

  //create a new element of the list
  LPRegression newp = new Regression;
  newp->lenA = lenA;
  newp->mc_likelihood = mc_likelihood;
  newp->laplace_likelihood = laplace_likelihood;
  newp->A = new int[lenA];
  newp->beta = new double[lenA+1];
  
  //copy the predictors
  for(i=0;i<lenA;i++)
  {
    newp->A[i] = A[i];
  }

  //copy beta
  for(i=0;i<lenA+1;i++)
  {
    newp->beta[i] = gsl_matrix_get(beta, i, 0);
  }

  //insert the new element in the list
  p->Next = newp;
  newp->Next = pnext;

  if(lenA==1) {
    printf("inserted [%d]\n",A[0]);
    printf("\n");
  }

  int len; //Tracking length of the linked list
  len =0;

  // Start counting length
  LPRegression reg = regressions;
  LPRegression regnext = reg->Next;

  while(regnext) {
    len += 1;
    reg = regnext;
    regnext = reg->Next;
  }

  // Delete the last regression whenever we have more than we need
  // Only need to delete last one since we insert one each time
  if(len > nMaxRegs) {
      DeleteLastRegression(regressions);
  }
  return;
}

//this function deletes the last element of the list
//with the head "regressions"
//again, the head is not touched
void DeleteLastRegression(LPRegression regressions)
{
  //this is the element before the first regression
  LPRegression pprev = regressions;
  //this is the first regression
  LPRegression p = regressions->Next;

  //if the list does not have any elements, return
  if(NULL==p)
  {
     return;
  }

  //the last element of the list is the only
  //element that has the "Next" field equal to NULL
  while(NULL!=p->Next)
  {
    pprev = p;
    p = p->Next;
  }
  
  //now "p" should give the last element
  //delete it
  delete[] p->A;
  p->Next = NULL;
  delete p;

  //now the previous element in the list
  //becomes the last element
  pprev->Next = NULL;

  return;
}

//this function deletes all the elements of the list
//with the head "regressions"
//remark that the head is not touched
void DeleteAllRegressions(LPRegression regressions)
{
  //this is the first regression
  LPRegression p = regressions->Next;
  LPRegression pnext;

  while(NULL!=p)
  {
    //save the link to the next element of p
    pnext = p->Next;

    //delete the element specified by p
    //first free the memory of the vector of regressors
    delete[] p->A;
	delete[] p->beta;
    p->Next = NULL;
    delete p;

    //move to the next element
    p = pnext;
  }

  return;
}

//this function saves the regressions in the list with
//head "regressions" in a file with name "filename"
// Revised to print 5 things
void SaveRegressions(char* filename,LPRegression regressions)
{
  int i;
  //open the output file
  FILE* out = fopen(filename,"w");
	
  if(NULL==out)
  {
    printf("Cannot open output file [%s]\n",filename);
    exit(1);
  }

  //this is the first regression
  LPRegression p = regressions->Next;
  while(NULL!=p)
  {
    //now save the predictors
    for(i=0;i<p->lenA;i++)
    {
       fprintf(out,"\t%d",p->A[i]);
    }
    fprintf(out,"\n");

    //print the log marginal likelhood and the number of predictors
    fprintf(out,"%.5lf\t%.5lf",p->mc_likelihood, p->laplace_likelihood);

	for(i=0;i<2;i++)
    {
       fprintf(out,"\t%.5lf",p->beta[i]);
    }
    fprintf(out,"\n");

    //go to the next regression
    p = p->Next;
  }

  //close the output file
  fclose(out);

  return;
}

// From HW 7, random generation

gsl_matrix* makeCholesky(gsl_matrix* K)
{
	int i, j;
	gsl_matrix* Phi = gsl_matrix_alloc(K->size1,K->size2);

	if(GSL_SUCCESS!=gsl_matrix_memcpy(Phi,K))
        {
                printf("GSL failed to copy a matrix.\n");
                exit(1);
        }
    
	if(GSL_SUCCESS!=gsl_linalg_cholesky_decomp(Phi))
        {
                printf("GSL failed Cholesky decomposition.\n");
                exit(1);
        }

	for(i=0;i<Phi->size1;i++) {
		for(j=i+1;j<Phi->size2;j++) {
			gsl_matrix_set(Phi, i, j, 0.0);
		}
	}
	return(Phi);
}

// Modified to include means
void randomMVN(gsl_rng* mystream, gsl_matrix* samples, gsl_matrix* sigma, gsl_matrix* means)
{
	int i,j;
	gsl_matrix* psi = makeCholesky(sigma);
	
	gsl_matrix* z = gsl_matrix_alloc(sigma->size2, 1);
	gsl_matrix* x = gsl_matrix_alloc(sigma->size2, 1);
	gsl_vector* v = gsl_vector_alloc(sigma->size2);
	
	for(i = 0; i < samples->size2; i++) {
		for(j = 0; j < sigma->size1; j++) {
			gsl_matrix_set(z, j, 0, gsl_ran_ugaussian(mystream));
		}
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, psi, z, 0.0, x);
		gsl_matrix_add(x, means);
		gsl_matrix_get_col(v, x, 0);
		gsl_matrix_set_col(samples, i, v);
	}

	gsl_matrix_free(psi);
	gsl_matrix_free(z);
	gsl_matrix_free(x);
	gsl_vector_free(v);
}

// Now the main functions are as follows, translated from R

// Inverse logit function
double inverseLogit(double x) {
	return(exp(x)/(1.0+exp(x)));
}

// function for the computation of the Hessian
double inverseLogit2(double x) {
	return(exp(x)/pow(1.0+exp(x), 2.0));
}

// Computes pi_i = P(y_i = 1 | x_i)
gsl_matrix* getPi(int n, gsl_matrix* mat, gsl_matrix* beta) {

	gsl_matrix* x0 = gsl_matrix_alloc(n, 2);
	gsl_matrix* res = gsl_matrix_alloc(n, 1);
	int i;

	for(i=0;i<n;i++) {
		// Intercept
		gsl_matrix_set(x0, i, 0, 1.0);

		// Each predictor
		gsl_matrix_set(x0, i, 1, gsl_matrix_get(mat, i, 0));
	}

	// x0 %*% beta
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, x0, beta, 0.0, res);

	// Inverse logit
	for(i=0;i<n;i++) {
		gsl_matrix_set(res, i, 0, inverseLogit(gsl_matrix_get(res, i, 0)));
	}

	// Free memory
	gsl_matrix_free(x0);

	return(res);
}

// Another function for the computation of the Hessian
gsl_matrix* getPi2(int n, gsl_matrix* mat, gsl_matrix* beta) {

	gsl_matrix* x0 = gsl_matrix_alloc(n, 2);
	gsl_matrix* res = gsl_matrix_alloc(n, 1);
	int i;

	for(i=0;i<n;i++) {
		// Intercept
		gsl_matrix_set(x0, i, 0, 1.0);

		// Each predictor
		gsl_matrix_set(x0, i, 1, gsl_matrix_get(mat, i, 0));
	}

	// x0 %*% beta
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, x0, beta, 0.0, res);

	// Inverse logit
	for(i=0;i<n;i++) {
		gsl_matrix_set(res, i, 0, inverseLogit2(gsl_matrix_get(res, i, 0)));
	}

	// Free memory
	gsl_matrix_free(x0);

	return(res);
}

// Logistic log-likelihood (formula (3) in your handout)
double logisticLoglik(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta) {

	// Get Pi
	gsl_matrix* Pi = getPi(n, x, beta);

	double logisticLoglik = 0;
	int i;

	// Log likelihood
	for(i=0;i<n;i++) {
		double yi = gsl_matrix_get(y, i, 0);
		double pi = gsl_matrix_get(Pi, i, 0);

		logisticLoglik += yi*log(pi) + (1.0-yi)*log(1.0-pi);
	}

	// Free memory
	gsl_matrix_free(Pi);

	return(logisticLoglik);
}

// Logistic log-likelihood for l*
double logisticLoglik1(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta) {

	int i,j;

	// Part1
	double Part1 = -1.0*log(2.0*M_PI);
	double Part2 = 0;
	
	// Part2
	for(i=0;i<(beta->size1);i++) {
		for(j=0;j<(beta->size2);j++) {
			Part2 -= 0.5 * pow(gsl_matrix_get(beta, i, j), 2.0);
		}
	}

	// Part3: Log likelihood
	double Part3 = logisticLoglik(n,y,x,beta);

	// Calculate logisticLoglik_star
	double logisticLoglik_star =  Part1 + Part2 + Part3;

	return(logisticLoglik_star);
}

// Obtain the Hessian for Newton-Raphson
gsl_matrix* getHessian(int n, gsl_matrix* x, gsl_matrix* beta) {

	int i;

	// Initialize Hessian
	gsl_matrix* hessian = gsl_matrix_alloc(2, 2);

	double hessian00 = 0.0;
	double hessian01 = 0.0;
	double hessian10 = 0.0;
	double hessian11 = 0.0;

	// Get Pi2
	gsl_matrix* Pi2 = getPi2(n, x, beta);

	// Calculate each element
	for(i=0;i<n;i++) {
		double pi = gsl_matrix_get(Pi2, i, 0);
		double xi = gsl_matrix_get(x, i, 0);

		hessian00 += pi;
		hessian01 += pi*xi;
		hessian11 += pi*pow(xi, 2.0);
	}

	hessian00 += 1.0;
	hessian10 = hessian01;
	hessian11 += 1.0;

	gsl_matrix_set(hessian, 0, 0, -hessian00);
	gsl_matrix_set(hessian, 1, 0, -hessian10);
	gsl_matrix_set(hessian, 0, 1, -hessian01);
	gsl_matrix_set(hessian, 1, 1, -hessian11);

	// Free memory
	gsl_matrix_free(Pi2);
	return(hessian);
}

// Obtain the gradient for Newton-Raphson
gsl_matrix* getGradient(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta) {

	int i;

	// Initialize Gradient
	gsl_matrix* gradient = gsl_matrix_alloc(2, 1);

	double gradient00 = 0.0;
	double gradient10 = 0.0;

	// Get Pi
	gsl_matrix* Pi = getPi(n, x, beta);

	// Calculate elements
	for(i=0;i<n;i++) {
		double pi = gsl_matrix_get(Pi, i, 0);
		double xi = gsl_matrix_get(x, i, 0);
		double yi = gsl_matrix_get(y, i, 0);

		gradient00 += yi - pi;
		gradient10 += (yi - pi)*xi;
	}

	gradient00 -= gsl_matrix_get(beta, 0, 0);
	gradient10 -= gsl_matrix_get(beta, 1, 0);

	gsl_matrix_set(gradient, 0, 0, gradient00);
	gsl_matrix_set(gradient, 1, 0, gradient10);

	// Free memory
	gsl_matrix_free(Pi);
	return(gradient);
}

// This function implements our own Newton-Raphson procedure
gsl_matrix* getcoefNR(int n, gsl_matrix* y, gsl_matrix* x) {

	int iter = 0;
	double thres = 0.000001;
	double newLoglik;
	double currentLoglik;
	
	// Initialize matrices
	gsl_matrix* beta = gsl_matrix_alloc(2, 1);
	gsl_matrix_set_zero(beta);

	currentLoglik = logisticLoglik1(n, y, x, beta);

	gsl_matrix* hessian = gsl_matrix_alloc(2, 2);
	gsl_matrix* gradient = gsl_matrix_alloc(2, 1);
	gsl_matrix* hessianInv = gsl_matrix_alloc(2, 2);
	gsl_matrix* hessian_prod = gsl_matrix_alloc(2, 1);
	gsl_matrix* newbeta = gsl_matrix_alloc(2, 1);

	// For 10000 iterations at most
	while(iter < 10000) {
		// Hessian and gradient
		gsl_matrix* hessian = getHessian(n, x, beta);
		gsl_matrix* gradient = getGradient(n, y, x, beta);

		// Hessian inverse
		hessianInv = inverse(hessian);
	
		// Get product
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, hessianInv, gradient, 0.0, hessian_prod);
	
		// Update newbeta
		gsl_matrix_memcpy(newbeta, beta);

		// gsl_matrix_sub not working
		int i,j;
		for(i=0;i<2;i++){
			for(j=0;j<1;j++){
				gsl_matrix_set(newbeta, i, j,
					(gsl_matrix_get(beta, i, j)- gsl_matrix_get(hessian_prod, i,j) ) );
			}
		}

		newLoglik = logisticLoglik1(n, y, x, newbeta);
	
		/*
		// At each iteration the log-likelihood must increase
		if(newLoglik < currentLoglik) {
			printf("Coding error!");
			exit(1);
		}
		*/
		
		// Update beta
		gsl_matrix_memcpy(beta, newbeta);

		// Stop if the log-likelihood does not improve by too much
		if((newLoglik - currentLoglik) < thres) {
			break;
		} else {
			currentLoglik = newLoglik;
		}

		iter += 1;
	}

	// Free memory
	gsl_matrix_free(newbeta);
	gsl_matrix_free(hessian);
	gsl_matrix_free(gradient);
	gsl_matrix_free(hessianInv);
	gsl_matrix_free(hessian_prod);

	return(beta);
}

// Laplace log likelihood
double getLaplaceApprox(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode) {

	// Get Hessian*(-1)
	gsl_matrix* hessian = getHessian(n, x, betaMode);
	gsl_matrix_scale(hessian, -1.0);

	// Calculate logisticLoglik_star
	double logisticLoglik_star = logisticLoglik1(n, y, x, betaMode);

	// Calculate log marginal likelihood
	double log_pd = log(2.0*M_PI) + logisticLoglik_star - 0.5*logdet(hessian);

	// Free memory
	gsl_matrix_free(hessian);

	return(log_pd);
}

// Calculates the posterior means
gsl_matrix* getPosteriorMeans(gsl_rng* mystream, int n,  gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode) {

	int i;
	double result;
	double u;

	// Initialization
	gsl_matrix* samples = gsl_matrix_alloc(10000, 2);
	gsl_matrix_set_zero(samples);

	gsl_matrix* hessian = getHessian(n, x, betaMode);
	gsl_matrix* hessian_Inv = gsl_matrix_alloc(2, 2);
	gsl_matrix* hessian_inv = inverse(hessian);
	gsl_matrix_scale(hessian_inv, -1.0);

	gsl_matrix* currentBeta = gsl_matrix_alloc(2,1);
	gsl_matrix_memcpy(currentBeta, betaMode);
	gsl_matrix* newbeta = gsl_matrix_alloc(2, 1);

	// printf("Everything good till simulation");
	// Repeat 10000 times for updating the beta's
	for(i=0; i<10000;i++) {
		//printf("Loop starts");
		// Draw sample beta's from multivariate normal
		randomMVN(mystream, newbeta, hessian_inv, currentBeta);

		//printf("random ok");
		// Compare the l* so to determine whether to update the markov chain
		result = logisticLoglik1(n, y, x, newbeta) - logisticLoglik1(n, y, x, currentBeta);

		//printf("Everything good before checking");
		if(result >= 0.0) {
			gsl_matrix_memcpy(currentBeta, newbeta);		
		} else {
			u = gsl_ran_flat(mystream, 0.0, 1.0);
			if(log(u) <= result) {
				gsl_matrix_memcpy(currentBeta, newbeta);
			}
		}
		//printf("Everything good after for loop in sim");

		// Update chain
		gsl_matrix_set(samples, i, 0, gsl_matrix_get(currentBeta, 0, 0));
		gsl_matrix_set(samples, i, 1, gsl_matrix_get(currentBeta, 1, 0));
	}

	//printf("Everything good after simulation");

	gsl_vector_view a;

	gsl_matrix* sapmle_means = gsl_matrix_alloc(2, 1);

	for(i=0; i<samples->size2; i++) {
		a = gsl_matrix_column(samples, i);
		gsl_matrix_set(sapmle_means, i, 0, 
			gsl_stats_mean(a.vector.data, a.vector.stride, (samples->size1)));
	}

	// Free memory
	gsl_matrix_free(hessian);
	gsl_matrix_free(hessian_inv);
	gsl_matrix_free(currentBeta);
	gsl_matrix_free(newbeta);
	gsl_matrix_free(samples);

	return(sapmle_means);
}


// This function gives likelihood for MC
double MC_likelihood(gsl_rng* mystream, int n, gsl_matrix* y, gsl_matrix* x) {

	int i;
	double sum = 0;
	double mc_likelihood;

	// Initialization
	gsl_matrix* prior_means = gsl_matrix_alloc(2, 1);
	gsl_matrix_set_zero(prior_means);
	gsl_matrix* prior_cov = gsl_matrix_alloc(2, 2);
	gsl_matrix_set_identity(prior_cov);

	// Beta's
	gsl_matrix* beta_j = gsl_matrix_alloc(2, 1);
	for(i=0;i<10000;i++) {
		// Sampling
		randomMVN(mystream, beta_j, prior_cov, prior_means);
		// Likelihood
		sum += logisticLoglik(n,y,x,beta_j);
	}

	mc_likelihood = sum/10000.0;

	// Free memory
	gsl_matrix_free(prior_means);
	gsl_matrix_free(prior_cov);
	gsl_matrix_free(beta_j);

	return(mc_likelihood);
}