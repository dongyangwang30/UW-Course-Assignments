/*
 This program uses GSL to create a sequence of 10000
 draws from Normal (0,Sigma)
*/

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include "matrices.h"

int main()
{
  const gsl_rng_type* T;
  gsl_rng* r;

  char datafilename[] = "erdata.txt"; //name of the data file

  int i;
  int n = 158;
  int p = 51;
  int m = 10000;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc(T);

  gsl_matrix* X = gsl_matrix_alloc(n, p);
	FILE * erdata = fopen(datafilename, "r");
	gsl_matrix_fscanf(erdata, X);
	fclose(erdata);

  gsl_matrix* covX = gsl_matrix_alloc(p, p);
	makeCovariance(covX,X);
  printmatrix("Cov.txt", covX);

  gsl_matrix* samples = gsl_matrix_alloc(m, p);
	randomMVN(r, samples, covX);

  gsl_matrix* sampleCov = gsl_matrix_alloc(p, p);
	makeCovariance(sampleCov, samples);
	printmatrix("sampleCov.txt", sampleCov);

	gsl_matrix_free(X);
	gsl_matrix_free(covX);
	gsl_matrix_free(samples);
  gsl_rng_free(r);
  gsl_matrix_free(sampleCov);

  return(1);
}
