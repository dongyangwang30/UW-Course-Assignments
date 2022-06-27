#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

//THESE ARE GSL FUNCTIONS
//YOU DO NOT NEED TO INCLUDE ALL THESE HEADER FILES IN YOUR CODE
//JUST THE ONES YOU ACTUALLY NEED;
//IN THIS APPLICATION, WE ONLY NEED gsl/gsl_matrix.h
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_errno.h>

void printmatrix(char* filename,gsl_matrix* m);
gsl_matrix* transposematrix(gsl_matrix* m);
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m);
gsl_matrix* inverse(gsl_matrix* K);
gsl_matrix* MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn);
double logdet(gsl_matrix* K);
void makeCovariance(gsl_matrix* covX, gsl_matrix* X);
gsl_matrix* makeCholesky(gsl_matrix* K);
void randomMVN(gsl_rng* mystream, gsl_matrix* samples, gsl_matrix* sigma);