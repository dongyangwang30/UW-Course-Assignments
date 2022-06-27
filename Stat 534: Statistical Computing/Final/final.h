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

// These are original functions from previous homeworks. (Modified)
void printmatrix(char* filename,gsl_matrix* m);
gsl_matrix* transposematrix(gsl_matrix* m);
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m);
gsl_matrix* inverse(gsl_matrix* K);
gsl_matrix* MakeSubmatrix(gsl_matrix* M,
			  int* IndRow,int lenIndRow,
			  int* IndColumn,int lenIndColumn);
double logdet(gsl_matrix* K);
gsl_matrix* makeCholesky(gsl_matrix* K);
void randomMVN(gsl_rng* mystream, gsl_matrix* samples, gsl_matrix* sigma, gsl_matrix* means);

// The following functions are written for the calculation of log likelihoods
double inverseLogit(double x);
double inverseLogit2(double x);
gsl_matrix* getPi(int n, gsl_matrix* mat, gsl_matrix* beta);
gsl_matrix* getPi2(int n, gsl_matrix* mat, gsl_matrix* beta);
double logisticLoglik(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
double logisticLoglik1(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);

gsl_matrix* getHessian(int n, gsl_matrix* x, gsl_matrix* beta);
gsl_matrix* getGradient(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
gsl_matrix* getcoefNR(int n, gsl_matrix* y, gsl_matrix* x);

double getLaplaceApprox(int n, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode);
gsl_matrix* getPosteriorMeans(gsl_rng* mystream, int n,  gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode);
double MC_likelihood(gsl_rng* mystream, int n, gsl_matrix* y, gsl_matrix* x);


//These are original functions from previous lectures
//this avoids including the function headers twice
#ifndef _REGMODELS
#define _REGMODELS

typedef struct myRegression* LPRegression;
typedef struct myRegression Regression;

struct myRegression // Updated to contain 2 likelihoods and beta
{
  int lenA; //number of regressors
  double* beta; 
  int* A; //regressors
  double mc_likelihood; //log marginal likelihood of the regression based on MC
  double laplace_likelihood; //log marginal likelihood of the regression based on Laplace

  LPRegression Next; //link to the next regression
};
void AddRegression(int nMaxRegs, LPRegression regressions, int lenA,int* A,gsl_matrix* beta, double mc_likelihood, double laplace_likelihood);
void DeleteAllRegressions(LPRegression regressions);
void DeleteLastRegression(LPRegression regressions);
void SaveRegressions(char* filename,LPRegression regressions);

#endif