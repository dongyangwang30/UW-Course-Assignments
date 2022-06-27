#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

extern "C" {
#include <clapack.h>
}

extern "C" {
 void dpotri_(char*,int*,double*,int*,int*);
 void dgeev_(char*, char*, int*, double*, int*, double*, double*, double*,
	 int*, double*, int*, double*, int*, int*);
}

double ** allocmatrix(int n,int p);
void freematrix(int n,double** m);
void copymatrix(int n,int p,double** source,double** dest);
void readmatrix(char* filename,int n,int p,double* m[]);
void printmatrix(char* filename,int n,int p,double** m);
double** transposematrix(int n,int p,double** m);
void dotmatrixproduct(int n,int p,double** m1,double** m2,double** m);
void matrixproduct(int n,int p,int l,double** m1,double** m2,double** m);
void inverse(int p,double** m);
double logdet(int p,double** m);
