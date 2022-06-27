#include "matrices.h"

int main() {
  int n = 158; //sample size
  int p = 51; //number of variables
  int i;
  int A[] = {2,5,10};//indices of the variables present in the regression
  int lenA = 3; //number of indices
  char datafilename[] = "erdata.txt";
  //allocate the data matrix
  double** data = allocmatrix(n,p);
  //read the data
  readmatrix(datafilename,n,p,data);
  printf("Marginal likelihood of regression [1|%d",A[0]);
  for(i=1;i<lenA;i++)
  {
    printf(",%d",A[i]);
  }
  printf("] = %.5lf\n",marglik(n,p,data,lenA,A));
  //free memory
  freematrix(n,data);
return(1); }
