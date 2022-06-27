#include "matrices.h"
int main() {
  int n = 158; //sample size
  int p = 51; //number of variables
  int i;
  int A[] = {2,5,10};//indices of the variables present in the regression
  int lenA = 3; //number of indices
  char datafilename[] = "erdata.txt";
  //allocate the data matrix
  gsl_matrix* data = gsl_matrix_alloc(n,p);
  //read the data
  FILE* datafile = fopen(datafilename,"r");
  if(NULL==datafile)
  {
    fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
return(0); }
  if(0!=gsl_matrix_fscanf(datafile,data))
  {
    fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
return(0); }
  fclose(datafile);
  printf("Marginal likelihood of regression [1|%d",A[0]);
  for(i=1;i<lenA;i++)
  {
    printf(",%d",A[i]);
  }
  printf("] = %.5lf\n",marglik(data,lenA,A));
  //free memory
  gsl_matrix_free(data);
return(1); }