/*
 FILE: MAIN.CPP

 This program creates a linked list with all the regressions with 1 or 2 predictors.
 This list keeps the regressions in decreasing order of their marginal likelihoods.
*/


#include "matrices.h"
#include "regmodels.h"

int main()
{
  int i,j;
  int nMaxRegs = 10; //Choose top 10

  int n = 158; //sample size
  int p = 51; //number of variables
  char datafilename[] = "erdata.txt"; //name of the data file
  char outputfilename[] = "result.txt";

  //allocate the data matrix
  double** data = allocmatrix(n,p);

  //read the data
  readmatrix(datafilename,n,p,data);

  //create the head of the list of regressions
  LPRegression regressions = new Regression;
  //properly mark the end of the list
  regressions->Next = NULL;

  int A[p-1]; //indices of the variables present in the regression
  int lenA = -1; //number of indices

  //add the regressions with one or two predictors
  
  for(i=1;i<p;i++)
  {
    lenA = 1;
    A[0] = i+1;
    AddRegression(nMaxRegs,regressions,
                  lenA, A,
                  marglik(n,p,data,lenA,(int*)A));
    for(j = 1; j<p; j++) // We don't want two predictors to be same, so keep only if j>i
    {
      if(j > i) {
        lenA = 2;
        A[1] = j+1;
        AddRegression(nMaxRegs,regressions,
                  lenA, A,
                  marglik(n,p,data,lenA,(int*)A));
      }
    }
  }

  //save the list in a file
  SaveRegressions(outputfilename,regressions);

  //delete all regressions
  DeleteAllRegressions(regressions);

  //free memory
  freematrix(n,data);
  delete regressions; regressions = NULL;

  return(1);
}
