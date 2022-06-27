/*
 FILE: MAIN.CPP

 This program creates a linked list with all the regressions with 1 or 2 predictors.
 This list keeps the regressions in decreasing order of their marginal likelihoods.
*/


#include "matrices.h"

double recursive_det(double** data, int n){
	double det;
  int j;
	if (n == 1){
	  return((double)data[0][0]);
	}
	else if (n ==2) {
    double determ = data[0][0] * data[1][1] - data[0][1]*data[1][0];
    //printf("determ is : %.4lf\n",  determ);
	  return(determ);
	}
	else{
	  for (j = 0; j<n; j++) {
	  	double** data1 = allocmatrix(n-1,n-1);
		  int i;
		  int k =1;
      for (k = 1; k<n; k++){
        for (i = 0; i<n ;i++){
				  if (i > j){
					  data1[k - 1][i -1] = data[k][i];
				  }
          else if (i <j){
            data1[k - 1][i] = data[k][i];
          }
			  }
		  }
      //printf("data[0][j] after step %d is : %.4lf\n", j, data[0][j]);
      double part1 = (double)data[0][j];
      double part2 = pow(-1.0, (double)j);
      //printf("part1&2 after step %d is : %.4lf\n and %.4lf\n", j, part1, part2);
		  det +=  part1 * part2 * (double)recursive_det(data1, n-1);
      //printf("result after step %d is : %.4lf\n", j, det);
      //printf("Data1 matrix is : %.4lf\n %.4lf\n %.4lf\n %.4lf\n", data1[0][0],  data1[0][1],  data1[1][0],  data1[1][1]);
    }
	}
  return(det);
}

int main()
{
  int i,j;

  int n = 10; //number of rows and columns

  char datafilename[] = "mybandedmatrix.txt"; //name of the data file

  //allocate the data matrix
  double** data = allocmatrix(n,n);

  //read the data
  readmatrix(datafilename,n,n,data);
  
  double res = recursive_det(data,n);

  //free memory
  freematrix(n,data);

  // Print determinant to console
	printf("The determinant of the matrix is: %.4lf\n", res);

  return(1);
}
