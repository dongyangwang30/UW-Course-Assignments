#include "matrices.h"

//allocates the memory for a matrix with 
//n rows and p columns
double ** allocmatrix(int n,int p)
{
	int i;
	double** m;
	
	m = new double*[n];
	for(i=0;i<n;i++)
	{
		m[i] = new double[p];
		memset(m[i],0,p*sizeof(double));
	}
	return(m);
}

//frees the memory for a matrix with n rows
void freematrix(int n,double** m)
{
	int i;
	
	for(i=0;i<n;i++)
	{
		delete[] m[i]; m[i] = NULL;
	}
	delete[] m; m = NULL;
	return;
}

//creates the copy of a matrix with n rows and p columns
void copymatrix(int n,int p,double** source,double** dest)
{
	int i,j;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			dest[i][j] = source[i][j];
		}
	}
	return;
}

//reads from a file a matrix with n rows and p columns
void readmatrix(char* filename,int n,int p,double* m[])
{
	int i,j;
	double s;
	FILE* in = fopen(filename,"r");
	
	if(NULL==in)
	{
		printf("Cannot open input file [%s]\n",filename);
		exit(1);
	}
	for(i=0;i<n;i++)
	{
		for(j=0;j<p;j++)
		{
			fscanf(in,"%lf",&s);
			m[i][j] = s;
		}
	}
	fclose(in);
	return;
}

//prints the elements of a matrix in a file
void printmatrix(char* filename,int n,int p,double** m)
{
	int i,j;
	double s;
	FILE* out = fopen(filename,"w");
	
	if(NULL==out)
	{
		printf("Cannot open output file [%s]\n",filename);
		exit(1);
	}
	for(i=0;i<n;i++)
	{
		fprintf(out,"%.3lf",m[i][0]);
		for(j=1;j<p;j++)
		{
			fprintf(out,"\t%.3lf",m[i][j]);
		}
		fprintf(out,"\n");
	}
	fclose(out);
	return;
}

//creates the transpose of the matrix m
double** transposematrix(int n,int p,double** m)
{
	int i,j;
	
	double** tm = allocmatrix(p,n);
	
	for(i=0;i<p;i++)
	{
		for(j=0;j<n;j++)
		{
			tm[i][j] = m[j][i];
		}
	}	
	
	return(tm);
}

//calculates the dot (element by element) product of two matrices m1 and m2
//with n rows and p columns; the result is saved in m
void dotmatrixproduct(int n,int p,double** m1,double** m2,double** m)
{
	int i,j;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<p;j++)
		{
			m[i][j] = m1[i][j]*m2[i][j];
		}
	}
	
	return;
}

//calculates the product of a nxp matrix m1 with a pxl matrix m2
//returns a nxl matrix m
void matrixproduct(int n,int p,int l,double** m1,double** m2,double** m)
{
	int i,j,k;
	double s;
	
	for(i=0;i<n;i++)
	{
		for(k=0;k<l;k++)
		{
			s = 0;
			for(j=0;j<p;j++)
			{
				s += m1[i][j]*m2[j][k];
			}
			m[i][k] = s;
		}
	}
	return;
}

void set_mat_identity(int p, double *A)
{
 int i;

 for(i = 0; i < p * p; i++) A[i] = 0;
 for(i = 0; i < p; i++) A[i * p + i] = 1;
 return;
}

//computes the inverse of a symmetric positive definite matrix
void inverse(int p,double** m)
{
  int i,j,k;
  double* m_copy = (double*)malloc((p * p) * sizeof(double));
  double* m_inv = (double*)malloc((p * p) * sizeof(double));

  k=0;
  for(i=0;i<p;i++)
  {
     for(j=0;j<p;j++)
     {
        m_copy[k] = m[i][j];
        k++;
     }
  }

  set_mat_identity(p, m_inv);

  //-----  Use LAPACK  -------
  if(0!=(k=clapack_dposv(CblasRowMajor, CblasUpper, p, p, m_copy, p, m_inv, p)))
  {
    fprintf(stderr,"Something was wrong with clapack_dposv [%d]\n",k);
     exit(1);
  }
  //--------------------------

  k=0;
  for(i=0;i<p;i++)
  {
     for(j=0;j<p;j++)
     {
        m[i][j] = m_inv[k];
        k++;
     }
  }  

  free(m_copy);
  free(m_inv);

  return;
}


//computes the log of the determinant of a symmetric positive definite matrix
double logdet(int p,double** m)
{
	int i,j;
	char jobvl = 'N';
	char jobvr = 'N';
	int lda = p;
	double wr[2*p];
	double wi[2*p];
	double vl[p][p];
	int ldvl = p*p;
	double vr[p][p];
	int ldvr = p*p;
	double work[p*p];
	int lwork = p*p;
	double a[p][p];
	int info;
	
	for(i=0;i<p;i++)
	{
		for(j=0;j<p;j++)
		{
			a[i][j] = m[i][j];
		}
	}
	dgeev_(&jobvl,&jobvr,&p,(double*)a,&lda,(double*)wr,(double*)wi,(double*)vl, 
		  &ldvl,(double*)vr,&ldvr,(double*)work,&lwork,&info);

	if(0!=info)
	{
		printf("Smth wrong in the call of 'dgeev' error is [info = %d]\n",info);
		exit(1);
	}	   
	
	double logdet = 0;
	for(i=0;i<p;i++) logdet+=log(wr[i]);	
	return(logdet);
}

double marglik(int n,int p,double** data,int lenA,int* A)
{
	// Generating some key matrices for later use
	int i,j,l;

	double** D1 = allocmatrix(n,1);
	for (i=0;i<n;i++)
  	{
    	D1[i][0] = data[i][0];
	}

	double** D1T = transposematrix(n,1, D1);

	double** DA = allocmatrix(n,lenA);
	for(i=0;i<n;i++)
  	{
		
    	for(j=0;j<lenA;j++)
     	{
			l = A[j] - 1;
        	DA[i][j] = data[i][l];
     	}
  	}  

	double** DAT = transposematrix(n,lenA,DA);

	double** MA = allocmatrix(lenA,lenA);
	matrixproduct(lenA, n, lenA, DAT, DA, MA);
	
	for(i=0;i<lenA;i++)
  	{
    	for(j=0;j<lenA;j++)
     	{
			if (i == j){
				MA[i][j] += 1;
			}
     	}
  	}  

	// Part 1
	double part1 = lgamma((n + lenA + 2.0)/2.0) - lgamma((lenA + 2.0)/2.0);

	// Part 2
	double part2 = -1.0/2.0 * logdet(lenA,MA);	

	// Part 3
	// D1T D1
	double** res1 = allocmatrix(1,1);
	matrixproduct(1, n, 1, D1T, D1, res1);

	// D1T DA
	double** res2 = allocmatrix(1,lenA);
	matrixproduct(1, n, lenA, D1T, DA, res2);	

	//MA Inverse
	double** res3 = allocmatrix(lenA,lenA);
	copymatrix(lenA,lenA, MA, res3);
	inverse(lenA, res3);

	//DAT D1
	double** res4 = allocmatrix(lenA,1);
	matrixproduct(lenA, n, 1, DAT, D1, res4);	

	//res2 & res3
	double** res5 = allocmatrix(1,lenA);
	matrixproduct(1, lenA, lenA, res2, res3, res5);

	//res2 & res3 & res5
	double** res6 = allocmatrix(1,1);
	matrixproduct(1, lenA, 1, res5, res4, res6);

	double part3 = -1.0/2.0 * (n + lenA + 2.0)* log(1.0 +  **res1 - **res6);
	//printf("See part 3,%lf",part3);
	double log_likelihood = part1 + part2 + part3;
	
	// Free memory
	freematrix(n, D1);
	freematrix(n, DA);
	freematrix(1, D1T);
	freematrix(1, DAT);
	freematrix(lenA, MA);
	freematrix(1,res1);
	freematrix(1,res2);
	freematrix(lenA,res3);
	freematrix(lenA,res4);
	freematrix(1,res5);
	freematrix(1,res6);

	// return final result
	return(log_likelihood);
}