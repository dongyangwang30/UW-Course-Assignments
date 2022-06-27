#include "matrices.h"

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
double marglik(gsl_matrix* data,int lenA,int* A)
{
	// Generating some key matrices for later use
	int i;

	// Number of rows
	int row = data -> size1;

	// A Vector of rows
	int* rows = new int[row];

	for (i=0;i<row;i++)
	{
		rows[i] = i;
	}
	
	// A Vector of columns for A
	int* cols = new int[lenA];

	for (i=0;i<lenA;i++)
	{
		cols[i] = A[i] - 1;
	}

	// One col vector: RESPONSE
	int* col1 = new int[1];
	col1[0] = 0;

	// DA D1
	gsl_matrix* DA = MakeSubmatrix(data, rows,row,cols,lenA);
	gsl_matrix* D1 = MakeSubmatrix(data, rows,row,col1,1);
	
	// DAT D1T
	gsl_matrix* DAT = transposematrix(DA);
	gsl_matrix* D1T = transposematrix(D1);

	// Identity
	gsl_matrix* IA = gsl_matrix_alloc(lenA,lenA);
	gsl_matrix_set_identity(IA);
	
	// MA
	gsl_matrix* MA = gsl_matrix_alloc(lenA,lenA);
	matrixproduct(DAT,DA, MA);
	gsl_matrix_add(MA,IA);

	// Part 1
	double part1 = lgamma((row + lenA + 2.0)/2.0) - lgamma((lenA + 2.0)/2.0);

	// Part 2
	double part2 = -1.0/2.0 * logdet(MA);	

	// Part 3
	// D1T D1
	gsl_matrix* res1 = gsl_matrix_alloc(1,1);
	matrixproduct(D1T,D1, res1);

	// D1T DA
	gsl_matrix* res2 = gsl_matrix_alloc(1,lenA);
	matrixproduct(D1T,DA, res2);

	//MA Inverse
	gsl_matrix* res3= inverse(MA);

	//DAT D1
	gsl_matrix* res4 = gsl_matrix_alloc(lenA,1);
	matrixproduct(DAT,D1, res4);

	//res2 & res3
	gsl_matrix* res5 = gsl_matrix_alloc(1,lenA);
	matrixproduct(res2,res3, res5);

	//res2 & res3 & res5
	gsl_matrix* res6 = gsl_matrix_alloc(1,1);
	matrixproduct(res5,res4, res6);

	double part3 = -1.0/2.0 * (row + lenA + 2.0)* log(1.0 +  (double)gsl_matrix_get(res1,0,0)- (double)gsl_matrix_get(res6,0,0));
	//printf("See part 1,%lf",part1);
	//printf("See part 2,%lf",part2);
	//printf("See part 3,%lf",part3);
	double log_likelihood = part1 + part2 + part3;
	
	// Free memory
	delete[] rows; rows = NULL;
	delete[] col1; col1 = NULL;
	delete[] cols; cols = NULL;
	gsl_matrix_free(DA);
	gsl_matrix_free(DAT);
	gsl_matrix_free(D1);
	gsl_matrix_free(D1T);
	gsl_matrix_free(IA);
	gsl_matrix_free(MA);
	gsl_matrix_free(res1);
	gsl_matrix_free(res2);
	gsl_matrix_free(res3);
	gsl_matrix_free(res4);
	gsl_matrix_free(res5);
	gsl_matrix_free(res6);

	// return final result
	return(log_likelihood);
}


