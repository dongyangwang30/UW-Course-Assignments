#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <iomanip>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include "final.h"

// For MPI communication
#define GETR2	1
#define SHUTDOWNTAG	0

// Used to determine PRIMARY or REPLICA
static int myrank;

// Global variables
int n = 148;
int p = 61;
int response = 60;

///////////
gsl_matrix* data = gsl_matrix_alloc(n, p);
gsl_matrix* y = gsl_matrix_alloc(n, 1);

// Function Declarations
void primary();
void replica(int primaryname);
void bayesLogistic(int index, gsl_rng* mystream, double* out);

int main(int argc, char* argv[])
{
   int i;

   ///////////////////////////
   // START THE MPI SESSION //
   ///////////////////////////
   MPI_Init(&argc, &argv);

   /////////////////////////////////////
   // What is the ID for the process? //   
   /////////////////////////////////////
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

   // Read in the data
   FILE * f = fopen("534finalprojectdata.txt", "r");
   gsl_matrix_fscanf(f, data);
   fclose(f);

   // Define response variable
   for(i=0;i<n;i++) {
      gsl_matrix_set(y, i, 0, gsl_matrix_get(data, i, response));
   }

   if(myrank==0)
   {
      primary();
   }
   else
   {
      replica(myrank);
   }

   // clean memory
   gsl_matrix_free(data);
   gsl_matrix_free(y);

   // Finalize the MPI session
   MPI_Finalize();

   return(0);
}

void primary() {
   int i;		// to loop over the variables
   int rank;		// another looping variable
   int ntasks;		// the total number of replicas
   int jobsRunning;	// how many replicas we have working
   int work[1];		// information to send to the replicas
   double output[5]; // info received from the replicas
   MPI_Status status;	// MPI information

   int nMaxRegs = 5; // Maximum number of regressions to keep track of
   int A[p-1]; // Indices
   int lenA = 1; // Index number
   char outputfilename[] = "best5regressions.txt"; // The output filename for regressions

   double laplace_likelihood;
   double mc_likelihood;
   gsl_matrix* betas = gsl_matrix_alloc(2, 1); // Placeholder for coefficient estimates

   // Crreate new regression
   LPRegression regressions = new Regression;
   regressions->Next = NULL;

   // Find out how many replicas there are
   MPI_Comm_size(MPI_COMM_WORLD,&ntasks);

   fprintf(stdout, "Total Number of processors = %d\n",ntasks);

   // Now loop through the variables and compute the R2 values in
   // parallel
   jobsRunning = 1;

   for(i=0;i<response;i++) {
      // This will tell a replica which variable to work on
      work[0] = i;

      if(jobsRunning < ntasks) // Do we have an available processor?
      {
         // Send out a work request
         MPI_Send(&work, 	// the vector with the variable
		            1, 		// the size of the vector
		            MPI_INT,	// the type of the vector
                  jobsRunning,	// the ID of the replica to use
                  GETR2,	// tells the replica what to do
                  MPI_COMM_WORLD); // send the request out to anyone
				   // who is available
         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],jobsRunning);

         // Increase the # of processors in use
         jobsRunning++;

      }
      else // all the processors are in use!
      {
         MPI_Recv(output,	// where to store the results
 		            5,		// the size of the vector
		            MPI_DOUBLE,	// the type of the vector
	 	            MPI_ANY_SOURCE,
		            MPI_ANY_TAG, 	
		            MPI_COMM_WORLD,
		            &status);     // lets us know which processor
				// returned these results

         printf("Primary has received the result of work request [%d] from replica [%d]\n",
                (int) output[0],status.MPI_SOURCE);
 
         // Add the results to the regressions list
         lenA = 1;
         A[0] = (int)output[0];
         mc_likelihood = output[1];
         laplace_likelihood = output[2];
         gsl_matrix_set(betas, 0, 0, output[3]);
         gsl_matrix_set(betas, 1, 0, output[4]);

         AddRegression(nMaxRegs, regressions,
            lenA, A, betas, mc_likelihood, laplace_likelihood);

         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],status.MPI_SOURCE);

         // Send out a new work order to the processors that just
         // returned
         MPI_Send(&work,
                  1,
                  MPI_INT,
                  status.MPI_SOURCE, // the replica that just returned
                  GETR2,
                  MPI_COMM_WORLD); 
      } // using all the processors
   } // loop over all the variables


   ///////////////////////////////////////////////////////////////
   // NOTE: we still have some work requests out that need to be
   // collected. Collect those results now.
   ///////////////////////////////////////////////////////////////

   // loop over all the replicas
   for(rank=1; rank<jobsRunning; rank++)
   {
      MPI_Recv(output,
               5,
               MPI_DOUBLE,
               MPI_ANY_SOURCE,	// whoever is ready to report back
               MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &status);

       printf("Primary has received the result of work request [%d]\n",
                (int) output[0]);
 
      //save the results received
      lenA = 1;
      A[0] = (int)output[0];
      mc_likelihood = output[1];
      laplace_likelihood = output[2];
      gsl_matrix_set(betas, 0, 0, output[3]);
      gsl_matrix_set(betas, 1, 0, output[4]);

      AddRegression(nMaxRegs, regressions,
         lenA, A, betas, mc_likelihood, laplace_likelihood);
   }

   printf("Tell the replicas to shutdown.\n");

   // Shut down the replica processes
   for(rank=1; rank<ntasks; rank++)
   {
      printf("Primary is shutting down replica [%d]\n",rank);
      MPI_Send(0,
	            0,
               MPI_INT,
               rank,		// shutdown this particular node
               SHUTDOWNTAG,		// tell it to shutdown
	       MPI_COMM_WORLD);
   }

   printf("got to the end of Primary code\n");

   // Save & Deleta all regressions
   SaveRegressions(outputfilename,regressions);
   DeleteAllRegressions(regressions);

   // Free memory
   gsl_matrix_free(betas);
   delete regressions; regressions = NULL;

   return;
  
}

void replica(int replicaname) {
   int work[1];			// the input from primary
   double output[5];	// the output for primary
   MPI_Status status;		// for MPI communication

   // Initialize random number generator
   const gsl_rng_type* T;
   gsl_rng* r;
   gsl_rng_env_setup();
   T = gsl_rng_default;
   r = gsl_rng_alloc(T);
   
   // Set seed based on replica name
   gsl_rng_set(r, replicaname);

   // the replica listens for instructions...
   int notDone = 1;
   while(notDone)
   {
      printf("Replica %d is waiting\n",replicaname);
      MPI_Recv(&work, // the input from primary
	            1,		// the size of the input
	            MPI_INT,		// the type of the input
               0,		// from the PRIMARY node (rank=0)
               MPI_ANY_TAG,	// any type of order is fine
               MPI_COMM_WORLD,
               &status);
      printf("Replica %d just received smth\n",replicaname);

      // switch on the type of work request
      switch(status.MPI_TAG)
      {
         case GETR2:
            // Run the Bayesian logistic regression for this variable
            // ...and save it in the results vector

           printf("Replica %d has received work request [%d]\n",
                  replicaname,work[0]);
          
	        bayesLogistic(work[0], r, output);

            // Send the results
            MPI_Send(&output,
                     5,
                     MPI_DOUBLE,
                     0,		// send it to primary
                     0,		// doesn't need a TAG
                     MPI_COMM_WORLD);

            printf("Replica %d finished processing work request [%d]\n",
                   replicaname,work[0]);

            break;

         case SHUTDOWNTAG:
            printf("Replica %d was told to shutdown\n",replicaname);
            return;

         default:
            notDone = 0;
            printf("The replica code should never get here.\n");
            return;
      }
   }

   // Free memory
   gsl_rng_free(r);

   // Return to main function
   return;
}

void bayesLogistic(int index, gsl_rng* mystream, double* out) {

   int i;
   double laplace_likelihood;
   double mc_likelihood;

   // Initialization
   gsl_matrix* x = gsl_matrix_alloc(n, 1);
   for(i=0;i<n;i++) {
      gsl_matrix_set(x, i, 0, gsl_matrix_get(data, i, index));
   }

   // Newton-Raphson for beta
   gsl_matrix* betaMode = getcoefNR(n, y, x);

   // Calculate posterior means for betas
   gsl_matrix* sapmle_means = getPosteriorMeans(mystream, n, y, x, betaMode);

   // Laplace approximation
   laplace_likelihood = getLaplaceApprox(n, y, x, betaMode);

   // MC
   mc_likelihood = MC_likelihood(mystream, n, y, x);

   // Update output
   out[0] = (double)(index+1);
   out[1] = mc_likelihood;
   out[2] = laplace_likelihood;
   out[3] = gsl_matrix_get(sapmle_means, 0, 0);
   out[4] = gsl_matrix_get(sapmle_means, 1, 0);

    // Free memory
   gsl_matrix_free(x);
   gsl_matrix_free(betaMode);
   gsl_matrix_free(sapmle_means);
}