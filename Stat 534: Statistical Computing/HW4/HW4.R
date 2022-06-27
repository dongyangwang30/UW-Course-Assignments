#rm(list=ls())
#setwd("/Users/dongyangwang/Desktop/UW/Stat 534/HW/HW 4")

################### Necessary Functions ###################

# From HW1 and Lectures

#log determinant
logdet <- function(R)
{
  return(sum(log(eigen(R)$values)))	
}

#this function uses 'glm' to fit a logistic regression
#and obtain the MLEs of the two coefficients beta0 and beta1
getcoefglm <- function(response,explanatory,data)
{
  return(coef(glm(data[,response] ~ data[,explanatory],family=binomial(link=logit))));
}

#the inverse of the logit function
inverseLogit <- function(x)
{
  return(exp(x)/(1+exp(x))); 
}

#function for the computation of the Hessian
inverseLogit2 <- function(x)
{
  return(exp(x)/(1+exp(x))^2); 
}

#computes pi_i = P(y_i = 1 | x_i)
getPi <- function(x,beta)
{
  x0 = cbind(rep(1,length(x)),x);
  return(inverseLogit(x0%*%beta));
}

#another function for the computation of the Hessian
getPi2 <- function(x,beta)
{
  x0 = cbind(rep(1,length(x)),x);
  return(inverseLogit2(x0%*%beta));
}

#logistic log-likelihood (formula (3) in your handout)
logisticLoglik <- function(y,x,beta)
{
  Pi = getPi(x,beta);
  return(sum(y*log(Pi))+sum((1-y)*log(1-Pi)));
}

#logistic log-likelihood for l*
logisticLoglik1 <- function(y,x,beta)
{
  logisticLoglik1 = -log(2*pi) - 1/2 * ((beta[1])^2 + (beta[2])^2) + logisticLoglik(y,x,beta)
  return(logisticLoglik1);
}

#obtain the gradient for Newton-Raphson
getGradient <- function(y,x,beta)
{
  gradient = matrix(0,2,1);
  Pi = getPi(x,beta);
  
  gradient[1,1] = sum(y-Pi);
  gradient[2,1] = sum((y-Pi)*x);
  
  return(gradient);
}

#obtain the Hessian for Newton-Raphson
getHessian <- function(y,x,beta)
{
  hessian = matrix(0,2,2);
  Pi2 = getPi2(x,beta);
  
  hessian[1,1] = sum(Pi2);
  hessian[1,2] = sum(Pi2*x);
  hessian[2,1] = hessian[1,2];
  hessian[2,2] = sum(Pi2*x^2);
  
  return(-hessian);
}

#obtain the Hessian UPDATED
getHessian1 <- function(y,x,beta)
{
  hessian = matrix(0,2,2);
  Pi2 = getPi2(x,beta);
  
  hessian[1,1] = sum(Pi2) + 1;
  hessian[1,2] = sum(Pi2*x);
  hessian[2,1] = hessian[1,2];
  hessian[2,2] = sum(Pi2*x^2) + 1;
  
  return(-hessian);
}
#this function implements our own Newton-Raphson procedure
getcoefNR <- function(response,explanatory,data)
{
  #2x1 matrix of coefficients`
  beta = matrix(0,2,1);
  y = data[,response];
  x = data[,explanatory];
  
  #current value of log-likelihood
  currentLoglik = logisticLoglik(y,x,beta);
  
  #infinite loop unless we stop it someplace inside
  while(1)
  {
    newBeta = beta - solve(getHessian1(y,x,beta))%*%getGradient(y,x,beta);
    newLoglik = logisticLoglik(y,x,newBeta);
    
    #at each iteration the log-likelihood must increase
    if(newLoglik<currentLoglik)
    {
      cat("CODING ERROR!!\n");
      break;
    }
    beta = newBeta;
    #stop if the log-likelihood does not improve by too much
    if(newLoglik-currentLoglik<1e-6)
    {
      break; 
    }
    currentLoglik = newLoglik;
  }
  
  return(beta);
}

################### QUESTION 1 ###################

# Test Running
# First find beta with NR
# betaMode = getcoefNR(61,1,df)
# y = df[,61]
# x = df[,1]
# logisticLoglik1(y,x,betaMode)
# getLaplaceApprox(61,1,df,betaMode = betaMode)

getLaplaceApprox <- function(response, explanatory, data, betaMode){
  # Calculate l*
  likelihood = logisticLoglik1(data[,response], data[,explanatory], betaMode)
  
  # Calculate the Hessian for l*
  hessian = getHessian1(data[,response], data[,explanatory], betaMode)
  
  # Calculate the log marginal likelihood
  log_pd = log(2*pi) + likelihood -1/2 * logdet(-hessian)
  
  return(log_pd)
}

################### QUESTION 2 ###################
library(MASS)

getPosteriorMeans <- function(response,explanatory,data, betaMode, niter){
  # Generate a storage vector of size 2, and a sum vector for each beta
  vec = c(0,0)
  sum_vec = c(0,0)
  
  # Initialize the beta's
  vec = betaMode
  
  # Initialize mean and variance for multivariate normal
  hessian = getHessian1(data[,response], data[,explanatory], betaMode)
  sigma = -solve(hessian)
  
  # Initialize a likelihood vector to store the likelihood.
  l_init = logisticLoglik1(data[,response], data[,explanatory], vec)
  l_old = l_init
  
  # Repeat niter times for updating the beta's
  for (i in 2:niter){
    # Checking: print(i)
    
    # Produce the sample result
    res = mvrnorm(mu = vec,Sigma = sigma)
    l_new = logisticLoglik1(data[,response], data[,explanatory], res)
    
    # Compare the l* so to determine whether to update the markov chain
    if (l_new >= l_old){
      l_old = l_new
      vec = res
    }else{
      lsample = log(runif(1))
      if (lsample <= l_new - l_old){
        l_old = l_new
        vec = res
      }
    }
    sum_vec[1] = sum_vec[1] + vec[1]
    sum_vec[2] = sum_vec[2] + vec[2]
    #print(vec)
  }
  return(beta = sum_vec/niter)
}

# Test Running
# getPosteriorMeans(61,1,df,betaMode,20000)

################### QUESTION 3 ###################

# df_small <- read.table("534binarydatasmall.txt", header = F)
df <- read.table("534binarydata.txt", header = F)

# Test Running
# bayesLogistic(1,61,df,2500)

bayesLogistic = function(apredictor,response,data,NumberOfIterations)
{
  # Get initial beta's
  betaMode = getcoefNR(response,apredictor,data)
  
  # Apply MC3 to get posterior means
  PosteriorMeans = c(getPosteriorMeans(response,apredictor,data,betaMode,NumberOfIterations)[1],
                     getPosteriorMeans(response,apredictor,data,betaMode,NumberOfIterations)[2])
  
  # Calculate logmarglik
  # SHOULD WE USE PosteriorMeans?????
  logmarglik = getLaplaceApprox(response,apredictor,data,betaMode = betaMode)
  
  # Calculate MLEs
  mle = getcoefglm(response,apredictor, data)
  return(list(apredictor = apredictor, logmarglik = logmarglik, 
         beta0bayes = PosteriorMeans[1], beta1bayes = PosteriorMeans[2],
         beta0mle = mle[1], beta1mle = mle[2]))
}

#PARALLEL VERSION
#datafile = the name of the file with the data
#NumberOfIterations = number of iterations of the Metropolis-Hastings algorithm
#clusterSize = number of separate processes; each process performs one or more
#univariate regressions
main <- function(datafile,NumberOfIterations,clusterSize)
{
  #read the data
  data = read.table(datafile,header=FALSE);
  
  #the sample size is 148 (number of rows)
  #the explanatory variables are the first 60 columns for '534binarydata.txt'
  #the last column is the binary response
  response = ncol(data);
  lastPredictor = ncol(data)-1;
 
  #initialize a cluster for parallel computing
  cluster <- makeCluster(clusterSize, type = "SOCK")
  
  # import function to the cluster
  clusterExport(cluster, c("bayesLogistic","getcoefglm","getcoefNR",
                           "getGradient","getHessian","getHessian1",
                           "getLaplaceApprox","getPi","getPi2","getPosteriorMeans",
                           "inverseLogit","inverseLogit2","logdet","logisticLoglik",
                           "logisticLoglik1","main","mvrnorm"))
  
  #run the MC3 algorithm from several times
  results = clusterApply(cluster, 1:lastPredictor, bayesLogistic,
                         response,data,NumberOfIterations);
  
  #print out the results
  for(i in 1:lastPredictor)
  {
    cat('Regression of Y on explanatory variable ',results[[i]]$apredictor,
        ' has log marginal likelihood ',results[[i]]$logmarglik,
        ' with beta0 = ',results[[i]]$beta0bayes,' (',results[[i]]$beta0mle,')',
        ' and beta1 = ',results[[i]]$beta1bayes,' (',results[[i]]$beta1mle,')',
        '\n');    
  }
  
  #destroy the cluster
  stopCluster(cluster);  
}

#NOTE: YOU NEED THE PACKAGE 'SNOW' FOR PARALLEL COMPUTING
require(snow);

#this is where the program starts
main('534binarydata.txt',10000,10);
