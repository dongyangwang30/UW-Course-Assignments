#rm(list=ls())
#setwd("/Users/dongyangwang/Desktop/UW/Stat 534/HW/HW 3")

################### Necessary Functions ###################

#load the 'RCDD' package
library(rcdd);

#this is the version of the 'isValidLogistic' function
#based on Charles Geyers RCDD package
#returns TRUE if the calculated MLEs can be trusted
#returns FALSE otherwise
isValidLogisticRCDD <- function(response,explanatory,data)
{
  if(0==length(explanatory))
  {
    #we assume that the empty logistic regression is valid
    return(TRUE);
  }
  logisticreg = suppressWarnings(glm(data[,response] ~ 
                                    as.matrix(data[,as.numeric(explanatory)]),
                                    family=binomial(link=logit),x=TRUE));
  tanv = logisticreg$x;
  tanv[data[,response] == 1, ] <- (-tanv[data[,response] == 1, ]);
  vrep = cbind(0, 0, tanv);
  #with exact arithmetic; takes a long time
  #lout = linearity(d2q(vrep), rep = "V");
  
  lout = linearity(vrep, rep = "V");
  return(length(lout)==nrow(data));
}

# compute the AIC for a single model.
getLogisticAIC <- function(y, vars, data)
{
  # check if the regression has no explanatory variables
  if(0 == length(vars))
  {
    out = glm(data[,y] ~ 1, family = binomial);
  }
  
  # regression with at least one explanatory variable. We suppress
  # convergence warnings to reduce screen clutter. 
  else
  {
    out = suppressWarnings(glm(data[,y] ~ as.matrix(data[,as.numeric(vars)]),
                               family = binomial));
  }
  
  # compute the AIC
  AIC = out$deviance+2*(1+length(vars))
  
  # we must check whether the glm algorithm properly converged,
  # as judged by the algorithm.
  
  converged = out$converged
  
  # when glm fails to converge return NA, otherwise return AIC
  return(ifelse(converged, AIC, NA))
}

################### Helper Functions ################### 

# Generates a sample
generate_sample <- function(explanatory = 1:60, k =10){
  return (sample(explanatory, k))
}

# Create the Union of add_1 element cases and delete_1 element cases
add_1_or_del_1 <- function(explanatory = 1:60, cur_pred){
  res = list()
  #  add_1 element
  set_diff = setdiff(explanatory, cur_pred)
  for (i in 1:length(set_diff)){
    res[[i]] = c(cur_pred, set_diff[i])
  }
  # delete_1 element 
  for (i in 1:length(cur_pred)){
    res[[length(set_diff) + i]] = cur_pred[-i]
  }
  return(res)
}


# Testing
# explanatory = 1:60
# cur_pred = c(1,2,3)
# k = add_1_or_del_1(cur_pred = cur_pred)
# length(k)
# class(unlist(sample(k,1)))
# class(k[[1]])
# for(i in k){
#   print(class(i))
# }
# a = c(1,5,10,60)
# k[[a]] <- NULL
# b = k[-a]
# k
#b

################### QUESTION 1 ###################

MC3search <- function(response, data, n_iter){
  # Iteration 0
  # Generate sample
  explanatory = generate_sample()
  
  # Check if logistic regression is valid, update if not
  while (isValidLogisticRCDD(response, explanatory, data) == FALSE){
    explanatory = generate_sample()
  }
  
  # Save best predictors, best predictor AIC, and current predictors
  best_pred = explanatory
  bestAIC = getLogisticAIC(61, best_pred, data)
  cur_pred = explanatory
  
  # Iteration r
  iter = 0
  # While loop as of Step 9: Keep iterating until n_iter is reached
  while (iter < n_iter){
    # Step 1
    nbh_a = add_1_or_del_1(explanatory = 1:60, cur_pred = cur_pred)
    
    # Step 2
    index1 = c()
    for (i in 1:length(nbh_a)){
      if (isValidLogisticRCDD(response, nbh_a[[i]], data) == FALSE){
        # Record the set of predictors when logistic regression invalid
        index1 = c(index1, i)
      }
    }
    # Debugging code
    # print(index1)
    
    # Delete the set of predictors when logistic regression invalid, if any
    if (!is.null(index1)){
      nbh_a = nbh_a[-index1]
    }
    
    # Step 3
    new_pred = unlist(sample(nbh_a, 1))
    
    # Step 4
    index2 = c()
    nbh_a2 = add_1_or_del_1(explanatory = 1:60, cur_pred = new_pred)
    for (i in 1:length(nbh_a2)){
      if (isValidLogisticRCDD(response, nbh_a2[[i]], data) == FALSE){
        # Record the set of predictors when logistic regression invalid
        index2 = c(index2, i)
      }
    }
    # Delete the set of predictors when logistic regression invalid, if any
    if (!is.null(index2)){
      nbh_a2 = nbh_a2[-index2]
    }
    
    # Step 5
    new_aic = getLogisticAIC(61, new_pred, data)
    p_new = -new_aic - log(length(nbh_a2))
    
    # Step 6
    old_aic = getLogisticAIC(61, cur_pred, data) 
    p_old = -old_aic- log(length(nbh_a))
    
    # Step 7
    if (p_new > p_old){
      # Update current model
      cur_pred = new_pred
      if (getLogisticAIC(61, cur_pred, data) < bestAIC){
        # Update best model and AIC
        best_pred = cur_pred
        bestAIC = getLogisticAIC(61, best_pred, data)
      }
    } # Step 8
    else {
      u <- runif(1)
      if (log(u) < p_new - p_old){
        # Update current model
        cur_pred = new_pred
        if (getLogisticAIC(61, cur_pred, data) < bestAIC){
          # Update best model and AIC
          best_pred = cur_pred
          bestAIC = getLogisticAIC(61, best_pred, data)
        }
      }
    }
    # Step 9: Increasing index to keep iterating until n_iter times
    iter = iter + 1
  }
  return(list(bestAIC, bestAICvars = sort(best_pred)))
}

################### QUESTION 2 ###################

# df_small <- read.table("534binarydatasmall.txt", header = F)
# df <- read.table("534binarydata.txt", header = F)
# set.seed(42)
set.seed(40)
MC3search(11,df_small,25)
# Testing code
# res = MC3search(61,df,1)
# saved <- replicate(10,MC3search(61,df,1))

# To solve the problem, run the following code
# res <- replicate(10,MC3search(61,df,25))
# for (i in 1:length(res)){
#  print(sort(res[[i]]))
# }

# Printed result for convenience
#[1] 99.12352
#[1]  1  8 10 11 18 24 25 28 31 32 36 37 45 47 48 49 58 59 60
#[1] 58.54345
#[1]  3  4  7 13 14 23 26 35 42 45 46 47 49 50 53 55 56 59
#[1] 96.95143
#[1]  8  9 18 21 34 36 39 40 41 42 46 48 53 54 55 58
#[1] 68.75807
#[1]  1  3  5  6 13 14 17 18 19 32 37 39 41 43 46 47 51 54 55 59 60
#[1] 93.18668
#[1]  5 10 11 13 16 17 19 21 25 27 28 37 39 40 41 44 46 49 51 54 59 60
#[1] 80.35443
#[1]  1  3  6  8 12 13 15 16 28 31 33 37 41 43 48 54 55 59
#[1] 70.68695
#[1]  1  8 11 12 14 15 17 21 25 32 34 35 37 45 47 54 58 59 60
#[1] 82.56365
#[1]  6  8 12 14 16 19 20 21 22 25 29 30 33 40 42 44 46 48 49 54
#[1] 84.43894
#[1]  3  5  7 12 13 15 22 28 31 32 36 38 39 42 47
#[1] 88.17002
#[1]  1  4  6 10 14 17 18 19 30 32 37 38 39 45 46 47 50 51 54 58

# Based on my result, a few things I'd like to comment. First, it takes nearly
# an hour to execute the above code. It's either worth considering calling each
# replication individually for better management of time/result, or decrease the
# number of n_iter, or improve the code in some manner.

# Second, my AIC values range from 66 to 99. And the best models have 15-20 variables
# selected. One caveat is that the starting point (number of variables to include 
# in the first place) matter a lot. If with different numbers, such as 20 or even 50, 
# the result could be drastically different. Also note that the optimal solution
# can be unachievable because we only repeat 25 times, and if the optimal solution
# contains 50 variables, we wouldn't be able to get there.

# Third, since there are a lot of random generation/sampling process happening, it
# is preferably that we do the iterations as many times as we can so we can
# reach the steady state. But with limited times, we might not actually get there
# possibly simply due to bad luck in the generation processes.