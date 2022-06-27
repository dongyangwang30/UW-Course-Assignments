#HOMEWORK 2, PROBLEM 1
#this function uses 'glm' to fit a logistic regression
#and returns the AIC = deviance + 2*NumberOfCoefficients 
getLogisticAIC <- function(response,explanatory,data)
{
  #check if the regression has no explanatory variables
  if(0==length(explanatory))
  {
    #regression with no explanatory variables
    deviance = glm(data[,response] ~ 1,family=binomial(link=logit))$deviance;
  }
  else
  {
    #regression with at least one explanatory variable
    deviance = glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit))$deviance;
  }
  return(deviance+2*(1+length(explanatory)));
}

# Checking by hand
# c = c()
# for (i in 2:10){
#  c = c(c,getLogisticAIC(11,c(1,i),df_small))
# }
# min(c)
# c

#HOMEWORK 2, PROBLEM 2: Forward greedy search
forwardSearchAIC <- function(response,data,lastPredictor)
{
  
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the AIC of the empty regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearch :: The empty logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor;
  
  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of AIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller AIC than the AIC of the current regression
  
  #stop when there are no variables that can be included in the model
  #stepNumber = 0;
  while(length(VariablesNotInModel)>=1)
  {
     #record the number of steps performed
     #stepNumber = stepNumber + 1;
    
     #create a vector that records the AIC values of the regressions
     #we are examining; the number of these regressions is equal
     #with the number of variables that are not in the model
     regAIC = c();
    
     #take each variable that is not in the model
     #and include it in the model
     for (i in VariablesNotInModel){
       v1 = c(bestRegression,i)
       regAIC = c(regAIC, getLogisticAIC(response,v1,data))
     }
     
     currentAIC = min(regAIC)
     index = which(regAIC == currentAIC)
     currentRegression = c(bestRegression,VariablesNotInModel[index])
     
     if(currentAIC<bestRegressionAIC)
     {
       bestRegressionAIC = currentAIC
       bestRegression = currentRegression
       VariablesNotInModel = VariablesNotInModel[-index]
     }
     else
     {
       break
     }
  }
  
  return(list(aic=bestRegressionAIC,reg=sort(bestRegression)));
}


##HOMEWORK 2, PROBLEM 3: Backward greedy search
backwardSearchAIC <- function(response,data,lastPredictor)
{
  #start with the full regression that includes all the variables
  bestRegression = 1:lastPredictor;
  #calculate the AIC of the full regression
  bestRegressionAIC = getLogisticAIC(response,bestRegression,data);
  cat('\n\n\n\nbackwardSearch :: The full logistic regression has AIC = ',bestRegressionAIC,'\n');
  
  #sequentially delete one variable from the current regression
  #and retain that variable that gives the smallest AIC; make the model
  #in which that variable is deleted to be the current model if
  #this leads to a current model with a smaller AIC
  #stepNumber = 0;
  while(length(bestRegression)>=1)
  {
    #record the number of steps performed
    #stepNumber = stepNumber + 1;
    
    #create a vector that records the AIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regAIC = c();
    
    #take each variable that is not in the model
    #and include it in the model
    for (i in 1:length(bestRegression)){
      v1 = bestRegression[-i]
      regAIC = c(regAIC, getLogisticAIC(response,v1,data))
    }
    
    currentAIC = min(regAIC)
    index = which(regAIC == currentAIC)
    currentRegression = bestRegression[-index]
    
    if(currentAIC<bestRegressionAIC)
    {
      bestRegressionAIC = currentAIC
      bestRegression = currentRegression
    }
    else
    {
      break
    }
  }
  
  return(list(aic=bestRegressionAIC,reg=bestRegression));
}

##HOMEWORK 2, PROBLEM 4: Implementation

#df_small <- read.table("534binarydatasmall.txt", header = F)
#df <- read.table("534binarydata.txt", header = F)

# Comments: On the small dataset, the models are the same with variables selected
# being (1  2  3  5  6  7  8  9 10),. Both models have 111.2943 as the AIC. On the large data set, 
# the forward function selects (3  4  20 21  22  23  34  46  49  53) with AIC 22 
# but the backward function selects (1  3  9 12 20 23 25 34 41 46) with AIC 22
# AIC are the same

# Later following this code are a repitition for BIC. The results are
# On the small dataset, the models are the same with variables selected
# being (1  2  3  7  8  9 10),. Both models have 137.1848 as the BIC On the large data set, 
# the forward function selects (3  4  20 21  22  23  34  46  49  53) with BIC 54.96934 
# but the backward function selects (1  3  9 12 20 23 25 34 41 46) with BIC 54.96934

#forward AIC
# forwardSearchAIC(11, df_small, 10)
forwardSearchAIC(61, df, 60)

#backward AIC
# backwardSearchAIC(11, df_small, 10)
backwardSearchAIC(61, df, 60)

#HOMEWORK 2, PROBLEM 4.1: BIC
getLogisticBIC <- function(response,explanatory,data)
{
  #check if the regression has no explanatory variables
  if(0==length(explanatory))
  {
    #regression with no explanatory variables
    deviance = glm(data[,response] ~ 1,family=binomial(link=logit))$deviance;
  }
  else
  {
    #regression with at least one explanatory variable
    deviance = glm(data[,response] ~ as.matrix(data[,as.numeric(explanatory)]),family=binomial(link=logit))$deviance;
  }
  return(deviance+log(nrow(data))*(1+length(explanatory)));
}

#HOMEWORK 2, PROBLEM 4.2: Forward greedy search
forwardSearchBIC <- function(response,data,lastPredictor)
{
  
  #start with the empty regression with no predictors
  bestRegression = NULL;
  #calculate the BIC of the empty regression
  bestRegressionBIC = getLogisticBIC(response,bestRegression,data);
  cat('\n\n\n\nforwardSearch :: The empty logistic regression has BIC = ',bestRegressionBIC,'\n');
  
  #vector that keeps track of all the variables
  #that are currently NOT in the model
  VariablesNotInModel = 1:lastPredictor;
  
  #add variables one at a time to the current regression
  #and retain that variable that gives the smallest values of BIC associated
  #Make the model that includes that variable to be the current model
  #if it gives a smaller BIC than the BIC of the current regression
  
  #stop when there are no variables that can be included in the model
  #stepNumber = 0;
  while(length(VariablesNotInModel)>=1)
  {
    #record the number of steps performed
    #stepNumber = stepNumber + 1;
    
    #create a vector that records the BIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regBIC = c();
    
    #take each variable that is not in the model
    #and include it in the model
    for (i in VariablesNotInModel){
      v1 = c(bestRegression,i)
      regBIC = c(regBIC, getLogisticBIC(response,v1,data))
    }
    
    currentBIC = min(regBIC)
    index = which(regBIC == currentBIC)
    currentRegression = c(bestRegression,VariablesNotInModel[index])
    
    if(currentBIC<bestRegressionBIC)
    {
      bestRegressionBIC = currentBIC
      bestRegression = currentRegression
      VariablesNotInModel = VariablesNotInModel[-index]
    }
    else
    {
      break
    }
  }
  
  return(list(BIC=bestRegressionBIC,reg=sort(bestRegression)));
}


##HOMEWORK 2, PROBLEM 4.3: Backward greedy search
backwardSearchBIC <- function(response,data,lastPredictor)
{
  #start with the full regression that includes all the variables
  bestRegression = 1:lastPredictor;
  #calculate the BIC of the full regression
  bestRegressionBIC = getLogisticBIC(response,bestRegression,data);
  cat('\n\n\n\nbackwardSearch :: The full logistic regression has BIC = ',bestRegressionBIC,'\n');
  
  #sequentially delete one variable from the current regression
  #and retain that variable that gives the smallest BIC; make the model
  #in which that variable is deleted to be the current model if
  #this leads to a current model with a smaller BIC
  #stepNumber = 0;
  while(length(bestRegression)>=1)
  {
    #record the number of steps performed
    #stepNumber = stepNumber + 1;
    
    #create a vector that records the BIC values of the regressions
    #we are examining; the number of these regressions is equal
    #with the number of variables that are not in the model
    regBIC = c();
    
    #take each variable that is not in the model
    #and include it in the model
    for (i in 1:length(bestRegression)){
      v1 = bestRegression[-i]
      regBIC = c(regBIC, getLogisticBIC(response,v1,data))
    }
    
    currentBIC = min(regBIC)
    index = which(regBIC == currentBIC)
    currentRegression = bestRegression[-index]
    
    if(currentBIC<bestRegressionBIC)
    {
      bestRegressionBIC = currentBIC
      bestRegression = currentRegression
    }
    else
    {
      break
    }
  }
  
  return(list(BIC=bestRegressionBIC,reg=bestRegression));
}


#forward BIC
# forwardSearchBIC(11, df_small, 10)
forwardSearchBIC(61, df, 60)

#backward BIC
# backwardSearchBIC(11, df_small, 10)
backwardSearchBIC(61, df, 60)

