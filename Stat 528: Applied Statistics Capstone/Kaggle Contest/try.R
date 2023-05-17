setwd("/Users/dongyangwang/Desktop/UW/Stat 528/Kaggle")
MissingData=read.csv('Data.csv')
#IsMissingMatrix=is.na(MissingData)

df = MissingData
res = matrix(rep(0,length(IsMissingMatrix)),ncol=length(MissingData),byrow=TRUE)

# mean
for (i in 1:length(df[1,])){

  res[,i][is.na(df[,i])]<-mean(df[,i], na.rm = T)
  
}

# dist
for (i in 1:length(df[1,])){
  
  n = length(res[,i][is.na(df[,i])])
  std = sd(df[,i], na.rm = T)
  ran = rnorm(n, 0, std)
  res[,i][is.na(df[,i])]<-ran
  
}

# max row
for (i in 1:length(df[,1])){
  
  res[i,][is.na(df[i,])]<-max(df[i,], na.rm = T)
  
  
}

# package
library(mice)
md.pattern(df)
tempData <- mice(df,m=50,maxit=10,meth='pmm',seed=42)
res = as.matrix(complete(tempData))
ImputedValues=as.vector(res)[as.vector(IsMissingMatrix)]

# another package
library(Amelia)
amelia_fit <- amelia(df, m=15, 
                     tolerance = 1e-05, parallel = "multicore")
CompleteData=as.matrix(amelia_fit$imputations[[10]])
ImputedValues=as.vector(CompleteData)[as.vector(IsMissingMatrix)]
output=data.frame(Id=c(1:length(ImputedValues)), Predicted=ImputedValues)
write.csv(output,'amelia.csv',row.names = FALSE)


#CompleteData = df

#ImputedValues=as.vector(CompleteData)[as.vector(IsMissingMatrix)]
ImputedValues = as.vector(res[res != 0])
output=data.frame(Id=c(1:length(ImputedValues)), Predicted=ImputedValues)
write.csv(output,'output_exceptional.csv',row.names = FALSE)
