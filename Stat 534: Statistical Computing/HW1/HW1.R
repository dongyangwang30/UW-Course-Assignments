############################# HW 1 #############################
rm(list = ls())
getwd()
setwd('/Users/dongyangwang/Desktop/UW/Stat 534/HW/HW 1')

####################### Problem 1 #######################

# Playing around with matrices
?eigen
sample <- matrix(c(2,-1,0,-1,2,-1,0,-1,2), nrow = 3)
sample1 <- matrix(c(2,-1,0,-1,2,-1,0,-1,3), nrow = 3)
sample1 + sample

log(det(sample))
k <- eigen(sample)
j <- k[[1]][1]
log(k[[1]])
sum(k[[1]])
class(j)

# The logdet function: log determinant is the sum of log eigenvalues
logdet <- function(R){
  return (sum(log(eigen(R)[[1]])))
}

# Sanity check: On PD matrix, correct answer can be computed
logdet(sample)

####################### Problem 2 #######################

df <- read.table('erdata.txt', header = F)

logmarglik <- function(df, vector){
  # preparation
  # Test Using: vector = c(2,5,10)
  a = length(vector)
  n = nrow(df)
  DA <- as.matrix(df[,vector])
  D1 <- as.matrix(df[1])
  MA <- diag(a) + t(DA) %*% DA
  
  # part 1: Calculate using lgamma
  part1 <- lgamma((n+a+2)/2)-lgamma((a+2)/2)

  # part 2: Calculate using logdet
  part2 <- -logdet(MA)/2

  # part 3: Matrix multiplication
  part31 <- as.numeric(t(D1) %*% D1)
  part32 <- as.numeric(t(D1) %*% DA %*% solve(MA) %*% t(DA) %*% D1)
  part3 <- -(n+a+2)*log(1 + part31- part32)/2
  
  print(part2)
  print(part3)
  
  return(part1 + part2 + part3)
}

c1 <- c(2,5,10)
logmarglik(df, c1)

# Obtained the log marginal likelihood: -59.97893