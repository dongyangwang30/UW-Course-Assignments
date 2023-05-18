
rm(list = ls())

##### generic data setup:
set.seed(42)
ni     = 100  # 100 subjects
nj     =  10  # 10 observation for each subject
prop.m = .07  # 7% missingness
beta0 = 0.5
beta1 = 1.5

simu_miss=function(ni,nj){
  x   = rep(rnorm(ni, mean=250, sd=10), each=nj)
  time   = rep(1:nj, times=ni)
  id     = rep(1:ni, each=nj)
  
  e = rnorm(ni*nj, mean=0, sd=1)
  b = rep(rnorm(ni, mean = 0, sd = 1),nj)
  y      = beta0 + beta1 * x + b + e
  
  # MCAR
  mcar   = runif(ni*nj, min=0, max=1)
  y.mcar = ifelse(mcar<prop.m, NA, y)  # unrelated to anything
  
  # MAR
  y.mar = matrix(y, ncol=nj, nrow=ni, byrow=TRUE)
  for(i in 1:ni){
    for(j in 4:nj){
      dif1 = y.mar[i,j-2]-y.mar[i,j-3]
      dif2 = y.mar[i,j-1]-y.mar[i,j-2]
      if(dif1>0 & dif2>0){  # if weight goes up twice, drops out
        y.mar[i,j:nj] = NA;  break
      }
    }
  }
  y.mar = as.vector(t(y.mar))
  
  # NMAR
  sort.y = sort(y, decreasing=TRUE)
  nmar   = sort.y[ceiling(prop.m*length(y))]
  y.nmar = ifelse(y>nmar, NA, y)  # doesn't show up when heavier
  return(data.frame(id, time, x, y, y.mcar,y.mar,y.nmar))
}

library(geepack)
library(lme4)

gen.one=function(ni,nj, case = 1){
  df=simu_miss(ni,nj)
  
  #if (case == 1){
  #  df = na.omit(df)
  #}
  
  if (case == 2){
    df[is.na(df$y.mcar), "y.mcar"] = mean(df$y.mcar, na.rm = T)
    df[is.na(df$y.mar), "y.mar"] = mean(df$y.mar, na.rm = T)
    df[is.na(df$y.nmar), "y.nmar"] = mean(df$y.nmar, na.rm = T)
  }
  
  # lmm
  lmm1 = lmer(y.mcar ~ x + (1|id), data = df, REML = T)
  lmm2 = lmer(y.mar ~ x + (1|id), data = df, REML = T)
  lmm3 = lmer(y.nmar ~ x + (1|id), data = df, REML = T)
  
  # gee
  gee1 = geeglm(y.mcar ~ x, id = id, data = df, corstr = "exchangeable")
  gee2 = geeglm(y.mar ~ x, id = id, data = df, corstr = "exchangeable")
  gee3 = geeglm(y.nmar ~ x, id = id, data = df, corstr = "exchangeable")
  
  # Estimate variance for efficiency
  lmm1_var0 = vcov(lmm1)[1,1]
  lmm1_var1 = vcov(lmm1)[2,2]
  lmm2_var0 = vcov(lmm2)[1,1]
  lmm2_var1 = vcov(lmm2)[2,2]
  lmm3_var0 = vcov(lmm3)[1,1]
  lmm3_var1 = vcov(lmm3)[2,2]
  
  gee1_var0 = vcov(gee1)[1,1]
  gee1_var1 = vcov(gee1)[2,2]
  gee2_var0 = vcov(gee2)[1,1]
  gee2_var1 = vcov(gee2)[2,2]
  gee3_var0 = vcov(gee3)[1,1]
  gee3_var1 = vcov(gee3)[2,2]
  
  # Estimate bias
  lmm1_bias0 = fixef(lmm1)[1] - beta0
  lmm1_bias1 = fixef(lmm1)[1] - beta1
  lmm2_bias0 = fixef(lmm2)[1] - beta0
  lmm2_bias1 = fixef(lmm2)[1] - beta1
  lmm3_bias0 = fixef(lmm3)[1] - beta0
  lmm3_bias1 = fixef(lmm3)[1] - beta1
  
  gee1_bias0 = coef(gee1)[1] - beta0
  gee1_bias1 = coef(gee1)[2] - beta1
  gee2_bias0 = coef(gee2)[1] - beta0
  gee2_bias1 = coef(gee2)[2] - beta1
  gee3_bias0 = coef(gee3)[1] - beta0
  gee3_bias1 = coef(gee3)[2] - beta1
  
  return(data.frame( lmm1_var0, lmm1_var1, lmm2_var0, lmm2_var1,
                     lmm3_var0, lmm3_var1, gee1_var0, gee1_var1,
                     gee2_var0, gee2_var1, gee3_var0, gee3_var1,
                     lmm1_bias0, lmm1_bias1, lmm2_bias0, lmm2_bias1,
                     lmm3_bias0, lmm3_bias1, gee1_bias0, gee1_bias1,
                     gee2_bias0, gee2_bias1, gee3_bias0, gee3_bias1
  ) )
}

nrep = 100

res <- do.call(rbind, lapply(c(1:nrep), function(nrep){
  gen.one(ni,nj, case = 2)
}))
mean_res <- colMeans(res)

simulation_res = as.data.frame(mean_res)
simulation_res

category = c(rep(c("lmm", "gee"), 2, each = 6))
number = c(rep(1:3, 4, each = 2))
metric = c(rep(c("variance", "bias"), each = 12))
param = c(rep(c("beta0", "beta1"),  12))

simulation_res$category = category
simulation_res$number = number
simulation_res$metric = metric
simulation_res$param = param

simulation_res_var = simulation_res[simulation_res$metric == "variance",]
simulation_res_bias = simulation_res[simulation_res$metric != "variance",]

library(ggplot2)
ggplot(data=simulation_res_var, aes(x=category, y=abs(mean_res), color = as.factor(param)))+geom_point()+
  facet_grid(cols=vars(number))
ggplot(data=simulation_res_bias, aes(x=category, y=abs(mean_res), color = as.factor(param)))+geom_point()+
  facet_grid(cols=vars(number))
