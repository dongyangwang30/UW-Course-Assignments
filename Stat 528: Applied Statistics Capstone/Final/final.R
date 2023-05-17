setwd("/Users/dongyangwang/Desktop/UW/Stat 528/Final")
rm(list=ls())
set.seed(42)

############ Q1 ############

n = 100
x1 = rnorm(n, 0, 1)
x2 = rnorm(n, 0, 1)
x3 = rnorm(n, 0, 1)
x4 = rnorm(n, 0, 1)
e = rnorm(n, 0, 1)

x5 = 0.5*x1 +0.5*x2 + e
x6 = 0.5*x2 +0.5*x3 +0.5*x4 + e


n_list = c(50, 100, 200, 400, 800)





############ Q2 ############

df = read.table("SCHIZREP.DAT.txt")
names(df) = c("ID", "Illness", "Week", "Treatment", "Sex")

library(xtable)
xtable(summary(df))

# Clean up
df = df[df$Week %in% c("1", "3", "6", "0"),]

# EDA
hist(df$Illness, main = "Histogram of Illness", ylab = "frequency",xlab = "Illness")

barplot(table(df$Sex), ylab = "frequency", names.arg=c("Female", "Male"), main = "Barplot of Gender")
barplot(table(df$Treatment), ylab = "frequency", names.arg=c("Placebo", "Treatment"), main = "Barplot of Treatment")

boxplot(df$Illness ~ df$Week, ylab = "frequency", main = "Boxplot of Illness vs Week", xlab = "Week")
boxplot(df$Illness ~ df$Treatment, ylab = "frequency", main = "Boxplot of Illness vs Week", xlab = "Week")

library(dplyr)

week_unique = df %>%
  group_by(ID) %>%
  summarize(total = n()) 

barplot(table(week_unique$total), main = "Bar Plot of Number of Weeks Present", xlab = "Number of Weeks", ylab = "count" )

# model
library(lme4)
lmm = lmer(Illness ~ Week * Treatment  + (1|ID), data = df, REML = T)
summary(lmm)

