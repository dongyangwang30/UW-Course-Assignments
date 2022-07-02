rm(list = ls())
getwd()
setwd('/Users/dongyangwang/Desktop/UW/CSE 512/Projects/Proj 1')

df <- read.csv('sunshine.csv')
library(ggplot2)

df$month = factor(df$month, 
                  levels = c("Jan", "Feb", "Mar", "Apr", "May","Jun",
                             "Jul", "Aug", "Sep" ,"Oct" ,"Nov" ,"Dec"))

ggplot(df, aes(x = month, y = sunshine)) +
  geom_point(size = 0.5, alpha =0.5) +
  geom_line(aes(group = city, color = city)) +
  scale_color_brewer(palette = "PuOr") +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_cartesian(ylim = c(0,NA)) +
  ggtitle("Average Hours of Sunshine By Month (1981 - 2010)") +
  xlab("Month") + 
  ylab("Average Hours of Sunshine") 

