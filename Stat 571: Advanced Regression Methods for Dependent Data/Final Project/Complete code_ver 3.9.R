################# set up #################

rm(list = ls())
#install.packages('tidyverse')
library(tidyverse)

suicide_data = read.csv("master.csv")
colnames(suicide_data) = c("country",  "year"  ,  "sex"   , "age"   ,  "suicides_no"   ,  "population" ,       
                           "suicides.100k.pop" , "country.year" ,"HDI.for.year"  , "gdp_for_year...."  , "gdp_per_capita...." ,"generation")
# structure of data
summary(suicide_data)
str(suicide_data)

################# feature engineering #################

# age to levels
suicide_data$age <- gsub(" years", "", suicide_data$age)
suicide_data$age <- factor(
  suicide_data$age, 
  levels = c("5-14","15-24","25-34","35-54","55-74","75+"), 
  order = TRUE)

#suicide_data$age = case_when(
#  suicide_data$age == "5-14 years" ~ 1,
#  suicide_data$age == "15-24 years" ~ 2,
#  suicide_data$age == "25-34 years" ~ 3,
#  suicide_data$age == "35-54 years" ~ 4,
#  suicide_data$age == "55-74 years" ~ 5,
#  suicide_data$age == "75+ years" ~ 6)
#suicide_data$age = as.factor(suicide_data$age)

# generation to levels
suicide_data$generation <- factor(
  suicide_data$generation, 
  levels = c("G.I. Generation","Silent","Boomers",
             "Generation X","Millenials","Generation Z"),
  order = TRUE)

# renaming a few vars
names(suicide_data)[names(suicide_data) == 'suicides.100k.pop'] <- 'suicide_rates'
names(suicide_data)[names(suicide_data) == 'country.year'] <- 'country_year'
names(suicide_data)[names(suicide_data) == 'HDI.for.year'] <- 'HDI'
names(suicide_data)[names(suicide_data) == 'gdp_for_year....'] <- 'GDP'
names(suicide_data)[names(suicide_data) == 'gdp_per_capita....'] <- 'GDP_capita'

# dropping a few countries since they have little data
ordercountry <- arrange(count(suicide_data,country), n)  

# keep only countries with more than 100 observations
suicide_data <- suicide_data %>%
  filter(!(country %in% head(ordercountry$country, 11)))

################# visualization #################

# attach data
attach(suicide_data)

############ distribution of key vars ############

# year
ggplot(suicide_data, aes(x=year)) + 
  geom_bar() +
  labs(title="Distribution of Year", x = "Year")

# age
ggplot(suicide_data, aes(x=age)) + 
  geom_bar()

# country
ggplot(suicide_data, aes(x=country)) + 
  geom_bar() +
  labs(title="Distribution of Country", x = "Country")

# suicide rates
ggplot(suicide_data, aes(x=suicide_rates)) + 
  geom_histogram() +
  labs(title="Distribution of suicide rates", x = "Suicide Rates")

# suicide number
ggplot(suicide_data, aes(x=suicides_no)) + 
  geom_histogram() +
  labs(title="Distribution of suicide number", x = "Suicide Number")

# generation
ggplot(suicide_data, aes(x=generation)) + 
  geom_bar()

############ distribution of response by category ############

# by sex
ggplot(suicide_data, aes(x=sex, y=suicide_rates)) + 
  geom_boxplot() +
  labs(title="Suicide Rates by Sex", x = "Sex", y = "Suicide Rates")

# by age
ggplot(suicide_data, aes(x=age, y=suicide_rates)) + 
  geom_boxplot() +
  labs(title="Suicide Rates by Age", x = "Age", y = "Suicide Rates")


########### Highest/lowest suicide rates on average by country ########### 

summary <- suicide_data %>%
  group_by(country) %>%
  summarize(avg_rate = mean(suicide_rates)) %>%
  arrange(desc(avg_rate))

highest <- summary %>%
  head(n = 4)
lowest <- summary %>%
  tail(n <- 4)

########### Trends over time by country ########### 

us <- suicide_data %>%
  filter(country == "United States")
canada <- suicide_data %>%
  filter(country == "Canada")

ggplot(us, aes(x = year, y = suicide_rates, color = age,
               shape = sex)) +
  geom_point()
ggplot(canada, aes(x = year, y = suicide_rates, color = age,
               shape = sex)) +
  geom_point()

########### Trends over time for highest rate countries ########### 

lithuania <- suicide_data %>%
  filter(country == "Lithuania")
sri_lanka <- suicide_data %>%
  filter(country == "Sri Lanka")
russia <- suicide_data %>%
  filter(country == "Russian Federation")
hungary <- suicide_data %>%
  filter(country == "Hungary")

ggplot(lithuania, aes(x = year, y = suicide_rates, color = age,
               shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)
ggplot(sri_lanka, aes(x = year, y = suicide_rates, color = age,
                   shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)
ggplot(russia, aes(x = year, y = suicide_rates, color = age,
               shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)
ggplot(hungary, aes(x = year, y = suicide_rates, color = age,
                   shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)

########### Trends over time for lowest rate countries ########### 

kuwait <- suicide_data %>%
  filter(country == "Kuwait")
sa <- suicide_data %>%
  filter(country == "South Africa")
aab <- suicide_data %>%
  filter(country == "Antigua and Barbuda")
jamaica <- suicide_data %>%
  filter(country == "Jamaica")

ggplot(kuwait, aes(x = year, y = suicide_rates, color = age,
                      shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)
ggplot(sa, aes(x = year, y = suicide_rates, color = age,
                      shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)
ggplot(aab, aes(x = year, y = suicide_rates, color = age,
                   shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)
ggplot(jamaica, aes(x = year, y = suicide_rates, color = age,
                    shape = sex)) +
  geom_point() +
  geom_smooth(se = FALSE)


# detach data
detach(suicide_data)

################# modeling: continuous #################
# keep only countries with more than 100 observations
suicide_data <- suicide_data %>%
  filter(!(country %in% head(ordercountry$country, 11)))

suicide_data$id <- reorder(suicide_data$country, suicide_data$suicide_rates)

############ cross section ############

# plotting the response var
hist(suicide_data$suicide_rates)

formula = suicide_rates ~ year + sex * age + population + GDP_capita + generation
#formula = suicide_rates ~ year + sex + population + GDP_capita + generation
# linear regression
lm = lm(formula, data = suicide_data)
summary(lm)

# glm: remove 0 to ensure model left with only positive values
# log transformation might also work
suicide_data_pos = suicide_data[suicide_data$suicide_rates != 0,]
glm_gamma_pos = glm(formula, data = suicide_data_pos, family = Gamma(link = "inverse"))
summary(glm_gamma_pos)

# might also run gamma model with zero inflation: but no interaction term
zigamma_y = suicide_data$suicide_rates
zigamma_x = suicide_data %>%
  select(c('year','sex', "age", "population", "GDP_capita", "generation"))
zigamma_x = suicide_data %>%
  select(c('year','sex', "population", "GDP_capita", "generation"))

# install.packages('Rfast2')
#library(Rfast2)
#zigamma = zigamma.reg(zigamma_y, zigamma_x, tol = 1e-07, maxiters = 100) 
#summary(zigamma)
############ longitudinal ############
library(lme4)
library(geepack)
library(gee)
# lmm

lmm = lmer(suicide_rates ~ year + sex + population + GDP_capita + 
             generation + (1|id), data = suicide_data, REML = T)
summary(lmm)

# GEE

gee_indep = geeglm(suicide_rates ~ 
                     year + sex  + population + GDP_capita + generation, 
                   id = id, 
                   data = suicide_data, 
                   corstr = "independence")

summary(gee_indep)


gee_exch = geeglm(suicide_rates ~ 
                    year + sex + population + GDP_capita + generation, 
                  id = id, 
                  data = suicide_data, 
                  corstr = "exchangeable")

summary(gee_exch)

gee_unstr = geeglm(suicide_rates ~ 
               year + sex  + population + GDP_capita + generation, 
             id = id, 
             data = suicide_data, 
             corstr = "unstructured")
summary(gee_unstr)

# can also change family to see performance

################# modeling: binary #################

############ cross section ############

# plotting the response var

formula = suicide_rates ~ year + sex * age + population + GDP_capita + generation

threshold <- mean(suicide_data$suicide_rates)
suicide_data$suicides_binary <- ifelse(
  suicide_data$suicide_rates > threshold, 1, 0)

clog <- clogit(suicides_binary ~ year + population + GDP_capita + 
                 strata(sex) + strata(age) +
                 strata(generation) + strata(id), data = suicide_data)

summary(clog)$coefficients

############ longitudinal ############

rate_thres <- mean(suicide_data$suicide_rates)

# suicide rate > threshold: rate_categ=1, 
suicide_data$rate_categ <-  as.numeric(suicide_data$suicide_rates>rate_thres)

glmm <- glmer(rate_categ ~ year + sex * age + population + GDP_capita + 
                generation + (1|id), family="binomial", data=suicide_data,
              nAGQ=0,
              control=glmerControl(optimizer = "nloptwrap"))

glmm.res <- summary(glmm)

# Evaluation
pred.error <- sqrt(mean(glmm.res$residuals^2))
AIC(glmm)

############ evaluation ############

continous_models = c(lm, glm_gamma_pos, zigamma, lmm, gee_indep, gee_exch, gee_unstr)

# AIC
AIC(lm)
AIC(glm_gamma_pos)
AIC(lmm)
AIC(clog)
AIC(glmm)

# GEE has AIC of 0?
QIC(gee_indep)
QIC(gee_exch)
QIC(gee_unstr)


