setwd("C:/Blog/ab")

#Fetch command line arguments (if this file is being called from Python)
myArgs<-commandArgs(trailingOnly=TRUE)

#Load packages
if(!require(h2o)) install.packages("h2o", dependencies=T)
library(h2o)

#Read data
alldata<-read.csv("docvecs.csv", header=T)

#Check factor levels
summary(alldata$Dataset)
summary(alldata$Sentiment)

##############################
###2 CLASS CLASSIFIER#########
##############################

#Initialize cluster
h2o.init(nthreads=-1, max_mem_size='12g')
h2o.clusterInfo()

#Create training and test sets
train<-alldata[alldata$Dataset=='train',]
test<-alldata[alldata$Dataset=='test',]
train$Sentiment<-as.factor(train$Sentiment)
test$Sentiment<-as.factor(test$Sentiment)

#Create h2o dataframes
train.hex<-as.h2o(train, destination_frame="train.hex")
test.hex<-as.h2o(test, destination_frame="test.hex")

#Try a logistic regression classifier?
model.log<-h2o.glm(x=6:305,
                   y=4,
                   training_frame=train.hex,
                   validation_frame=test.hex, 
                   family="binomial",
                   link="logit",
                   lambda_search=T,
                   nlambdas = 5,
                   alpha=0.5)

#Train deep learning model?
model.dl<-h2o.deeplearning(x=6:305,
                           y=4,
                           training_frame=train.hex,
                           validation_frame=test.hex, 
                           activation="RectifierWithDropout",
                           hidden=c(100, 100, 100),
                           distribution="bernoulli",
                           hidden_dropout_ratios=c(0.5, 0.5, 0.5))

