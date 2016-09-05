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

#Try a logistic regression classifier
model.log<-h2o.glm(x=8:407,
                   y=4,
                   training_frame=train.hex,
                   validation_frame=test.hex, 
                   nfolds=5,
                   fold_assignment="AUTO", 
                   family="binomial",
                   link="logit",
                   lambda=0,
                   alpha=0.5)

#Train deep learning model
model.dl<-h2o.deeplearning(x=6:305,
                           y=4,
                           training_frame=train.hex,
                           validation_frame=test.hex, 
                           #nfolds=5,
                           #fold_assignment="AUTO", 
                           activation="Rectifier",
                           hidden=c(50, 50),
                           distribution="bernoulli")

#Get threshold and accuracy for maximum-per-class-accuracy
thresh.maxpca<-model.dl@model$cross_validation_metrics@metrics$max_criteria_and_metric_scores[7,'threshold']
cv.accuracy<-model.dl@model$cross_validation_metrics@metrics$max_criteria_and_metric_scores[7,'value']

##################################
###PREDICT ON ALL REVIEWS#########
##################################

#Take unlabeled reviews
unlabeled<-alldata[alldata$Sentiment=="",]
unlabeled$Sentiment<-NULL

#Create h2o dataframe for all unlabeled reviews
unlabeled.hex<-as.h2o(unlabeled, destination_frame="unlabeled.hex")

#Predict using trained model
predictions<-h2o.predict(model.dl, newdata=unlabeled.hex)

#Convert to R dataframe
predictions<-as.data.frame(predictions)
#This contains predictions based on a threshold of 0.5

#Set all predictions to 'Negative'
predictions$predict<-'Negative'

#Make predictions based on threshold calculated above
predictions$predict[predictions$Positive>=thresh.maxpca]<-'Positive'

#Delete probabilities
predictions$Negative<-NULL
predictions$Positive<-NULL

#Add other columns
predictions$ReviewID<-unlabeled$ReviewID

#Rearrange columns and rename
predictions<-predictions[,c(2,1)]
names(predictions)<-c("ReviewID","Sentiment")

#Write to file
write.csv(predictions, 'Output - Sentiment_Reviews.csv', row.names=F)

#Send return code to Python (if this file is being called from Python)
cat(sprintf("Output saved to file. Had a CV accuracy of: %f percent!", cv.accuracy*100))
