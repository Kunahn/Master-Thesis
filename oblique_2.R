#Download required packages
library(randomForest)

library(tree)
library(ggplot2)
library(GGally)
library(dplyr)
require(obliqueRF)

# extract feature matrix
x<-as.matrix(iris[,1:4])

# convert to 0/1 class labels
y<-as.numeric(iris[,5]=="setosa")

## train
smp<-sample(1:nrow(iris), nrow(iris)/5)
obj <- obliqueRF(x[-smp,], y[-smp])

## test
pred <- predict(obj, x[smp,], type="prob")
plot(pred[,2],col=y[smp]+1,ylab="setosa probability")
table(pred[,2]>.5,y[smp])

obj<-obliqueRF(x,y,
               training_method="log", bImportance=TRUE,
               mtry=2, ntree=500)
obj
acc <- 0
for(i in 1:150){
  acc <- acc+sqrt((obj$pred$y[i] - obj$pred$votes.test[i])^2)
}
acc <- ((acc/150))*100
acc