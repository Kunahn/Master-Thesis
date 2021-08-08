require(obliqueRF)
data(iris)
## data
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

## example: importance
imp<-rep(0,ncol(x))
names(imp)<-colnames(x)
numIterations<-10      #increase the number of iterations for better results, e.g., numIterations=100

for(i in 1:numIterations){
	obj<-obliqueRF(x,y,
		training_method="log", bImportance=TRUE,
		mtry=2, ntree=500)
	imp<-imp+obj$imp
	plot(imp,t='l', main=paste("steps:", i*100), ylab="obliqueRF importance")
}
obj$trees
#plot(obj$trees)