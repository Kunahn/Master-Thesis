#Download required packages
library(randomForest)

library(tree)
library(ggplot2)
library(GGally)
library(dplyr)

#Get a glimpse on the dataset and see the summary
#iris %>% head()
#iris %>% tail()
#summary(iris)

#Radom forest is based on decision tree.
#decision_tree <- tree(Species ~ ., data = iris) # Interpretation
                                                # 1. use tree function  
                                                # 2. sort species
                                                # 3. based on all(.) variables
                                                # 4. data is iris dataset
#decision_tree

#summary(decision_tree)

#plot(decision_tree)
#text(decision_tree)

#Testset, Trainset
index_row <- sample(2, 
                    nrow(iris), 
                    replace = T, 
                    prob = c(0.7, 0.3)
                    )                 #assign values to the rows (1: Training, 2: Test)
train_data <- iris[index_row == 1,]
test_data <- iris[index_row == 2,]

#Random Forest(Training)
iris_classifier <- randomForest(Species ~., 
                                data = train_data, #train data set 
                                importance = T,ntree=500) 
iris_classifier$predicted               #Confusion matrix: prediction evaluation


#plot(iris_classifier)

#importance(iris_classifier)   #Petal features are more important
#varImpPlot(iris_classifier)

#qplot(Petal.Width, Petal.Length, data=iris, color = Species)
#qplot(Sepal.Width, Sepal.Length, data=iris, color = Species)

#predicted_table <- predict(iris_classifier, test_data[,-5])
#table(observed = test_data[,5], predicted = predicted_table)
