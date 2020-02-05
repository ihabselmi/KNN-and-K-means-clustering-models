
# Clear environment

rm(list = ls())
set.seed(12)
# Load the kernlab library (which contains the ksvm function) and read in the data
#

library(kernlab)
library(kknn)
# Download Data

ccdata <- read.delim("~/Desktop/Georgia Tech Classes/Warming up/ISyE 6501/credit_card_data.csv", header=FALSE)

head(ccdata)
# Question 3.a

# Method 1: Train.kknn 
# set maximum value of k (number of neighbors) to test

km <- 10

model_kknn <- train.kknn(V11~.,ccdata,kmax=km,scale=TRUE)

# Create an array of prediction qualities

accuracy <- rep(0,km)

# Compute prediction

for (k in 1:km) {
  predicted <- as.integer(fitted(model_kknn)[[k]][1:nrow(ccdata)] + 0.5) 
  accuracy[k] <- sum(predicted == ccdata$V11)
}

# Compute the accuracy
accuracy

plot(1:km, accuracy,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of neighbors K",
     ylab="Accuracy")

# Accuracy was used to select the optimal model using  the largest value. The final value used for the model was k = 5.

# Method 2: Caret package 

#install.packages("caret",dependencies = TRUE)
#install.packages("quantreg")
library(caret)

kmax <- 10

knn_caret <- train(as.factor(V11)~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10,
                 ccdata, 
                 method = "knn", 
                 trControl=trainControl(
                   method="repeatedcv", 
                   number=10, 
                   repeats=5), 
                 preProcess = c("center", "scale"), 
                 tuneLength = kmax) 

# We now check the result to identify the best value of k and the associated accuracy

knn_caret
plot(knn_caret$results[,1], knn_caret$results[,2],
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of neighbors K",
     ylab="Accuracy")

# Method 3: cv.kknn
library(kknn)
kmax <- 10

# create array of prediction qualities

accuracy_cv <- rep(0,kmax)

# Compute prediction

for (k in 1:kmax) {
  
  model_cv <- cv.kknn(V11~.,ccdata,
                   kcv=10,
                   k=k,
                   scale=TRUE) 
  predicted <- as.integer(model_cv[[1]][,2] + 0.5) 
  accuracy_cv[k] <- sum(predicted == ccdata$V11)
}

# Compute accuracies

accuracy_cv
plot(1:kmax, accuracy_cv,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of neighbors K",
     ylab="Accuracy")


# Question 3.b
rm(list = ls())
library(kernlab)
library(kknn)
set.seed(10)
# Download the data
ccdata <- read.delim("~/Desktop/Georgia Tech Classes/Warming up/ISyE 6501/credit_card_data.csv", header=FALSE)

head(ccdata)
# Splitting the Data

#Set the fractions of the dataframe you want to split into training, 
# validation, and test.
fractionTraining   <- 0.60

# Compute sample sizes.
sampleSizeTraining   <- floor(fractionTraining   * nrow(ccdata))
train = sample(nrow(ccdata), sampleSizeTraining)
# Using the remaining data for test and validation split
remaining = ccdata[-train, ]  
# Half of what's left for validation, half for test
rem_val = sample(nrow(remaining), size = floor(nrow(remaining)/2))
# Finally, output the three dataframes for training, validation and test.
dataTraining = ccdata[train,]
dataValidation = remaining[rem_val,]  # validation data set
dataTest = remaining[-rem_val, ] # test data set


# We'll pick the best of 9 SVM models and 20 KNN models

acc <- rep(0,29)  # 1-9 are SVM, 10-29 are KNN

# Method 1: SVM models 

# Set up values of C to test

amounts <- c(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000) 

for (i in 1:9) {
  
  # fit model 
  
  model_svm <- ksvm(as.matrix(dataTraining[,1:10]),
                       as.factor(dataTraining[,11]),
                       type = "C-svc", 
                       kernel = "vanilladot", 
                       C = amounts[i],
                       scaled=TRUE) # have ksvm scale the data for you
  
  #  compare models using validation set
  
  pred <- predict(model_svm,dataValidation[,1:10])
  acc[i] = sum(pred == dataValidation$V11) / nrow(dataValidation)
}

acc[1:9]

# find best-performing SVM model on validation data

cat("Best SVM model is number ",which.max(acc[1:9]),"\n")
cat("Best C value is ",amounts[which.max(acc[1:9])],"\n")
cat("Best validation set correctness is ",max(acc[1:9]),"\n")


# retrain the best model

model_svm <- ksvm(as.matrix(dataTraining[,1:10]),
                     as.factor(dataTraining[,11]),
                     type = "C-svc", 
                     kernel = "vanilladot", 
                     C = amounts[which.max(acc[1:9])],
                     scaled=TRUE) 


cat("Performance on test data = ",sum(predict(model_svm,dataTest[,1:10]) == dataTest$V11) / nrow(dataTest),"\n")

## Performance on test data =  0.8396947

# Method 2: KNN models -------------------

for (k in 1:20) {
  
  # fit k-nearest-neighbor model using training set, validate on test set
  
  knn_model <- kknn(V11~.,dataTraining,dataValidation,k=k,scale=TRUE)
  
  #  compare models using validation set
  
  pred <- as.integer(fitted(knn_model)+0.5) # round off to 0 or 1
  
  acc[k+9] = sum(pred == dataValidation$V11) / nrow(dataValidation)
}

acc[10:29]


# find best-performing KNN model on validation data

cat("Best KNN model is k=",which.max(acc[10:29]),"\n")
cat("Best validation set correctness is ",max(acc[10:29]),"\n")


# run the best model on test data

knn_model <- kknn(V11~.,dataTraining,dataTest,
                  k=which.max(acc[10:29]),
                  scale=TRUE)

pred <- as.integer(fitted(knn_model)+0.5) # round off to 0 or 1

cat("Performance on test data = ",sum(pred == dataTest$V11) / nrow(dataTest),"\n")

#
# --------------- Evaluate overall best model on test data -------------------
#

if (which.max(acc) <= 9)  {        # if a ksvm method is best
  
  # evaluate the ksvm method on the test set to find estimated quality
  
  cat("Use ksvm with C = ",amounts[which.max(acc[1:9])],"\n")
  cat("Test performace = ",sum(predict(model_svm,dataTest[,1:10]) == dataTest$V11) / nrow(dataTest),"\n")
  
} else {    # the best is a knn method
  
  # evaluate the knn method on the test set to find estimated quality
  
  cat("Use knn with k = ",which.max(acc[10:29]),"\n")
  cat("Test performance = ",sum(pred == dataValidation$V11) / nrow(dataValidation),"\n")
  
}

## Use ksvm with C =  0.01 


# Question 4.2:
rm(list = ls())
set.seed(123)
data_iris <- read.csv("~/Desktop/Georgia Tech Classes/Warming up/ISyE 6501/iris.csv", sep="")

head(data_iris)
summary(data_iris)

# Visualizing the data:

library(ggplot2)

ggplot(data_iris, aes(Sepal.Length, Sepal.Width, color = Species))+geom_point()
# We observe that Sepal.Length and Sepal.Width are not as easily divisible between species 
# Indeed, setosa represented in red is clearly separate from versicolor and virginica, however, versicolor and virginica 
# overlap)
ggplot(data_iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point()
# Here we see a clear separation between species.

Cluster1 <- kmeans(data_iris[,1:4], 2, nstart = 50)
Cluster2 <- kmeans(data_iris[,1:4], 3, nstart = 50)
Cluster1
Cluster2

# Elbow Method for finding the optimal number of clusters
# Compute and plot cluster_ for k = 1 to k = 10.
k.max <- 10
data <- data_iris
cluster_ <- sapply(1:k.max, 
              function(k){kmeans(data[,1:4], k, nstart=50,iter.max = 15 )$tot.withinss})
cluster_
plot(1:k.max, cluster_,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#Elbow Method for finding the optimal number of clusters with 4 varaibles
# Compute and plot cluster_ for k = 1 to k = 10.
k.max <- 10
data <- data_iris
cluster_ <- sapply(1:k.max, 
                   function(k){kmeans(data[,1:4], k, nstart=50,iter.max = 15 )$tot.withinss})
cluster_
plot(1:k.max, cluster_,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#Elbow Method for finding the optimal number of clusters with 3 varaibles
# Compute and plot cluster_ for k = 1 to k = 10.
k.max <- 10
data <- data_iris
cluster_ <- sapply(1:k.max, 
                   function(k){kmeans(data[,1:3], k, nstart=50,iter.max = 15 )$tot.withinss})
cluster_
plot(1:k.max, cluster_,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#Elbow Method for finding the optimal number of clusters with 2 varaibles
# Compute and plot cluster_ for k = 1 to k = 10.
k.max <- 10
data <- data_iris
cluster_ <- sapply(1:k.max, 
                   function(k){kmeans(data[,1:2], k, nstart=50,iter.max = 15 )$tot.withinss})
cluster_
plot(1:k.max, cluster_,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

# Therefore for k=4 the between_ss/total_ss ratio tends to change slowly and remain less 
# changing as compared to other k???s.so for this data k=4 should be a good choice for number 
# of clusters however k=5 also seems to be a potential candidate. 
# The observation is the same for 2 up to 5 different combinations of variables.

# Cluster1 and Cluster2 are obtained with unscaled data. Based on the class notes, it's usually good practice to scale the data:

scale_data <- data_iris # initialize value/size of sdata
for (i in 1:4) { scale_data[,i] <- (data_iris[,i]-min(data_iris[,i]))/(max(data_iris[,i])-min(data_iris[,i])) }

# Now we can run the clustering algorithm on the scaled data:

scale_data_Cluster1 <- kmeans(scale_data[,1:4], 2, nstart = 50)
scale_data_Cluster2 <- kmeans(scale_data[,1:4], 3, nstart = 50)

# Let's see what the total distance is between data points and cluster centers:

# initialize distance to zero

csum = 0

# for each data point...

for (i in 1:nrow(data_iris)) {
  
  # ...add the distance between its point and its cluster center
  
  csum = csum + dist(rbind(data_iris[i,1:4],Cluster2$centers[Cluster2$cluster[i],]))
}

# report the total

csum[1]

#Now, let's compare the clusters with the species.
#We can only do this because we happen to know the species of each data point.
# Normally, this information wouldn't be available.

table(Cluster2$cluster, data_iris$Species)
table(scale_data_Cluster2$cluster, data_iris$Species)
table(Cluster1$cluster, data_iris$Species)
table(scale_data_Cluster1$cluster, data_iris$Species)

# -------------------------- Using Petal Length and Petal Width ---------------------

#Because of the graphical analysis above, let's try just using petal length and petal width

irisClusterPET2 <- kmeans(data_iris[,3:4], 2, nstart = 20)
irisClusterPET3 <- kmeans(data_iris[,3:4], 3, nstart = 20)
irisClusterPET4 <- kmeans(data_iris[,3:4], 4, nstart = 20)
irisClusterPET5 <- kmeans(data_iris[,3:4], 5, nstart = 20)

irisClusterPETsc2 <- kmeans(scale_data[,3:4], 2, nstart = 20)
irisClusterPETsc3 <- kmeans(scale_data[,3:4], 3, nstart = 20)
irisClusterPETsc4 <- kmeans(scale_data[,3:4], 4, nstart = 20)
irisClusterPETsc5 <- kmeans(scale_data[,3:4], 5, nstart = 20)

#Let us compare the clusters with the species.
#We can only do this because we happen to know the species of each data point.
# Normally, this information wouldn't be available.

table(irisClusterPET2$cluster, data_iris$Species)
table(irisClusterPETsc2$cluster, data_iris$Species)
table(irisClusterPET3$cluster, data_iris$Species)
table(irisClusterPETsc3$cluster, data_iris$Species)
table(irisClusterPET4$cluster, data_iris$Species)
table(irisClusterPETsc4$cluster, data_iris$Species)
table(irisClusterPET5$cluster, data_iris$Species)
table(irisClusterPETsc5$cluster, data_iris$Species)

irisClusterPETsc3

# plot solution
ggplot(iris, aes(Petal.Length, Petal.Width, color = irisClusterPETsc3$cluster)) + geom_point()
