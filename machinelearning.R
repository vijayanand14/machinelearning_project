# Loading data
data <- read.csv("pml-training.csv")
colnames(data)
summary(data)

# Cross validation
library(caret)
set.seed(1111)
train <- createDataPartition(y=data$classe,p=.70,list=F)
training <- data[train,]
testing <- data[-train,]

# Cleaning the training data
# exclude identifier, timestamp, and window data (they cannot be used for prediction)
Cl <- grep("name|timestamp|window|X", colnames(training), value=F) 
trainingCl <- training[,-Cl]
# select variables with high (over 95%) missing data --> exclude them from the analysis
trainingCl[trainingCl==""] <- NA
NArate <- apply(trainingCl, 2, function(x) sum(is.na(x)))/nrow(trainingCl)
trainingCl <- trainingCl[!(NArate>0.95)]
summary(trainingCl)

# PCA
preProc <- preProcess(trainingCl[,1:52],method="pca",thresh=.8) #12 components are required
preProc <- preProcess(trainingCl[,1:52],method="pca",thresh=.9) #18 components are required
preProc <- preProcess(trainingCl[,1:52],method="pca",thresh=.95) #25 components are required

preProc <- preProcess(trainingCl[,1:52],method="pca",pcaComp=25) 
preProc$rotation
trainingPC <- predict(preProc,trainingCl[,1:52])

# Random forest
library(randomForest)
modFitRF <- randomForest(trainingCl$classe ~ .,   data=trainingPC, do.trace=F)
print(modFitRF)

importance(modFitRF)

# Check with test set
testingCl <- testing[,-Cl]
testingCl[testingCl==""] <- NA
NArate <- apply(testingCl, 2, function(x) sum(is.na(x)))/nrow(testingCl)
testingCl <- testingCl[!(NArate>0.95)]
testingPC <- predict(preProc,testingCl[,1:52])
confusionMatrix(testingCl$classe,predict(modFitRF,testingPC))

# Predict classes of 20 test data
testdata <- read.csv("pml-testing.csv")
testdataCl <- testdata[,-Cl]
testdataCl[testdataCl==""] <- NA
NArate <- apply(testdataCl, 2, function(x) sum(is.na(x)))/nrow(testdataCl)
testdataCl <- testdataCl[!(NArate>0.95)]
testdataPC <- predict(preProc,testdataCl[,1:52])
testdataCl$classe <- predict(modFitRF,testdataPC)

