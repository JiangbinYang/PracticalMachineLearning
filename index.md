# Machine Learning Project
Jiangbin Yang  
October 16, 2017  



## Executive Summary

Various machine learning models have been fitted to predict in which fashion (A, B, C, D, or E) a participant is performing weight lifting dumbbell exercise, using wearable sensor measurement data. A Random Forest model has shown the best performance, with 98.8% of estimated out-of-sample Accuracy.

## Introduction

In a weight lifting exercise experiment, participants were asked to perform repetitions of Unilateral Dumbbell Biceps Curl in five different fashions: Class A, B, C, D, and E. Measurements from sensors such as accelerometer, gyroscope and magnetometer that were worn on participants' belt, arm, forearm and dumbbell had been collected. More information about the experiment can be found in [the weblink](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har#weight_lifting_exercises) and [the paper](http://web.archive.org/web/20161224072740/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf) here.

In this project, we will build a machine learning model to predict in which fashion (A, B, C, D, or E) a participant was performing the dumbbell exercise, using the sensor measurement data. The following will be my overall **prediction study design**:

1. Use Accurarcy as the prediction model performance evaluation criterion.
2. Partition the training dataset into 3 parts: "build" (60%), "probe" (20%) and "validation" (20%) datasets. We use the "build" dataset to develop and cross validate prediction models, the "probe" dataset to compare the model performance, and hold out the "validation" dataset to estimate the out-of-sample Accuracy. (Note that there is a *un-labeled* testing dataset, which is for submission of prediction results on 20 test cases to the Course host for project evaluation. Because it is un-labeled, this testing dataset cannot be used for out-of-sample error estimation before model submission.)
3. Select features for model building.
4. Develop, cross validate and select the best prediction model.
5. Apply the best prediction model to the held-out validation dataset to estimate out-of-sample accuracy.

## Data Loading, Exploration & Feature Selection

### Loading Data


```r
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("pml-training.csv")) 
  download.file(urlTrain, "pml-training.csv")
if (!file.exists("pml-testing.csv")) 
  download.file(urlTest, "pml-testing.csv")
WLEtraining <- read.csv("pml-training.csv")
WLEtesting <- read.csv("pml-testing.csv") # 20 test cases
sum(names(WLEtraining) == "classe")  # labeled
```

```
## [1] 1
```

```r
sum(names(WLEtesting) == "classe")  # un-labeled
```

```
## [1] 0
```

### Data Exploration & Feature Selection


```r
totalVars <- dim(WLEtraining)[2]

library(caret)
varsNearZero <- names(WLEtraining)[nearZeroVar(WLEtraining)]

sum.na <- function(x){ sum(is.na(x)) }
trainSumNA <- apply(WLEtraining, 2, sum.na)
trainVarsNoNA <- names(WLEtraining)[trainSumNA == 0]
testSumNA <- apply(WLEtesting, 2, sum.na)
testVarsAllNA <- names(WLEtesting)[testSumNA == dim(WLEtesting)[1]]
testVarsNoNA <- names(WLEtesting)[testSumNA == 0]

predictors <- testVarsNoNA[8:59]
sum(predictors %in% trainVarsNoNA)
```

```
## [1] 52
```

There are 160 variables in the training dataset. However, 60 of them are nearly zero. Among the 100 variables not nearlly zero, 93 of them have no missing values. 

In the testing dataset, 100 variables have all missing values. The rest 60 variables have no missing values. Among these 60 variables, variables #1 - #7 and #60 are id and timestamps, variables #8 - #59 are sensor measurement variables. These 52 sensor measurement variables are non-missing in the training dataset. We will use these 52 measurement features for our prediction model building.

## Model Building

To investigate the necessity of data transformation, some further exploration had been done on the data, such as histograms and pair-wise correlation / association. Principal Components Analysis had also been performed on the selected features. Models with some transformed data had been tried. However, there was no performance gain from the tried data transformation, based on preliminary model performance comparison. Due to page limitation, specific results on these exploration are not shown in this report.

### Sample Partition


```r
library(caret)
set.seed(276)
inBuild <- createDataPartition(y=WLEtraining$classe, p=0.6, list=F)
build <- WLEtraining[, c("classe", predictors)][inBuild,]
inProbe <- createDataPartition(y=WLEtraining[-inBuild,]$classe, p=0.5, list=F) # 0.4*0.5 = 0.2
probe <- WLEtraining[, c("classe", predictors)][-inBuild,][inProbe,]
validation <- WLEtraining[, c("classe", predictors)][-inBuild,][-inProbe,]
```

### Model Fitting

Different types of models have been fitted on the "build" dataset using cross-validation, including Decision Tree, Linear Discriminant Analysis, Naive Bayes, Gradient Boosting Machine and Random Forest. Prediction Accuracy on the "probe" dataset of the fitted models has been compared. It turned out that a Random Forest model would perform the best. 


```r
library(caret)
set.seed(1227)
fitControl <- trainControl(method = "cv", number = 5)
```

```r
modTree <- train(classe~., data = build, method = "rpart", trControl = fitControl)
cmTree <- confusionMatrix(predict(modTree, probe), probe$classe)
cmTree$overall[1] # Accuracy on the Probe Dataset
```

```
##  Accuracy 
## 0.4937548
```

```r
modLDA <- train(classe~., data = build, method = "lda", trControl = fitControl)
cmLDA <- confusionMatrix(predict(modLDA, probe), probe$classe)
cmLDA$overall[1] # Accuracy on the Probe Dataset
```

```
##  Accuracy 
## 0.7066021
```

```r
modNB <- train(classe~., data = build, method = "nb", trControl = fitControl)
cmNB <- confusionMatrix(predict(modNB, probe), probe$classe)
cmNB$overall[1] # Accuracy on the Probe Dataset
```

```
##  Accuracy 
## 0.7427989
```

```r
modGBM <- train(classe~., data = build, method = "gbm", trControl = fitControl, verbose = FALSE)
cmGBM <- confusionMatrix(predict(modGBM, probe), probe$classe)
cmGBM$overall[1] # Accuracy on the Probe Dataset
```

```
##  Accuracy 
## 0.9579404
```

```r
modRF <- train(classe~., data = build, method = "rf", trControl = fitControl)
cmRF <- confusionMatrix(predict(modRF, probe), probe$classe)
cmRF$overall[1] # Accuracy on the Probe Dataset
```

```
##  Accuracy 
## 0.9900586
```


```r
cmRF_validate <- confusionMatrix(predict(modRF, validation), validation$classe)
print(cmRF_validate)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114   10    0    0    0
##          B    2  746    8    0    0
##          C    0    3  674   16    2
##          D    0    0    2  625    4
##          E    0    0    0    2  715
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9875          
##                  95% CI : (0.9835, 0.9907)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9842          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9829   0.9854   0.9720   0.9917
## Specificity            0.9964   0.9968   0.9935   0.9982   0.9994
## Pos Pred Value         0.9911   0.9868   0.9698   0.9905   0.9972
## Neg Pred Value         0.9993   0.9959   0.9969   0.9945   0.9981
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1902   0.1718   0.1593   0.1823
## Detection Prevalence   0.2865   0.1927   0.1772   0.1608   0.1828
## Balanced Accuracy      0.9973   0.9899   0.9894   0.9851   0.9955
```

When applied to the held-out validation dataset, the Randome Forest model accuracy is 98.8%, with 95% confidence interval of [98.4%, 99.1%]. This would be our estimated out-of-sample prediction accuracy using the Randome Forest model.

To predict on the 20 test cases, we would apply the following code. (Results are not shown here.)


```r
predTestDF <- data.frame(prediction=predict(modRF, WLEtesting), WLEtesting)
write.csv(predTestDF, file = "pml-testing-prediction.csv", row.names = F)
```
