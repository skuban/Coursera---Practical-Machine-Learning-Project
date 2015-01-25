library(caret); library(Hmisc); library(ggplot2)

setwd('C:\\Users\\StephenRobertKuban\\Documents\\School\\4B\\Coursera - Practical Machine Learning')
dd <- read.csv('pml-training.csv')
dd <- dd[,-(1:4)]
dd2 <- dd[,colSums(is.na(dd))==0]  #eliminate any column with an NA value
ddNoNum <- dd2[,-which(sapply(dd2, class)=='numeric' | sapply(dd2,class)=="integer")] # only numeric columns
ddnum <- dd2[,which(sapply(dd2, class)=='numeric' | sapply(dd2,class)=="integer")] # only numeric columns
ddnum2 <- cbind(ddnum, classe=dd$classe)

c <- cor(ddnum)  # correlation matrix
  for (i in 1:dim(c)[1]){ for (j in i:dim(c)[1]) { c[i,j] <- 0}}
  for (i in 1:dim(c)[1]){ for (j in 1:(i)) { if(abs(c[i,j]) < .2) {c[i,j] <- 0}}}
  # only show correlation greater than 0.2

###################################
######## get 4 cross validation sets
set.seed(56789)

CV1 <- createDataPartition(y=ddnum2$classe, p=.7, list=F)
CV1train <- ddnum2[CV1, ]
CV1test <- ddnum2[-CV1, ]

CV2 <- createDataPartition(y=ddnum2$classe, p=.7, list=F)
CV2train <- ddnum2[CV2, ]
CV2test <- ddnum2[-CV2, ]

CV3 <- createDataPartition(y=ddnum2$classe, p=.7, list=F)
CV3train <- ddnum2[CV3, ]
CV3test <- ddnum2[-CV3, ]

CV4 <- createDataPartition(y=ddnum2$classe, p=.7, list=F)
CV4train <- ddnum2[CV4, ]
CV4test <- ddnum2[-CV4, ]

###################################


ddtest <- read.csv('pml-testing.csv')
# PRINCIPAL COMPONENT ANALYSIS
pca <- preProcess(ddnum, method='pca')


# TREES
set.seed(123)
 tree1 <- train(classe~ ., data=ddnum2, method='rpart', preProc='pca')
 confusionMatrix(tree1)
 # Seems to predict A far too often.
 confusionMatrix(predict(tree1, newdata=dd), dd$classe)
 # Doesn't predict class C at all. Accuracy of only 34%


# EVALUATE ALL BELT STUFF BY PCA
belts <- dd[,which(sapply(names(dd), function(h) substring(h, nchar(h)-3)=="belt"))]

# Multinomial model
set.seed(4567)
mn <- train(classe ~ ., data=ddnum2, method='multinom', preProcess='pca', verbose=F)
mnPred <- predict(mn, newdata=dd)
confusionMatrix(mnPred, dd$classe)
  # This doesn't do very well. 45% accuracy. :/ But better than tree1.

# BOOSTING
set.seed(89787)
boost <- train(classe~ ., data=ddnum2, preProc='pca', method='gbm', verbose=F)
 boostCM <- confusionMatrix(predict(boost, newdata=dd), dd$classe)
 boostCM <-  confusionMatrix(predict(boost, newdata=dd), dd$classe)
 # High accuracy of 85%, by far the best predictive model so far, at least in accuracy

# CROSS VALIDATION OF THE BOOSTING MODEL
set.seed(6576)
boost1 <- train(classe ~ ., data=CV1train, preProc='pca', method='gbm', verbose=F)
boost1CM <- confusionMatrix(predict(boost1, newdata=CV1test), CV1test$classe)
  #               Accuracy : 0.8233
#                 95% CI : (0.8133, 0.8329)
  # Seems pretty accurate after this one run.
set.seed(90909)
boost2 <- train(classe ~ ., data=CV2train, preProc='pca', method='gbm', verbose=F)
boost2CM <- confusionMatrix(predict(boost2, newdata=CV2test), CV2test$classe)
#                 Accuracy : 0.8233
#                 95% CI : (0.8133, 0.8329)
#     Shockingly it's the exact same estimate and CI as boost1
set.seed(12123)
boost3 <- train(classe ~ ., data=CV3train, preProc='pca', method='gbm', verbose=F)
boost3CM <- confusionMatrix(predict(boost3, newdata=CV3test), CV3test$classe)
#                 Accuracy : 0.8241
#                 95% CI : (0.8142, 0.8338)
set.seed(38383)
boost4 <- train(classe ~ ., data=CV4train, preProc='pca', method='gbm', verbose=F)
boost4CM <- confusionMatrix(predict(boost4, newdata=CV4test), CV4test$classe)
  #               Accuracy : 0.8323
#                 95% CI : (0.8225, 0.8417)

# All four cross-validations in the random sampling algorithms produce a very
# narrow range of expected accuracy: between 82.3% and 83.2%. All four
# point-estimates of the accuracy are within the 95% confidence intervals of
# the other three estimates. Therefore it's reasonably expected that
# the out-of-sample error rate should be approximately 82% - 83%. i.e. we
# should expect approximately 16 of the 20 estimates in the test dataset
# to be accurately predicted.