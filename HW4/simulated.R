# # # # ################################
# Girish Sastry
# 
# This is an exercise to demonstrate the improvement
# of Bootstrap aggregated trees versus normal classification
# trees on a simulated data set.
#
# ###############################
library (ipred)
library (adabag)
# Simulated data
n0 <- 2000
n <- 1000
p <- 10 # make change of the dimension to see the  effect of Curse of Dimensionality
x <- matrix(rnorm(n*p), n0, p)
crit <- qchisq(p=0.5, df=p)
x2sum <- apply(x^2, 1, sum)

y <- ifelse(x2sum>crit, 1,0)
xy <- data.frame(cbind(x,y)[1:n,])
xy.test <- data.frame(cbind(x,y)[(n+1):n0,])
names(xy) <- c(paste("X", 1:p,sep=""),"Y")
names(xy.test) <- c(paste("X", 1:p,sep=""),"Y")
plot(X2 ~ X1, data=xy,col=y+1)
y.tr <- xy[,p+1]
y.test <- xy.test[,p+1]

# bagging

xy.bagging.train <- bagging.cv (Y ~ ., data = xy)
xy.bagging.test <- bagging.cv (Y ~ ., data=xy.test)

# ################################################
# Set up cross validation by splitting into 
# 10 folds, with an equal number of samples in 
# each fold. (For training)
# ################################################
numRows <- nrow (xy)
K <- 10 #10-FOLD CROSS VALIDATION
tail <- numRows %/% K
set.seed(5)
generator <- runif(numRows)
range <- rank(generator)
block <- (range - 1) %/% tail + 1 # associate with each individual in a block
block <- as.factor (block)
print (summary (block))

train.err <- numeric(0)
bagg.err <- numeric(0)
for (k in 1:K) {
  # Tree model
  train <- rpart (Y ~ ., data = xy[block!=k,], method = "class")
  train.pred <- predict (train, newdata = xy[block==k,], type = "class")
  m.confusion <- table (xy $ Y[block==k], train.pred)
  this.err <- 1.0 - (m.confusion[1,1] + m.confusion[2,2])/sum(m.confusion)
  train.err <- rbind (train.err, this.err)
}

train.err.cv <- mean (train.err)
print (train.err.cv)


# ################################################
# Set up cross validation by splitting into 
# 10 folds, with an equal number of samples in 
# each fold. (For test)
# ################################################
numRows <- nrow (xy.test)
K <- 10 #10-FOLD CROSS VALIDATION
tail <- numRows %/% K
set.seed(5)
generator <- runif(numRows)
range <- rank(generator)
block <- (range - 1) %/% tail + 1 # associate with each individual in a block
block <- as.factor (block)
print (summary (block))

test.err <- numeric(0)
for (k in 1:K) {
  # Tree model
  test<- rpart (Y ~ ., data = xy.test[block!=k,], method = "class")
  test.pred <- predict (train, newdata = xy.test[block==k,], type = "class")
  m.confusion <- table (xy.test $ Y[block==k], train.pred)
  err <- 1.0 - (m.confusion[1,1] + m.confusion[2,2])/sum(m.confusion)
  test.err <- rbind (train.err, this.err)
}

test.err.cv <- mean (test.err)
print (test.err.cv)
