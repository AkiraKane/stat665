# Classification of zipcode data
# Girish Sastry

train = read.table (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW3/zip.train")
test = read.table (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW3/zip.test")

# K-NN
library (class)

misclass.error <- function (knn.mat, class.vec, count.mat = as.integer (as.matrix (knn.mat)) - as.matrix (class.vec), expected.vec = rep (0, length (count.mat)) ) {
  j = 1
  num = 0
  while (j <= length (class.vec)) {
    if (count.mat[j] != expected.vec[j]) {
      num = num + 1
    }
  j = j + 1
  }
  return (num/length (class.vec))
}
  

# 1-NN
one.nn.train <- knn (train, train, cl = train[,1], k = 1, l = 0, prob = FALSE, use.all = TRUE)
one.nn.test <- knn (train, test, cl = train[,1], k = 1, l = 0, prob = FALSE, use.all = TRUE)

one.nn.train.err <- misclass.error (one.nn.train, train[,1])
one.nn.test.err <- misclass.error (one.nn.test, test[,1])

# 7-NN
seven.nn.train <- knn (train, train, cl = train[,1], k = 7, l = 0, prob = FALSE, use.all = TRUE)
seven.nn.test <- knn (train, test, cl = train[,1], k = 7, l = 0, prob = FALSE, use.all = TRUE)

seven.nn.train.err <- misclass.error (seven.nn.train, train[,1])
seven.nn.test.err <- misclass.error (seven.nn.test, test[,1])

# 15-NN
fifteen.nn.train <- knn (train, train, cl = train[,1], k = 15, l = 0, prob = FALSE, use.all = TRUE)
fifteen.nn.test <- knn (train, test, cl = train[,1], k = 15, l = 0, prob = FALSE, use.all = TRUE)

fifteen.nn.train.err <- misclass.error (fifteen.nn.train, train[,1])
fifteen.nn.test.err <- misclass.error (fifteen.nn.test, test[,1])

# LDA

library (MASS)
train.mat <- as.matrix (train)
test.mat <- as.matrix (test)
x.train.lda <- train.mat[,-1]
x.test.lda <- test.mat[,-1]
y.train.lda <- train.mat[,1]
y.test.lda <- test.mat[,1]

x.train <- x.train.lda
x.test <- x.test.lda
y.train <- y.train.lda
y.test <- y.test.lda


train.lda <- lda (y.train.lda ~ ., data = train[,-1])
train.lda.p <- predict (train.lda, train[,-1]) $ class 
train.lda.p.int <- as.integer (train.lda.p) - 1

train.lda.diff <- train.lda.p.int - y.train.lda
train.lda.err <- misclass.error (class.vec = train[,1], count.mat = train.lda.diff)

test.lda <- lda (y.test.lda ~ ., data = test[,-1])
test.lda.p <- predict (test.lda, test[,-1]) $ class
test.lda.p.int <- as.integer (test.lda.p) - 1

test.lda.diff <- test.lda.p.int - y.test.lda
test.lda.err <- misclass.error (class.vec = test[,1], count.mat = test.lda.diff)

# PCA and then QDA

x.qda.pc1 <- prcomp (train[,-1])
x.qda.pc <- x.train.lda %*% x.qda.pc1 $ rotation
x.pc.red <- as.data.frame (x.qda.pc[,1:50])
x.train.qda <- qda (y.train.lda ~ ., data = x.pc.red)
x.train.qda.p <- predict (x.train.qda, x.pc.red) $ class

train.qda.err <- misclass.error (class.vec = train[,1], count.mat = x.train.qda.p, expected.vec = y.train.lda)

x.qda.pc.test <- prcomp (test[,-1])
x.qda.pc.test <- x.test.lda %*% x.qda.pc.test $ rotation
x.qda.pc.test.red <- as.data.frame (x.qda.pc.test[,1:50])
x.test.qda <- qda (y.test.lda ~ ., data = x.qda.pc.test.red)
x.test.qda.p <- predict (x.test.qda, x.qda.pc.test.red) $ class

test.qda.err <- misclass.error (class.vec = test[,1], count.mat = x.test.qda.p, expected.vec = y.test.lda)

# Reduced rank LDA

# Crossvalidation procedure: we divide training data into 10 sets, 9 as train, 1 as test. 
# then compute the optimal dimension for each one, compute error, and choose dimension that minimizes error

indicator.mat <- mat.or.vec (length (y.train.lda), 10)

# Populate indicator matrix
for (i in 1 : length (y.train)) {
  k <- y.train[i]
  indicator.mat[i,k+1] <- 1
}

y.hat.rrlda <- x.train %*% solve (t (x.train) %*% x.train) %*% t (x.train) %*% indicator.mat

# Loop over dimensions and find misclassification rate
nFolds <- 10
subset.size <- floor (nrow (x.train)/nFolds)
error.mat <- mat.or.vec (10, 11)

k = 1
while(k <= 10) {
  i <- 1
  while(i <= nFolds) {
    l <- 1 + (i - 1) * subset.size
    u <- i * subset.size
    train.subset <- train[-(l:u),]
    test.subset <- train[l:u,]
    test.set <- train[,-1][l:u,]
    train.set <- train[,-1][-(l:u),]
    y.train.cv <- y.train[-(l:u)]
    y.test.cv <- y.train[l:u]
    indicator.mat.train <- mat.or.vec (length (y.train.cv), 10)
    indicator.mat.test <- mat.or.vec (length (y.test.cv), 10)

    for(l in 1:length(y.train.cv)) {
      a = y.train.cv[l]
      indicator.mat.train[l,a+1] = 1
    }

    for(L in 1:length(y.test.cv)) {
      a = y.test.cv[L]
      indicator.mat.train[L, a + 1] = 1
    }

    YhatTrain = as.matrix(train.set)%*%solve(t(as.matrix(train.set))%*%as.matrix(train.set))%*%t(as.matrix(train.set))%*%indicator.mat.train
    YhatTest = as.matrix(test.set)%*%solve(t(as.matrix(train.set))%*%as.matrix(train.set))%*%t(as.matrix(train.set))%*%indicator.mat.train
    YhatYtrain = t(YhatTrain)%*%indicator.mat.train
    YhatYtest = t(YhatTest)%*%indicator.mat.test
    eigendecTrain <- prcomp(YhatYtrain)
    eigenvecTrain = (eigendecTrain$rotation)[,1:k]
    Ytrans = YhatTrain%*%eigenvecTrain 
    YtransTest = YhatTest %*% eigenvecTrain
    ldaCV <- lda(y.train.cv~., data = as.data.frame(Ytrans))
    ldaCV2 <- predict(ldaCV, as.data.frame(YtransTest))$class

    j = 1
    count = 0
    while(j < length(test.set[,1])) {
      if(ldaCV2[j] != y.test.cv[j]) {
	count = count + 1
      }
    j = j + 1
    }

    errRate = count/length(test.set[,1])
    error.mat[k,i] <- errRate
    i = i + 1
  }
  k = k + 1
}

k = 1
while(k <= 10) {
  error.mat[k,11] = sum(error.mat[k,1:10])/10
  k = k + 1
}

#First compute our eigenvectors:

indicator.mat = mat.or.vec(length(y.train), 10)

for(j in 1:length(y.train)){
  a = y.train[j]
  indicator.mat.train[j,a+1] = 1
}

Yhat = xTrain%*%solve(t(xTrain)%*%xTrain)%*%t(xTrain)%*%indicator.mat
YhatTest = xTest%*%solve(t(xTrain)%*%xTrain)%*%t(xTrain)%*%indicator.mat
YhatY = t(Yhat)%*%indicator.mat
eigendec <- prcomp(YhatY)
eigenvec = (eigendecTrain$rotation)[,1:9]
Ytrans = Yhat%*%eigenvecTrain 
YtransTest = YhatTest %*% eigenvecTrain

ldaRed <- lda(y.train~., data = as.data.frame(Ytrans))
ldaRedTrain <- predict(ldaRed, as.data.frame(Ytrans))$class
ldaRedTest <- predict(ldaRed, as.data.frame(YtransTest))$class

train.rrlda.err <- misclass.error (class.vec = train[,1], count.mat = ldaRedTrain, expected.vec = y.train)
test.rrlda.err <- misclass.error (class.vec = test[,1], count.mat = ldaRedTest, expected.vec = y.test)


# Logistic Regression (from the DESIGN package)

library (Design)
train.lrm <- lrm (as.factor (y.train) ~ ., data = train[,-1])
train.lrm.p <- as.integer (predict (train.lrm, train[,-1])) 
train.lrm.train.err <- misclass.error (class.vec = train[,1], count.mat = train.lrm.p, expected.vec = y.train)

test.lrm.p <- predict (train.lrm, test[,-1])
test.lrm.test.err <- misclass.error (class.vec = test[,1], count.mat = test.lrm.p, expected.vec = y.test)
