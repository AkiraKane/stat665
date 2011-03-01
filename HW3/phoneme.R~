# Phone Data Classification
# Girish Sastry

phoneme.data <- read.csv (file <- "/home/gyratortron/Desktop/cs-yale/stat665/HW3/phoneme.data", header<-TRUE, sep <- ",") 

# Subset of data, as described in Elements of Statistical Learning: "aa" and "ao" only

library (Design)
library (nnet)

phoneme.sub <- phoneme.data[which(phoneme.data[,258] == 'aa' | phoneme.data[,258] == 'ao'),]
x.test.sub <- phoneme.sub[1:858,2:257]
x.train.sub <- phoneme.sub[859:1717, 2:257]
y.test.sub <- phoneme.sub[1:858, 258]
y.train.sub <- phoneme.sub[859:1717, 258]
raw.sub <- lrm(y.train.sub ~ ., data = x.train.sub)
raw.sub.train <- predict.lrm(raw.sub, x.train.sub, type = "fitted.ind")
raw.sub.test <- predict.lrm(raw.sub, x.test.sub, type = "fitted.ind")

raw.sub2 <- multinom(y.train.sub ~., data = x.train.sub)
raw.sub.train2 <- predict(raw.sub2, x.train.sub)
raw.sub.test2 <- predict(raw.sub2, x.test.sub)

raw.sub3 <- glm (y.train.sub ~., data = x.train.sub, family = binomial)
raw.sub.train3 <- predict(raw.sub3, x.train.sub)
raw.sub.test3 <- predict(raw.sub3, x.test.sub)

misclass.error.str <- function (knn.mat, class.vec, len = length (class.vec), count.mat = as.integer (as.matrix (knn.mat)) - as.matrix (class.vec), expected.vec = rep (0, length (count.mat)) ) {
  j = 1
  num = 0
  while (j <= len) {
    if (as.integer (count.mat[j]) != as.integer (expected.vec[j])) {
      num = num + 1
    }
  j = j + 1
  }
  return (num/len)
}

raw.sub.train.err <- misclass.error.str (len = 859, count.mat = raw.sub.train2, expected.vec = y.train.sub)
raw.sub.test.err <- misclass.error.str (len = 858, count.mat = raw.sub.test2, expected.vec = y.test.sub)

# Regularized logistic regression

library (splines)
x.train.sub.mat = as.matrix(x.train.sub)

# compute b-spline transf matrix
compute.transf.mat <- function (x.train.mat, df.val) {
  x.transf.mat.t = mat.or.vec(nrow(x.train.sub.mat), df.val) # x.transf.mat is a matrix of our transformed values
  i = 1
  while (i <= nrow(x.train.mat)) {
    spline.mat <- ns(x.train.mat[i,], df = df.val) 
    transformed = t(spline.mat) %*% (x.train.mat[i,])
    x.transf.mat.t[i,] <- transformed
    i = i + 1
  }
  return (x.transf.mat.t)
}

x.transf.mat.train <- compute.transf.mat (x.train.mat = x.train.sub.mat, df.val = 13)

# we can now run logistic regression on x.transf.mat

x.test.sub.mat <- as.matrix (x.test.sub)
x.transf.mat.test <- compute.transf.mat (x.train.mat = x.test.sub.mat, df.val = 13)

lrm.transf.sub <- multinom(y.train.sub ~., data = as.data.frame(x.transf.mat))
lrm.train.sub.p <- predict(lrm.transf.sub, x.transf.mat)
lrm.test.sub.p <- predict(lrm.transf.sub, x.transf.mat.test)

reg.sub.train.err <- misclass.error.str (len = 859, count.mat = lrm.train.sub.p, expected.vec = y.train.sub)
reg.sub.test.err <- misclass.error.str (len = 858, count.mat = lrm.test.sub.p, expected.vec = y.test.sub)

# Now extending analysis to the whole data set...

# First, split into training and test set (down the middle)
x.train = phoneme.data[1:2255,2:257]
x.test = phoneme.data[2256:4509,2:257]
y.train = phoneme.data[1:2255, 258]
y.test = phoneme.data[2256:4509, 258]

raw.log <- multinom(y.train ~., data = x.train)
raw.log.train <- predict(raw.log, x.train)
raw.log.test <- predict(raw.log, x.test)

reg.sub.train.err <- misclass.error.str (len = 2255, count.mat = raw.log.train, expected.vec = y.train)
reg.sub.test.err <- misclass.error.str (len = 2254, count.mat = raw.log.test, expected.vec = y.test)

#Regularized logistic regression

x.train.mat = as.matrix(x.train)

# Regularize training data
x.train.reg <- compute.transf.mat (x.train.mat = x.train.mat, df.val = 13)

# Regularize testing data

x.test.reg <- compute.transf.mat (x.train.mat = x.test.mat, df.val = 13)

reg.log.all <- multinom(y.train ~., data = as.data.frame(x.train.reg))
reg.log.train <- predict(reg.log.all, as.data.frame(x.train.reg))
reg.log.test <- predict(reg.log.all, as.data.frame(x.test.reg))

# QDA (raw)

library (MASS) 

raw.qda <- qda(y.train ~., data = as.data.frame(x.train))
raw.qda.train <- predict(raw.qda, x.train)$class
raw.qda.test <- predict(raw.qda, as.data.frame(x.test))$class

reg.sub.train.err <- misclass.error.str (count.mat = raw.qda.train, class.vec = raw.qda.train, expected.vec = y.train)
reg.sub.test.err <- misclass.error.str (count.mat = raw.qda.test, class.vec = raw.qda.test, expected.vec = y.test)

# QDA (regularized)
# Choose knots as: 6,7,8,9,10 knots in each case, cross-validate to pick # of knots

error.mat = mat.or.vec (5,11)

for(k in 6:10) {
  nFolds = 10
  subsetsize = floor(nrow(x.train.set)/nFolds)
  i = 1
  while(i <= nFolds) {
    l = 1 + (i-1)*subsetsize
    u = i*subsetsize
    x.test.set = x.test[l:u,]
    x.train.set = x.train[-(l:u),]
    y.train.set = y.train[-(l:u)]
    y.test.set = y.test[l:u]
    x.train.mat = as.matrix(x.train.set)
    x.test.mat = as.matrix(x.test.set)

  XTRtrain = mat.or.vec(nrow(x.train.set), k)
  XTRtest = mat.or.vec(nrow(x.test.set), k)
  n = 1
  while(n <= nrow(x.train.set)) {
    plin <- ns(x.train.mat[n,], df = k)
    rans = t(splin) %*% (x.train.mat[n,])
    XTRtrain[n,] <- trans
    n = n + 1 
  }

  m = 1
  while(m <= nrow(x.test.set)) {
    splin <- ns(x.test.mat[m,], df = k)
    trans = t(splin) %*% (x.test.mat[m,])
    XTRtest[m,] <- trans
    m = m + 1 
  }

  qdacv <- qda(y.train.set ~., data = as.data.frame(XTRtrain))
  qdacvTest <- predict(qdacv, as.data.frame(XTRtest))$class

  j = 1
  count = 0
  while(j <= length(qdacvTest)) {
    if(qdacvTest[j] != y.test.set[j]) {
      count = count + 1
    }
    j = j + 1
  }

  errRate = count/length(y.test.set)
  error.mat[k-5,i] <- errRate
  i = i + 1
  }
}

for(k in 1:5) {
  error.mat[k,11] = sum(error.mat[k,1:10])/10
}


# 8 knots minimizes error, now run QDA

# Regularize training data
x.train.mat = as.matrix(x.train)
x.train.reg <- compute.transf.mat (x.train.mat = x.train.mat, df = 9)

# Regularization of test data
x.test.mat = as.matrix(x.test)
x.test.reg <- compute.transf.mat (x.train.mat = x.test.mat, df = 9)

phoneme.qda.train <- qda(y.train ~., data = as.data.frame(x.train.reg))
phoneme.qda.train2 <- predict(phoneme.qda.train, as.data.frame(x.train.reg))$class
phoneme.qda.test.p <- predict(phoneme.qda.train, as.data.frame(x.test.reg))$class

reg.qda.train.err <- misclass.error.str (len = 2255, count.mat = phoneme.qda.train, expected.vec = y.train)
reg.qda.test.err <- misclass.error.str (len = 2254, count.mat = phoneme.qda.test, expected.vec = y.test)