# Girish Sastry
# Spam Classification
# STAT665 Final Project

data <- read.csv (file = "/home/gyratortron/Desktop/cs-yale/stat665/final/spambase/spambase.data")

pairs (~ spam + word_freq_make + word_freq_all + word_freq_our + word_freq_remove + word_freq_order, data = data, main = "Scatterplot Matrix of First 5 Features")

library (ipred)
library (adabag)

# #####################################
# Set up data 
# #####################################



# ######################################################################
# tree.learn
#
# This function grows a decision tree for the given training set using
# the rpart package, then evaluates the tree on both training and test
# data, returning the resulting predictions.
#
# (You may modify this function to return *additional* values in the
# returned list.  You also may modify the function to take additional
# *optional* input parameters, provided that the default values of
# these additional parameters cause the function to behave exactly
# like this one that we are providing.)
#
# Input parameters:
#
#    train.set  --  a data frame containing the training data
#    test.set   --  a data frame containing the test data
#    class.name --  (optional) the name of the target variable in the
#                   data frame that is to be modeled, i.e., regarded
#                   as the class or label (= "Class" by default)
#    weights    --  (optional) a vector specifying how much weight is
#                   to be assigned to each training example
#    maxdepth   --  (optional) the maximum depth of the decision trees
#                   being grown
#
# Returns a list with the following values:
#
#    train.pred --  the predictions of the computed decision tree on
#                   all of the training examples
#    test.pred  --  the predictions of the computed decision tree on
#                   all of the test examples
# #######################################################################
tree.learn <- function(train.set,
                       test.set,
                       class.name="Class",
                       weights=rep(1,times= dim(train.set)[1]),
                       maxdepth=30) {

  form <- as.formula(paste(class.name, " ~ ",
                           paste(names(train.set),collapse="+"),
                           "-", class.name))

  tree <- rpart(form, train.set,
                weights=weights,
                method="class",
                control=rpart.control(xval=1,maxdepth=maxdepth))

  train.pred <- predict(tree, train.set, type="class")
  test.pred  <- predict(tree, test.set,  type="class")

  list(train.pred=train.pred, test.pred=test.pred)
  
}

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

tree <- tree.learn (train.set = train, test.set = test, class.name = "V86")
plot (tree $ train.pred)
plot (tree $ test.pred)

tree.err.train <- misclass.error (class.vec = train $ V86, count.mat = tree $ train.pred, expected.vec = train $ V86)
print (tree.err.train)
tree.err.test<- misclass.error (class.vec = test$ V86, count.mat = tree $ test.pred, expected.vec = test$ V86)
print (tree.err.test)

library (randomForest)
forest.train.fit <- randomForest (V86 ~ ., data = train)
forest.train.pred <- predict (forest.train.fit, train, type = "class")
print (forest.train.fit)
round (importance (forest.train.fit), 2)
forest.train.pred <- round (forest.train.pred)
plot (forest.train.pred)
forest.err.train <- misclass.error (class.vec = train $ V86, count.mat = forest.train.pred, expected.vec = train $ V86)
print (forest.err.train)

forest.test.fit <- randomForest (V86 ~ ., data = test)
forest.test.pred <- predict (forest.test.fit, test, type = "class")
print (forest.test.fit)
round (importance (forest.test.fit), 2)
forest.test.pred <- round (forest.test.pred)
plot (forest.test.pred)
forest.err.test <- misclass.error (class.vec = test $ V86, count.mat = forest.test.pred, expected.vec = test $ V86)
print (forest.err.test)

# ####################################
#
# Boosting
# 
# ####################################


library (ada)

# default adaboost
ada.train.fit <- ada (V86 ~ ., data = train)
ada.test.fit <- ada (V86 ~ ., data = test)

print (ada.test.fit)
print (ada.train.fit)

# discrete AdaBoost
ada.train.fit <- ada (V86 ~ ., data = train, iter = 50, loss = "e", type = "discrete")
print (ada.train.fit)
ada.test.fit <- ada (V86 ~ ., data = test, iter = 50, loss = "e", type = "discrete")
print (ada.test.fit)


# gentle adaboost
ada.gentle.train <- ada (V86 ~ ., data = train, iter = 100, type = "gentle", nu = 0.1, bag.shift = TRUE, control = rpart.control (cp = -1, maxdepth = 8))
print (ada.gentle.train)
ada.gentle.test <- ada (V86 ~ ., data = test, iter = 100, type = "gentle", nu = 0.1, bag.shift = TRUE, control = rpart.control (cp = -1, maxdepth = 8))
print (ada.gentle.test)

# L2 Boost
ada.log.train <- ada (V86 ~ ., data = train, iter = 50, loss = "1", type = "gentle", test.x = as.matrix(test1), test.y = as.matrix (targets))
print (ada.log.train)
ada.log.test <- ada (V86 ~ ., data = test, iter = 50, loss = "1", type = "gentle")
print (ada.log.test)

