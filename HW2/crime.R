# Girish Sastry
# Predicting crime data

crime.data <- read.csv (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/crime.data.csv", head = TRUE, sep=",", na.strings = "?")
test.data <- read.csv (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/test.data.csv", head = TRUE, sep = ",", na.strings = "?")

crime.data.t <- t (crime.data)

training <- t (na.omit (t (crime.data)))
test <- t (na.omit (t (test.data)))
test <- test[1:336,]   # rest are missing values

training.t <- t (training)

write.csv (training, file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/training.csv")
write.csv (test, file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/test.csv")

# apply (crime.data.t, 1, function (x) sum (is.na(x)))  # missing values are significant

training.set <- read.csv (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/training.csv", head = TRUE, sep = ",")
test.set <- read.csv (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/test.csv", head = TRUE, sep = ",")

# Remove non predictors
training<- training.set[6:ncol (training.set)]
test<- test.set[6:ncol (test.set)]
test $ OtherPerCap <- NULL

n.test <- nrow (test.set)

# Least squares using all variables
g <- lm (ViolentCrimesPerPop ~ ., training)
yhat.test.ls <- predict (g, test)
sqrt (sum((yhat.test.ls - test$ViolentCrimesPerPop)^2)/n.test)

# Variable Selection

# Backwise Elimination (AIC)
#step (g)

# Compare Full and reduced models
g0 <- lm (ViolentCrimesPerPop ~ ., training)
yhat.test.ls <- predict (g0, test)
sqrt (sum((yhat.test.ls - test$ViolentCrimesPerPop)^2)/n.test)

g1 <- lm(formula = ViolentCrimesPerPop ~ racepctblack + agePct12t29 + pctUrban + pctWWage + pctWFarmSelf + pctWInvInc + pctWRetire + whitePerCap + PctPopUnderPov + PctEmploy + PctOccupMgmtProf +     MalePctDivorce + MalePctNevMarr + TotalPctDiv + PctKids2Par +     PctWorkMom + NumIlleg + PctIlleg + NumImmig + PctNotSpeakEnglWell +     PctLargHouseOccup + PersPerOccupHous + PersPerOwnOccHous +     PersPerRentOccHous + PctPersOwnOccup + PctPersDenseHous +     HousVacant + PctHousOccup + PctVacantBoarded + PctVacMore6Mos +     OwnOccLowQuart + OwnOccMedVal + RentLowQ + MedRent + MedRentPctHousInc +     MedOwnCostPctInc + MedOwnCostPctIncNoMtg + NumInShelters +     NumStreet + PctForeignBorn, data = training)
yhat.test.bw <- predict (g1, test)
sqrt (sum((yhat.test.bw - test$ViolentCrimesPerPop)^2)/n.test)

# Lasso
library (lars)
g.lasso <- lars (x = as.matrix (training[,-100]), y = as.matrix (training[,100]), type = "lasso")
g.lasso.cv <- cv.lars (x = as.matrix (training[,-100]), y = as.matrix (training[,100]), K = 10, type = "lasso", plot.it = TRUE)
# plot (g.lasso.cv $ fraction, g.lasso.cv $ cv, type = "l")

g.lasso.cv$fraction[which(g.lasso.cv$cv == min (g.lasso.cv$cv))]   # minimum mean square error using cross validation

tr.lasso <- lars (x = as.matrix (training[,-100]), y = as.matrix (training[,100]), type = "lasso")
#plot (tr.lasso)
test.lasso <- predict.lars (tr.lasso, test[,-100], s = 0.44, type = "fit", mode = "fraction")
sqrt(sum((test.lasso$fit - test$ViolentCrimesPerPop)^2)/n.test)

# Ridge Regression
library (MASS)
g.rg <- lm.ridge (ViolentCrimesPerPop ~ .,training, lam = 0) 		# Same result as LS
g.rg <- lm.ridge (ViolentCrimesPerPop ~ ., training, lam = seq (0, 200, by = .1))
# plot (g.rg)				# coef in plot is for scaled inputs

# Prediction
test.x <- scale (test[,-100])
coef <- g.rg$coef[,which(g.rg$GCV==min(g.rg$GCV))]
yhat.test.rg <- mean (training[,100]) + test.x%*%coef
sqrt(sum((test[,100]-yhat.test.rg)^2)/n.test)

# Cross validation for ridge regression
cv.ridge <- function(formula,data, lam, Kfold=10, seed=seed)
  {
    n= nrow(data); p=ncol(data)
    set.seed(seed)
    id <- sample(1:nrow(data));data1=data[id,]
    group <- rep(1:Kfold, n/Kfold+1)[1:n]
    yhat = matrix(0, nrow(data), length(lam))
   for (i in 1:Kfold)
      {
        test <- data1[group==i,]
        train <- data1[group!=i,]
        result <- lm.ridge(formula, train,lam=lam)
        test.x <- scale(test[,-p])
        coef <- result$coef
        yhat[group==i,] <- mean(train[,p])+test.x%*%coef
      }
    return(apply(data1[,p]-yhat, 2, function(x)return(mean(x^2))))
 }

cvridge <- cv.ridge (ViolentCrimesPerPop ~ ., training, lam = 0:200, seed=134)
plot (0:200, sqrt (cvridge), ylab = "Lambdas", xlab = "Mean Squared Error", main = "Cross Validation for Ridge Regression", type = "l")
rr.lambda <- which.min(cvridge)
g.rg <- lm.ridge(ViolentCrimesPerPop ~., training, lam=(0:200)[rr.lambda])
test.x <- scale(test[,-100])
yhat.test <- mean(training[,100])+test.x%*%g.rg$coef
sqrt(sum((test[,100]-yhat.test)^2)/n.test)

