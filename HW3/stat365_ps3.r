#PROBLEM 2 (Zip Code Data)


train = read.table("C:/Users/Aneesh/Documents/Documents/STAT 665 (S 10)/zip.train")
test = read.table("C:/Users/Aneesh/Documents/Documents/STAT 665 (S 10)/zip.test")

#We first use the KNN method, from the class package.

#1 Nearest Neighbors
oneNNtest = knn(train, test, cl = train[,1], k = 1, l = 0, prob = FALSE, use.all = TRUE)
oneNNtrain = knn(train, train, cl = train[,1], k = 1, l = 0, prob = FALSE, use.all = TRUE)

k = as.integer(as.matrix(oneNNtest)) - as.matrix(test[,1])
j = 1
count = 0
while(j <= length(test[,1])){
if(k[j] != 0){
	count = count + 1
}
j = j + 1
}
oneNNtestErrorRate = count/length(test[,1])

k = as.integer(as.matrix(oneNNtrain)) - as.matrix(train[,1])
j = 1
count = 0
while(j <= length(train[,1])){
if(k[j] != 0){
	count = count + 1
}
j = j + 1
}
oneNNtrainErrorRate = count/length(train[,1])



#7 Nearest Neighbors
sevenNNtest = knn(train, test, cl = train[,1], k = 7, l = 0, prob = FALSE, use.all = TRUE)
sevenNNtrain = knn(train, train, cl = train[,1], k = 7, l = 0, prob = FALSE, use.all = TRUE)

k = as.integer(as.matrix(sevenNNtest)) - as.matrix(test[,1])
j = 1
count = 0
while(j <= length(test[,1])){
if(k[j] != 0){
	count = count + 1
}
j = j + 1
}
sevenNNtestErrorRate = count/length(test[,1])

k = as.integer(as.matrix(sevenNNtrain)) - as.matrix(train[,1])
j = 1
count = 0
while(j <= length(train[,1])){
if(k[j] != 0){
	count = count + 1
}
j = j + 1
}
sevenNNtrainErrorRate = count/length(train[,1])


#15 Nearest Neighbors
fifteenNNtest = knn(train, test, cl = train[,1], k = 15, l = 0, prob = FALSE, use.all = TRUE)
fifteenNNtrain = knn(train, train, cl = train[,1], k = 15, l = 0, prob = FALSE, use.all = TRUE)

k = as.integer(as.matrix(fifteenNNtest)) - as.matrix(test[,1])
j = 1
count = 0
while(j <= length(test[,1])){
if(k[j] != 0){
	count = count + 1
}
j = j + 1
}
fifteenNNtestErrorRate = count/length(test[,1])

k = as.integer(as.matrix(fifteenNNtrain)) - as.matrix(train[,1])
j = 1
count = 0
while(j <= length(train[,1])){
if(k[j] != 0){
	count = count + 1
}
j = j + 1
}
fifteenNNtrainErrorRate = count/length(train[,1])



#Now, LDA and QDA.  We use the MASS package l:
library(MASS)

trainMat = as.matrix(train)
testMat = as.matrix(test)
xTrain2 = train[,-1]
xTest2 = test[,-1]
yTrain = trainMat[,1]
xTrain = trainMat[,-1]
yTest = testMat[,1]
xTest = testMat[,-1]

ldaTrain <- lda(yTrain~., data = xTrain2)
ldaTrain2 <- predict(ldaTrain, xTrain2)$class
ldaTrain2Int = as.integer(ldaTrain2) - 1

ldaTrainDiff = ldaTrain2Int - yTrain
j = 1
count = 0
while(j <= length(train[,1])){
if(ldaTrainDiff[j] != 0){
	count = count + 1
}
j = j + 1
}
ldaTrainErrorRate = count/length(train[,1])

ldaTest2 <- predict(ldaTrain, xTest2)$class
ldaTest2 = ldaTest2[1:length(yTest)]
ldaTest2Int = as.integer(ldaTest2) - 1
ldaTestDiff = ldaTest2Int - yTest

j = 1
hecount = 0
while(j <= length(test[,1])){
if(ldaTestDiff[j] != 0){
	count = count + 1
}
j = j + 1
}
ldaTestErrorRate = count/length(test[,1])

#QDA (Run PCA first)

xpc <- prcomp(xTrain2)
xPC = xTrain %*% xpc$rotation
xPCred = as.data.frame(xPC[,1:50])
qdaTrain <- qda(yTrain~., data = xPCred)
qdaTrain2 <- predict(qdaTrain, xPCred)$class

j = 1
count = 0
while(j <= length(train[,1])){
if(qdaTrain2[j] != yTrain[j]){
	count = count + 1
}
j = j + 1
}
qdaTrainErrorRate = count/length(train[,1])

xtestpc <- prcomp(xTest2)
xtestPC = xTest %*% xpc$rotation
xtestPCred = as.data.frame(xtestPC[,1:50])
qdaTest2 <- predict(qdaTrain, xtestPCred)$class
j = 1
count = 0
while(j <= length(test[,1])){
if(qdaTest2[j] != yTest[j]){
	count = count + 1
}
j = j + 1
}
qdaTestErrorRate = count/length(test[,1])


#Reduced-Rank LDA
#cross-validation procedure: divide the training data into 10 sets (use one as "test", 9 as train).  compute the optimal dimension for each one, and the resultant error; of all these, then, pick the dimension that gives the lowest.
#Since these are finite discrete variables, we use misclassification rate rather than mean square error to assess cross-validation


indicatorMatrix = mat.or.vec(length(yTrain), 10)

for(i in 1:length(yTrain)){
	k = yTrain[i]
	indicatorMatrix[i,k+1] = 1
}

Yhat = xTrain%*%solve(t(xTrain)%*%xTrain)%*%t(xTrain)%*%indicatorMatrix

singval = svd(centroidsCV)
V = singval$v
U = singval$u
diag = mat.or.vec(10,10)
diag <- sv$d

#Loop over number of dimensions; find misclassification rate.
numfolds = 10
subsetsize = floor(nrow(xTrain)/numfolds)
errorRates = mat.or.vec(10, 11) 
k = 1
while(k <= 10){
	i = 1
	while(i <= numfolds){
		lower = 1 + (i-1)*subsetsize
		upper = i*subsetsize
		trainSub = train[-(lower:upper),]
		testSub = train[lower:upper,]
		testSet = xTrain2[lower:upper,]
		trainSet = xTrain2[-(lower:upper),]
		ytrainCV = yTrain[-(lower:upper)]
		ytestCV = yTrain[lower:upper]

		indicatorMatrixTrain = mat.or.vec(length(ytrainCV), 10)
		indicatorMatrixTest = mat.or.vec(length(ytestCV), 10)

		for(l in 1:length(ytrainCV)){
			a = ytrainCV[l]
			indicatorMatrixTrain[l,a+1] = 1
		}

		for(L in 1:length(ytestCV)){
			a = ytestCV[L]
			indicatorMatrixTrain[L, a + 1] = 1
		}

		YhatTrain = as.matrix(trainSet)%*%solve(t(as.matrix(trainSet))%*%as.matrix(trainSet))%*%t(as.matrix(trainSet))%*%indicatorMatrixTrain
		YhatTest = as.matrix(testSet)%*%solve(t(as.matrix(trainSet))%*%as.matrix(trainSet))%*%t(as.matrix(trainSet))%*%indicatorMatrixTrain
		YhatYtrain = t(YhatTrain)%*%indicatorMatrixTrain
		YhatYtest = t(YhatTest)%*%indicatorMatrixTest
		eigendecTrain <- prcomp(YhatYtrain)
		eigenvecTrain = (eigendecTrain$rotation)[,1:k]
		Ytrans = YhatTrain%*%eigenvecTrain 
		YtransTest = YhatTest %*% eigenvecTrain
		ldaCV <- lda(ytrainCV~., data = as.data.frame(Ytrans))
		ldaCV2 <- predict(ldaCV, as.data.frame(YtransTest))$class

		j = 1
		count = 0
		while(j < length(testSet[,1])){
			if(ldaCV2[j] != ytestCV[j]){
				count = count + 1
			}
			j = j + 1
		}

		errRate = count/length(testSet[,1])
		errorRates[k,i] <- errRate
		i = i + 1
	}

	k = k + 1
}

k = 1
while(k <= 10){
	errorRates[k,11] = sum(errorRates[k,1:10])/10
	k = k + 1
}


#First compute our eigenvectors:


indicatorMatrix = mat.or.vec(length(yTrain), 10)

for(j in 1:length(yTrain)){
	a = yTrain[j]
	indicatorMatrixTrain[j,a+1] = 1
}

Yhat = xTrain%*%solve(t(xTrain)%*%xTrain)%*%t(xTrain)%*%indicatorMatrix
YhatTest = xTest%*%solve(t(xTrain)%*%xTrain)%*%t(xTrain)%*%indicatorMatrix
YhatY = t(Yhat)%*%indicatorMatrix
eigendec <- prcomp(YhatY)
eigenvec = (eigendecTrain$rotation)[,1:9]
Ytrans = Yhat%*%eigenvecTrain 
YtransTest = YhatTest %*% eigenvecTrain

ldaRed <- lda(yTrain~., data = as.data.frame(Ytrans))
ldaRedTrain <- predict(ldaRed, as.data.frame(Ytrans))$class
ldaRedTest <- predict(ldaRed, as.data.frame(YtransTest))$class

j = 1
count = 0
while(j <= length(train[,1])){
if(ldaRedTrain[j] != yTrain[j]){
	count = count + 1
}
j = j + 1
}
ldaRedTrainErrorRate = count/length(train[,1])

j = 1
count = 0
while(j <= length(test[,1])){
if(ldaRedTest[j] != yTest[j]){
	count = count + 1
}
j = j + 1
}
ldaRedTestErrorRate = count/length(test[,1])



#Logistic Regression
#We load the Design package

lrmTrain <- multinom(as.factor(yTrain)~., data = xTrain2, MaxNWts = 3000)
lrmTrainPred <- predict(lrmTrain, xTrain2)

j = 1
count = 0
while(j <= length(train[,1])){
if(lrmTrainPred[j] != yTrain[j]){
	count = count + 1
}
j = j + 1
}
lrmTrainErrorRate = count/length(train[,1])

lrmTestPred <- predict(lrmTrain, xTest2)

j = 1
count = 0
while(j <= length(test[,1])){
if(lrmTestPred[j] != yTest[j]){
	count = count + 1
}
j = j + 1
}
lrmTestErrorRate = count/length(test[,1])



#--------------------------------------------------------------------------------------

#PROBLEM 3
phoneme = read.table("C:/Users/Aneesh/Documents/Documents/STAT 665 (S 10)/phoneme.txt", header = TRUE)

#Part (a)
#SUBSET (aa and ao only)
phonemeSub = phoneme[which(phoneme[,257] == 'aa' | phoneme[,257] == 'ao'),]
XtestSub = phonemeSub[1:858,1:256]
XtrainSub = phonemeSub[859:1717, 1:256]
YtestSub = phonemeSub[1:858, 257]
YtrainSub = phonemeSub[859:1717, 257]
rawLogisticSub <- lrm(YtrainSub ~., XtrainSub)
rawLogisticSubTrain <- predict.lrm(rawLogisticSub, XtrainSub, type = "fitted.ind")
rawLogisticSubTest <- predict.lrm(rawLogisticSub, XtestSub, type = "fitted.ind")

rawLogSub <- multinom(YtrainSub ~., data = XtrainSub)
rawLogTrainSub <- predict(rawLogSub, XtrainSub)
rawLogTestSub <- predict(rawLogSub, XtestSub)

count = 0
j = 1
while(j <= 859){
	if(as.integer(rawLogTrainSub[j]) != as.integer(YtrainSub[j])){
		count = count + 1
	}
	j = j + 1
}
rawLogSubTrainError = count/859

count = 0
j = 1
while(j <= 858){
	if(as.integer(rawLogTestSub[j]) != as.integer(YtestSub[j])){
		count = count + 1
	}
	j = j + 1
}
rawLogSubTestError = count/858

#Note that the errors aren't exactly what's given in the book.  But they're close enough, and it's likely due to choosing different training/test sets.

#Regularized logistic regression.

XtrainMatSub = as.matrix(XtrainSub)
xTransf = mat.or.vec(nrow(XtrainMatSub), 13) #we fill this matrix with transformed values
i = 1
while(i <= nrow(XtrainMatSub)){
	splin <- ns(XtrainMatSub[i,], df = 13) #df = 13 uniformly spaces the basis functions
	transformed = t(splin) %*% (XtrainMatSub[i,])
	xTransf[i,] <- transformed
	i = i + 1
}
#xTransf are our "transformed" values. We can now perform logistic regression on these.

XtestMatSub = as.matrix(XtestSub)
xTransfTest = mat.or.vec(nrow(XtestMatSub), 13)
i = 1
while(i <= nrow(XtestMatSub)){
	splin <- ns(XtestMatSub[i,], df = 13)
	transformed = t(splin) %*% (XtestMatSub[i,])
	xTransfTest[i,] <- transformed
	i = i + 1 
}

lrmTransfSub <- multinom(YtrainSub ~., data = as.data.frame(xTransf))
lrmTrainPredSub <- predict(lrmTransfSub, xTransf)
lrmTestPredSub <- predict(lrmTransfSub, xTransfTest)

count = 0
j = 1
while(j <= 859){
	if(as.integer(lrmTrainPredSub[j]) != as.integer(YtrainSub[j])){
		count = count + 1
	}
	j = j + 1
}
regLogisticSubTrainError = count/859

count = 0
j = 1
while(j <= 858){
	if(as.integer(lrmTestPredSub[j]) != as.integer(YtestSub[j])){
		count = count + 1
	}
	j = j + 1
}
regLogisticSubTestError = count/858


#WHOLE DATA SET
#Raw logistic regression.
#We split the data into training and test sets.  Plotting verifies that splitting by index is okay (i.e. there are no patterns as to how the data is arranged)
Xtrain = phoneme[1:2255,1:256]
Xtest = phoneme[2256:4509,1:256]
Ytrain = phoneme[1:2255, 257]
Ytest = phoneme[2256:4509, 257]

rawLog <- multinom(Ytrain ~., data = Xtrain, MaxNWts = 2000)
rawLogisticTrain <- predict(rawLog, Xtrain)
rawLogisticTest <- predict(rawLog, Xtest)

count = 0
j = 1
while(j <= 2255){
	if(rawLogisticTrain[j] != Ytrain[j]){
		count = count + 1
	}
	j = j + 1
}

rawLogisticTrainError = count/2255

count2 = 0
j = 1
while(j <= 2254){
	if(rawLogisticTest[j] != Ytest[j]){
		count = count + 1
	}
	j = j + 1
}

rawLogisticTestError = count/2254


#Regularized logistic regression

XtrainMat = as.matrix(Xtrain)

#regularization of training data
xTrainReg = mat.or.vec(nrow(XtrainMat), 13) #we fill this matrix with transformed values
i = 1
while(i <= nrow(XtrainMat)){
	splin <- ns(XtrainMat[i,], df = 13) #df = 13 uniformly spaces 12 basis functions
	transformed = t(splin) %*% (XtrainMat[i,])
	xTrainReg[i,] <- transformed
	i = i + 1
}

#regularization of testing data
XtestMat = as.matrix(Xtest)
xTestReg = mat.or.vec(nrow(XtestMat), 13)
i = 1
while(i <= nrow(XtestMat)){
	splin <- ns(XtestMat[i,], df = 13)
	transformed = t(splin) %*% (XtestMat[i,])
	xTestReg[i,] <- transformed
	i = i + 1 
}


regLogFull <- multinom(Ytrain ~., data = as.data.frame(xTrainReg))
regLogTrain <- predict(regLogFull, as.data.frame(xTrainReg))
regLogTest <- predict(regLogFull, as.data.frame(xTestReg))

count = 0
j = 1
while(j <= 2255){
	if(regLogTrain[j] != Ytrain[j]){
		count = count + 1
	}
	j = j + 1
}

count2 = 0
j = 1
while(j <= 2254){
	if(regLogTest[j] != Ytest[j]){
		count2 = count2 + 1
	}
	j = j + 1
}

regLogisticTrainError = count/2255
regLogisticTestError = count2/2254



#Part (b)
#Raw QDA
phonemeQDAraw <- qda(Ytrain ~., data = as.data.frame(Xtrain))
phonemeQDArawTrain <- predict(phonemeQDAraw, Xtrain)$class
phonemeQDArawTest <- predict(phonemeQDAraw, as.data.frame(Xtest))$class

j = 1
count = 0
while(j <= length(phonemeQDArawTrain)){
	if(phonemeQDArawTrain[j] != Ytrain[j]){
		count = count + 1
	}
	j = j + 1
}
phonemeQDArawTrainError = count/length(phonemeQDArawTrain)

j = 1
count = 0
while(j <= length(phonemeQDArawTest)){
	if(phonemeQDArawTest[j] != Ytest[j]){
		count = count + 1
	}
	j = j + 1
}
phonemeQDArawTestError = count/length(phonemeQDArawTest)



#Regularized QDA
#In each case, we'll uniformly space the knots.  We use as our choices 6, 7, 8, 9, and 10 knots.

#Cross-validate to pick the number of knots.

errorRates = mat.or.vec(5,11)
for(k in 6:10){
	numfolds = 10
	subsetsize = floor(nrow(XtrainSet)/numfolds)
	i = 1
	while(i <= numfolds){
		lower = 1 + (i-1)*subsetsize
		upper = i*subsetsize
		XtestSet = Xtest[lower:upper,]
		XtrainSet = Xtrain[-(lower:upper),]
		YtrainSet = Ytrain[-(lower:upper)]
		YtestSet = Ytest[lower:upper]
		XtrainMat = as.matrix(XtrainSet)
		XtestMat = as.matrix(XtestSet)


		XTRtrain = mat.or.vec(nrow(XtrainSet), k)
		XTRtest = mat.or.vec(nrow(XtestSet), k)		
		n = 1
		while(n <= nrow(XtrainSet)){
			splin <- ns(XtrainMat[n,], df = k)
			trans = t(splin) %*% (XtrainMat[n,])
			XTRtrain[n,] <- trans
			n = n + 1 
		}

		m = 1
		while(m <= nrow(XtestSet)){
			splin <- ns(XtestMat[m,], df = k)
			trans = t(splin) %*% (XtestMat[m,])
			XTRtest[m,] <- trans
			m = m + 1 
		}


		qdacv <- qda(YtrainSet ~., data = as.data.frame(XTRtrain))
		qdacvTest <- predict(qdacv, as.data.frame(XTRtest))$class

		j = 1
		count = 0
		while(j <= length(qdacvTest)){
			if(qdacvTest[j] != YtestSet[j]){
				count = count + 1
			}
			j = j + 1
		}

		errRate = count/length(YtestSet)
		errorRates[k-5,i] <- errRate
		i = i + 1
	}
}

for(k in 1:5){
	errorRates[k,11] = sum(errorRates[k,1:10])/10
}

#After picking number of knots which minimizes error (8, since df = 9), we now simply run QDA. 

#regularization of training data
XtrainMat = as.matrix(Xtrain)
xTrainReg = mat.or.vec(nrow(XtrainMat), 9) #we fill this matrix with transformed values
i = 1
while(i <= nrow(XtrainMat)){
	splin <- ns(XtrainMat[i,], df = 9) 
	transformed = t(splin) %*% (XtrainMat[i,])
	xTrainReg[i,] <- transformed
	i = i + 1
}

#regularization of testing data
XtestMat = as.matrix(Xtest)
xTestReg = mat.or.vec(nrow(XtestMat), 9)
i = 1
while(i <= nrow(XtestMat)){
	splin <- ns(XtestMat[i,], df = 9)
	transformed = t(splin) %*% (XtestMat[i,])
	xTestReg[i,] <- transformed
	i = i + 1 
}

phonemeQDAtrain <- qda(Ytrain ~., data = as.data.frame(xTrainReg))
phonemeQDAtrain2 <- predict(phonemeQDAtrain, as.data.frame(xTrainReg))$class
phonemeQDAtest <- predict(phonemeQDAtrain, as.data.frame(xTestReg))$class

count = 0
j = 1
while(j <= 2254){
	if(phonemeQDAtest[j] != Ytest[j]){
		count = count + 1
	}
	j = j + 1
}

count2 = 0
j = 1
while(j <= 2255){
	if(phonemeQDAtrain2[j] != Ytrain[j]){
		count2 = count2 + 1
	}
	j = j + 1
}

phonemeQDAtrainError = count2/2255
phonemeQDAtestError = count/2254
