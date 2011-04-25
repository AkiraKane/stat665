library(rpart)
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
#plot(X2 ~ X1, data=xy,col=y+1)
y.tr <- xy[,p+1]
y.test <- xy.test[,p+1]
# Adaboost M1
w <- rep(1/1000, 1000)
ghat <- rep(0,n) #classification result from comittee
ghat.test <- rep(0,n)
M <- 100 # number of iterations
# to store the results after each iteration
w.m <- matrix(0,M,n)
misInd.m <- matrix(0,M,n)#For Individual Classifier
misInd.m.test <- matrix(0,M,n)#For Individual Classifier
misInd1.m <- matrix(0,M,n)#For Committee
misInd1.m.test <- matrix(0,M,n)#For Committee
err.v <- rep(0,M)# For Individual Classifier
alpha.v <- rep(0,M)
i <- 1
for( i in 1:M)
  {
    w.m[i,] <- w
    tr <- rpart(Y~., data=xy, weights=w, method="class",maxdepth=1)
    phat<-predict(tr,xy)

    yhat <- ifelse(phat[,1]>.5, 0, 1)
    misInd <- ifelse(yhat+y.tr==1,1,0) #Misclassification Indicator of an individual classifier
    err <- sum(w*misInd)/sum(w)
    alpha <- log((1-err)/err)
    w <- w*exp(alpha*misInd)
    ghat <- ghat + alpha*(yhat-1/2)
    yhat1 <- ifelse(ghat>0,1,0)
    misInd1 <- ifelse(yhat1+y.tr==1,1,0) #Misclassification Indicator of the committee

    misInd.m[i,] <- misInd
    misInd1.m[i,]<- misInd1
    err.v[i] <- err
    alpha.v[i] <- alpha

    phat.test <- predict(tr,xy.test)
    yhat.test <- ifelse(phat.test[,1]>.5, 0, 1)
    misInd.test <- ifelse(yhat.test+y.test==1,1,0)
    ghat.test <- ghat.test + alpha*(yhat.test-1/2)
    yhat1.test <- ifelse(ghat.test>0,1,0)
    misInd1.test <- ifelse(yhat1.test+y.test==1,1,0) #Misclassification Indicator of the committee
    misInd.m.test[i,] <- misInd.test
    misInd1.m.test[i,]<- misInd1.test
  

#    par(mfrow=c(1,2))
#        plot(X2~X1,data=xy, col=misInd+1,pch=16)
#        plot(X2~X1,data=xy, col=misInd1+1,pch=16)
#    par(mfrow=c(1,1))
#    Sys.sleep(3)
   }
#plot(x[,7], x[,1],col=y+1)
#abline(v=tr$split[1,4)]

w.sum <- apply(w.m,1,sum)
w.m1 <- w.m/(w.sum%*%t(rep(1,n)))

par(mfrow=c(4,2),mar=c(3,3,1,1),mgp=c(2,1,0))
for(i in 1:4)
  {
    plot(xy$X9,w.m1[i,]+rnorm(n,sd=.00001),col=misInd.m[i,]+1,ylim=range(w.m1[1:4,]),ylab="weight",main=paste("iter",i))
    plot(xy$X9,w.m1[i+1,]+rnorm(n,sd=.00001),col=misInd.m[i,]+1,ylim=range(w.m1[1:4,]),ylab="")
  }
par(mfrow=c(1,1))

par(mfrow=c(2,2))
error <- apply(misInd.m, 1, sum)
plot(error/n, type="l",ylab="training error", ylim=c(0.4,0.55),main="Individual Classifier")
error1 <- apply(misInd1.m, 1, sum)
plot(error1/n, type="l",ylab="training error", ylim=c(0.1,0.5),main="Committee")

error <- apply(misInd.m.test, 1, sum)
plot(error/n, type="l",ylab="test error",ylim=c(0.4,0.55),main="Individual Classifier")
error1 <- apply(misInd1.m.test, 1, sum)
plot(error1/n, type="l",ylab="test error",ylim=c(0.1,0.5),main="Committee")
par(mfrow=c(1,1))
