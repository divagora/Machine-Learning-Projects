#### GAUSSIAN PROCESS REGRESSION ####

####load packages###
#install/load required packages
require(MASS)
#install.packages("reshape2")
require(reshape2)
#install.packages("plyr")
require(plyr)
#install.packages("ggplot2")
require(ggplot2)
#install.packages("pracma")
require(pracma)
#install.packages("psych")
require(psych)
#install.packages("Rcgmin")
require(Rcgmin)

#####load and format CO2 data###
require(readr)
data <- read_csv("~/path/to/data")
data <- cbind(unlist(data[,1][[1]]),unlist(data[,2][[1]]))

#define training set of time inputs#
x <- data[,1]
year <- 1960+(x-x[21])/12
x <- x/12 #scale to years

#define observations corresponding to training inputs
y <- data[,2]

#plot data
#require(tikzDevice)
#tikz("co2train.tex",width=9,height=5,standAlone = T)
plot(x,y,type="p",xaxt="n",xlab="Year",ylab="CO2 Concentration",xlim=c(-2,36),cex=.5,pch=16,col="blue")
axis(1, at=(-2:36), labels=1958:1996)
#dev.off()
#tools::texi2dvi("co2train.tex",pdf=T)

#create prediction set of time inputs (i.e. 'next twenty years')
xstar <- c()
for (i in 1:(20*12)){
  xstar <- c(xstar,x[length(x)]+i/12)
}

#create 'years' vector
yearstar <- 1960+(xstar-x[21])
allyears <- c(year,yearstar)

####define covariance and mean functions###
#for long term smooth rising trend:
#theta_1=amplitude, theta_2=characteristic length scale
k1 <- function(r,theta){
  theta[1]^2*exp(-0.5*(r)^2/(theta[2]^2))
}

#for short term seasonal variation:
#theta_3 = magnitude, theta_4=decay-time of periodic component, theta_5=smoothness of periodic...
k2 <- function(r,theta){
  theta[1]^2*exp(-0.5*(r)^2/(theta[2]^2)-2*sin(pi*(r))^2/(theta[3]^2))
}

#for (small) medium-term irregularities:
#theta_6=magnitude, theta_7=typical length scale, theta_8=shape parameter
k3 <- function(r,theta){
  theta[1]^2*(1+0.5*(r)^2/(theta[3]*theta[2]^2))^(-theta[3])
}

#time dependent noise model
k4 <- function(r,theta){
  theta[1]^2*exp(-0.5*(x-xdash)^2/(theta[2]^2))
}
  
#independent noise 
k5 <- function(r,theta){
  theta[1]^2*ifelse(r==0,1,0)
} 

#linear mean
m1 <- function(x,theta,length_m=2){
  theta[1]+theta[2]*x
}

#use fixed mean function
m1fixed <- function(x,length_m=2){
  313.268+1.279*x
}

#quadratic mean
m2 <- function(x,theta,length_m=3){
  theta[1]+theta[2]*x+theta[3]*x^2
}


####optimisation of hyperparameters###
####functions to compute covariance, predictions, log-like###
#compute covariance matrix
comp_cov_matrix <- function(x1,x2,theta,kernel) {
  x1_matrix <- matrix(x1,nrow=length(x1),ncol=length(x1))
  x2_matrix <- matrix(x2,nrow=length(x2),ncol=length(x2),byrow=TRUE)
  cov_matrix <- kernel(x1_matrix-x2_matrix,theta)
  return(cov_matrix)
}

#compute marginal log-likelihood
#function inputs are X (matrix of observations), y (targets), k (cov function), sigma_n2 (noise), xstar (test input)
#optionally also compute prediction mean and covariance

predictions <- function(theta,x,y,xstar,kernel,mu,length_m){
  xfull=c(x,xstar)
  KXplusXstar <- comp_cov_matrix(xfull,xfull,theta[1:(length(theta)-(length_m+1))],kernel)
  KXX <- KXplusXstar[1:length(x),1:length(x)]
  KXXstar <- KXplusXstar[1:length(x),(length(x)+1):(length(x)+length(xstar))]
  KXstarX <- KXplusXstar[(length(x)+1):(length(x)+length(xstar)),1:length(x)]
  KXstarXstar <- KXplusXstar[(length(x)+1):(length(x)+length(xstar)),(length(x)+1):(length(x)+length(xstar))]
  
  C <- KXX+comp_cov_matrix(x,x,theta[length(theta)-length_m],k5)
  C_inverse <- solve(C)
  
  fstar_mean <- KXstarX%*%C_inverse%*%(y-mu(x,theta[(length(theta)-(length_m-1)):length(theta)]))+mu(xstar,theta[(length(theta)-(length_m-1)):length(theta)])
  fstar_cov <- KXstarXstar-KXstarX%*%C_inverse%*%KXXstar
  
  log_lik <- -0.5*t(y-mu(x,theta[(length(theta)-(length_m-1)):length(theta)]))%*%C_inverse%*%(y-mu(x,theta[(length(theta)-(length_m-1)):length(theta)]))-0.5*log(det(C))-0.5*length(x)*log(2*pi)
  
  list(mean=fstar_mean,cov=fstar_cov,neg_log_lik=-log_lik)
}

pred_neg_log_like <- function(theta,x,y,kernel,mu,length_m) {
  KXX <- comp_cov_matrix(x,x,theta[1:(length(theta)-(length_m+1))],kernel)
  C <- KXX+comp_cov_matrix(x,x,theta[length(theta)-length_m],k5)
  C_inverse <- solve(C)
  
  log_lik <- -0.5*t(y-mu(x,theta[(length(theta)-(length_m-1)):(length(theta))]))%*%C_inverse%*%(y-mu(x,theta[(length(theta)-(length_m-1)):(length(theta))]))-0.5*log(det(C))-0.5*length(x)*log(2*pi)
  print(theta)
  return(-log_lik)
}

####optimise hyperparameters###

#optimise just with first kernel - observe that characteristic length scale of covariance is not sufficient to capture increasing trend in data
theta.initial <- c(50,50,0.5)
theta <- optim(par=theta.initial,fn=pred_neg_log_like,control=list(trace=TRUE,maxit=5000,parscale=theta.initial),x=x,y=y-mean(y),kernel=function(r,theta) k1(r,theta[1:2]),mu=function(r,theta,length_m) 0,length_m=0)
pred <- predictions(theta$par,x,y,xstar,kernel=function(r,theta) k1(r,theta[1:2]),mu=function(r,theta,length_m) 0,length_m=0)

#optimise just on second kernel
theta.initial1 <- c(1,50,1,0.5)
theta1 <- optim(par=theta.initial1,fn=pred_neg_log_like,control=list(trace=TRUE,maxit=5000,parscale=theta.initial1),x=x,y=y-mean(y),kernel=function(r,theta) k2(r,theta[1:3]),mu=function(r,theta,length_m) 0,length_m=0)
pred1 <- predictions(theta1$par,x,y,xstar,kernel=function(r,theta) k2(r,theta[1:3]),mu=function(r,theta,length_m) 0,length_m=0)

#optimise over both kernels, using the intiial parameters determined via these runs
theta.initial2 <- c(theta$par[1:2],theta1$par)
theta2 <- optim(par=theta.initial2,fn=pred_neg_log_like,control=list(trace=TRUE,maxit=5000,parscale=theta.initial2),x=x,y=y-mean(y),kernel=function(r,theta) k1(r,theta[1:2])+k2(r,theta[3:5]),mu=function(r,theta,length_m) 0,length_m=0)
pred2 <- predictions(theta2$par,x,y,xstar,kernel=function(r,theta) k1(r,theta[1:2])+k2(r,theta[3:5]),mu=function(r,theta,length_m) 0,length_m=0)

#add mean function
theta.initial3 <- c(theta2$par,315,1)
theta3 <- optim(par=theta.initial3,fn=pred_neg_log_like,control=list(trace=TRUE,maxit=5000),x=x,y=y-mean(y),kernel=function(r,theta) k1(r,theta[1:2])+k2(r,theta[3:5]),mu=m1,length_m=2)
pred3 <- predictions(theta3$par,x,y,xstar,kernel=function(r,theta) k1(r,theta[1:2])+k2(r,theta[3:5]),mu=m1,length_m=2)

####
#now optimise also with second kernel function
#reset noise parameter
theta.initial1 <- c(theta$par[1:2],1,1,1,.5)
theta1 <- optim(par=theta.initial1,fn=pred_neg_log_like,control=list(trace=TRUE,maxit=5000),x=x,y=y-mean(y),kernel=function(r,theta) k1(r,theta[1:2])+k2(r,theta[3:5]),mu=m1fixed,length_m=0)
pred1 <- predictions(theta1$par,x,y,xstar,kernel=function(r,theta) k1(r,theta[1:2])+k2(r,theta[3:5]),mu=m1fixed,length_m=0)


####plot predictions###
set.seed(1)
n_samples <- 100
ystar <- matrix(0,nrow=length(xstar),ncol=n_samples)
for (i in 1:n_samples) {
  ystar[,i] <- mvrnorm(1, pred3[[1]], pred3[[2]])
}


#require(tikzDevice)
#tikz("2c3.tex",width=9,height=5,standAlone = T)
plot(x,y,type="p",xaxt="n",col="blue",xlab="Year", ylab="CO2 Concentration (ppm)",ylim=c(315,410),xlim=c(-1,55),pch=16,cex=.5)
axis(1, at=(-2:56), labels=1958:2016)
for (i in 2:n_samples){
  points(xstar,ystar[,i],lty=1,col="lightgrey",type="l",cex=0.5,pch=16)
}
points(xstar,colMeans(t(ystar)),col="red",cex=0.5,pch=16,type="p")
#dev.off()
#tools::texi2pdf("2c3.tex",clean=T)

