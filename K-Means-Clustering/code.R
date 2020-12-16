#### AUTOMATIC IMAGE SEGMENTATION AND CELL COUNTING ####

#### Question 1 ####

#install.packages("jpeg")
library(jpeg)

#load image
img <- readJPEG("~path/to/FluorescentCells.jpg")

#plot original image
if(FALSE){
require(tikzDevice)
tikz("q1_img.tex",standAlone = T, width=5, height=5)
plot(c(0, 512), c(0, 512), type = "n", xlab = "", ylab = "", xaxs="i",yaxs="i",xaxt="n",yaxt="n")
rasterImage(img, 0, 0, 512, 512)
dev.off()
tools::texi2dvi("q1_img.tex",pdf=T)
}

###K MEANS###

#install 'fields package'
#contains rdist function for computing distance between rows of two matrices
#install.packages("fields")
library(fields)

#convert array into matrix, each row is an intensity vector, order column by column of original
img.matrix <- matrix(img,ncol=dim(img)[3]) 

#inputs are x (img matrix ), k (number of clusters), max.iterations (max number of iterations)
k_means <- function(x, k, max.iterations=1000) {
  clusterOld <- list()
  centerOld <- list()
  
  set.seed(1)
  centers <- x[sample(nrow(x),k),] #randomly intialise cluster means
  centerOld[[1]] <- centers
  
  for(i in 2:(max.iterations+1)) {
    #compute distances between center of clusters and all data points
    dist.centers <- rdist(x,centers) #matrix with (i,j)^th element the dist. between ith row of x and j^th row of centers
    
    clusters <- apply(dist.centers, 1, which.min) #determine which cluster mean is closest to each point
    clusters.rows <- sapply(1:k,function(j) which(clusters==j)) #i^th vector in list is row indices of those points closest to i^th cluster mean
    
    #new cluster means
    for (j in 1:k){
      if (length(clusters.rows[[j]])==0) centers[j,]=centerOld[[i-1]][j,]
      else {centers[j,] <- colMeans(x[clusters.rows[[j]],])}
    }
    
    # convergence history
    clusterOld[[i]] <- clusters
    centerOld[[i]] <- centers
    
    #print the iteration
    cat("This is iteration", i-1, "\n")
    
    #break before reaching max iterations if all points have converged 
    if (identical(centerOld[[i]],centerOld[[i-1]])) break
  }
  list(clusters, centers)
}

#function to plot output from k means function
k_means_plot <- function(k_means_results) {
  #create vector of means
  #elements 1:(number of pixels) = 1st coordinate of intensity vector, (number of pixels +1):2*(number of pixels)
  #=2nd coordinate of intensity vector etc.
  img_vec <- sapply(1:length((k_means_results)[[1]]),function(j) k_means_results[[2]][k_means_results[[1]][j],])
  #reconstruct array, as per format of original img
  img_final <- array(c(img_vec[seq(1,3*640*640-2,3)],img_vec[seq(2,3*640*640-1,3)],
                       img_vec[seq(3,3*640*640,3)]),dim=c(640,640,3))
  plot(c(0, 512), c(0, 512), type = "n", xlab = "", ylab = "", xaxs="i",yaxs="i",xaxt="n",yaxt="n")
  rasterImage(img_final, 0, 0, 512, 512)
}

#run for several values of k
k_values <- c(2:5,7,10,20,50); counter <- 1
for (i in k_values){
  require(tikzDevice)
  filename <- paste("q1_kmeans_",k_values[counter],".tex",sep="")
  tikz(filename,standAlone = T, width=5,height=5)
  k_means_result <- k_means(img.matrix,k_values[counter])
  k_means_plot(k_means_result)
  dev.off()
  #tools::texi2dvi(filename,pdf=T)
  counter <- counter+1
}


###MIXTURE MODELS###

require(MASS); require(mvtnorm)

#will write function for each step of the algorithm separately

#X is matrix of intensity vectors for all pixels, theta = params, k = no of components
expectation <- function(X, theta, k) {
  #compute cluster probabilities
  prob_m_x <- sapply(1:k, function(i) theta$prob_m[i]*dmvnorm(X, theta$mu[[i]], theta$sig[[i]]))
  for (j in 1:ncol(prob_m_x)){
    #if all probs for particular mode are zero, reset parameters
    while (sum(prob_m_x[,j])==0 || is.nan(sum(prob_m_x[,j])) || sum(prob_m_x[,j])==Inf){  
       theta$mu[[j]] <- X[sample(1),]
       theta$sig[[j]] <- cov.wt(X[sample(nrow(X),100),])$cov
       prob_m_x <- sapply(1:k, function(i) theta$prob_m[i]*dmvnorm(X, theta$mu[[i]], theta$sig[[i]]))
    }
  }
  prob_m_x/rowSums(prob_m_x) #normalise
}

maximisation <- function(X, prob_m_x, k) {
  covs <- lapply(1:k, function(i) cov.wt(X, prob_m_x[,i])) #apply cov.wt to each column of prob(m|x_n)
  mu <- lapply(covs, "[[", "center") #specifiying [[center]] returns est. mean
  sig <- lapply(covs, "[[", "cov") #specifying [[covs]] returns est. covariance matrix
  prob_m <- colMeans(prob_m_x) #p(m), average weighted cluster membership size
  list(mu = mu, sig = sig, prob_m = prob_m)
}

# compute log-likelihood of model, given points and parameters
# termination condition for EM & to compute AIC for model selection (i.e. choice of k)
log.like <- function(X, theta, k) {
  probs <- sapply(1:k, function(i) theta$prob_m[i] * dmvnorm(X, theta$mu[[i]], theta$sig[[i]]))
  sum(log(rowSums(probs)))
}

#EM algorithm: inputs are X (data as matrix, rows are data points), k (number of components of mixture)
em.algorithm <- function(X, k, max.iter = 1000, tol=1) {
  
  #dimensionality of data
  d <- dim(X)[2]
  N <- dim(X)[1]
  
  #free parameters - f
  #for each Gaussian: d_g = d*(d-1)/2 + 2d + 1 parameters
  #(i) dxd symmetric covariance matrix: (D*D - D)/2 + D free parameters = ((D*D - D)/2 off-diagonal + D diagonal 
  #(ii) d-dim mean vector: d parameters
  #(iii) mixing weight: 1 parameter
  #k models, so k*d_g parameters => k*d_g - 1 free parameters (mixing weights must sum to 1)
  p <- k*(d*(d+3)/2+1)-1 
  
  #initial parameters
  set.seed(1)
  covs <- replicate(k, list(cov.wt(X[sample(nrow(X),100),]))) #for each cluster, take est. weighted covariance of 100 rows worth of the data 
  #mu <- X[sample(dim(X)[1],k),]
  #mu <- lapply(seq_len(nrow(mu)), function(i) X[i,]) #intial means
  mu <- lapply(covs, "[[", "center")
  sig <- lapply(covs, "[[", "cov") #initial covariances
  prob_m <- rep(1/k, k)
  clusters <- rep(0,nrow(X))
  
  theta <<- list(mu = mu, sig = sig, prob_m = prob_m) #all parameters
  
  #iteratively apply EM, up to specified max no. of iterations
  for(i in 1:max.iter) {
    
    #print the iteration
    cat("This is iteration", i, "\n")
    
    theta_old <- theta
    prob_m_x <<- expectation(X, theta, k) #E-step
    theta <<- maximisation(X, prob_m_x, k) #M-step
    clusters <- apply(expectation(X, theta, k), 1, which.max)
    if((log.like(X, theta, k) - log.like(X, theta_old, k)) < tol) #terminate when difference in log-likelihoods below specified tolerance
      break
  }
  return(list(theta = theta, clusters = clusters, bic = p*log(N) - 2*log.like(X, theta, k), aic = 2*p - 2*log.like(X, theta, k))) #return parameters and BIC
}

#function to plot output from em algorithm
em.plot <- function(em.results){
  #store cluster means in matrix
  means <- matrix(0,nrow=length(em.results$theta$mu),ncol=3)
  for (i in 1:length(em.results$theta$mu)){
    means[i,] <- em.results$theta$mu[[i]]
  }
  
  img_vec <- sapply(1:length(em.results$clusters),function(j) means[em.results$clusters[j],])
  #reconstruct array, as per format of original img
  img_final <- array(c(img_vec[seq(1,3*640*640-2,3)],img_vec[seq(2,3*640*640-1,3)],
                       img_vec[seq(3,3*640*640,3)]),dim=c(640,640,3))
  plot(c(0, 512), c(0, 512), type = "n", xlab = "", ylab = "")
  rasterImage(img_final, 0, 0, 512, 512)
}

#run for several values of k
#output files with _2 appended to filename implement algorithm initialising means as M random points 
#output files without _2 implement algorithm intialising as M sample means, from size 100 samples
k_values <- c(2:5,7,10,20,50); counter <- 1
for (i in k_values){
  require(tikzDevice)
  filename <- paste("q1_gmm_2_",k_values[counter],".tex",sep="")
  tikz(filename,standAlone = T, width=5,height=5)
  em.results <- em.algorithm(img.matrix,i)
  em.plot(em.results)
  dev.off()
  #tools::texi2dvi(filename,pdf=T)
  counter <- counter+1
  print(k_values[counter])
}


##RUN ALGORITHM TO COUNT NUMBER OF NUCLEUSES##

#function to reformat data into 2d position vector of only blue points,
counting_reformat <- function(k_means_input) {
  #determine which clusters correspond to blue regions in image (i.e. the nuclei)
  cluster.nucleus <- which(k_means_input[[2]][,3]==max(k_means_input[[2]][,3]))
  
  #reformat data: set equal to (0,0,0) if in cluster.nucleus, set equal to (1,1,1) otherwise
  k_means_bw_centers <- matrix(1,nrow=dim(k_means_input[[2]])[1],ncol=dim(k_means_input[[2]])[2])
  k_means_bw_centers[cluster.nucleus,] <- rep(0,dim(k_means_input[[2]])[1])
  k_means_bw <- list(k_means_input[[1]],k_means_bw_centers)
  
  #create vector of the b&w cluster mean of each pixel
  k_means_count <- sapply(1:length((k_means_bw)[[1]]),function(j) k_means_bw[[2]][k_means_bw[[1]][j],])
  
  #reformat as per original array
  k_means_bw_img <- array(c(k_means_count[seq(1,3*640*640-2,3)],k_means_count[seq(2,3*640*640-1,3)],k_means_count[seq(3,3*640*640,3)]),dim=c(640,640,3))
  
  
  counting_input <- which(k_means_bw_img[,,2]==0,arr.ind=T)
  counting_input[,1] <- 640-counting_input[,1]
  counting_input[,1:2] <- counting_input[,2:1]
  counting_input
}

#function thin data to increase speed of algorithm (dealing with large values of k)
thinning <- function(counting_input,thinning_factor){
  counting_input[sample(dim(counting_input)[1],ceiling(thinning_factor*dim(counting_input)[1])),]
}

#run k-means on data (with some value of k)
k_means_counting <- k_means(img.matrix,3)

#reformat k_means_counting into this format
GMM_counting_input <- counting_reformat(k_means_counting)

#apply thinning to original counting input (10% of data)
GMM_counting_input_10_percent <- thinning(GMM_counting_input,0.1)

#run over reasonable range of k values, and compute aic/bic to compare performance
k_values <- seq(20,140,1)
bic.aic <- sapply(k_values,function(i) {cat("k=", i, "\n\n"); em.algorithm(GMM_counting_input_10_percent,i,tol=1)})

#convert into matrix of k-values, bic, aic
bic.aic.matrix <- rbind(k_values,bic.aic[3:4,])
bic.aic.matrix <- rbind(unlist(k_values),unlist(bic.aic.matrix[2,]), unlist(bic.aic.matrix[3,]))

#save data
#write.matrix(bic.aic.matrix,file="")

#determine numbers of clusters according to minimisation of bic or aic
clusters.bic <- as.numeric(bic.aic.matrix[1,which(bic.aic.matrix.nonames[2,]==min(bic.aic.matrix.nonames[2,]))])
clusters.aic <- as.numeric(bic.aic.matrix[1,which(bic.aic.matrix.nonames[3,]==min(bic.aic.matrix.nonames[3,]))])

#column index of cluster count
clusters.bic.index <- which(bic.aic.matrix.nonames[2,]==min(bic.aic.matrix.nonames[2,]))
clusters.aic.index <- which(bic.aic.matrix.nonames[3,]==min(bic.aic.matrix.nonames[3,]))

##PLOT RESULTS
require(tikzDevice)
tikz("aic_bic.tex",standAlone = T, width = 7, height = 5)
plot(bic.aic.matrix[1,],bic.aic.matrix[2,],type="b",col="red",pch=16,ylim=c(102000,113000),
     xlab="$M$",ylab="Information Criterion")
lines(bic.aic.matrix[1,],bic.aic.matrix[3,],type="b",col="blue")

#add dashed lines corresponding to minima
segments(clusters.bic,0,clusters.bic,bic.aic.matrix.nonames[2,clusters.bic.index], lty=2,col="red")
segments(0,bic.aic.matrix.nonames[2,clusters.bic.index],clusters.bic,bic.aic.matrix.nonames[2,clusters.bic.index], lty=2,col="red")
axis(1, tick=T,at=clusters.bic, label=clusters.bic,col.axis="red",col="red",cex.axis=1,cex=1)

segments(clusters.aic,0,clusters.aic,bic.aic.matrix.nonames[3,clusters.aic.index], lty=2,col="blue")
segments(0,bic.aic.matrix.nonames[3,clusters.aic.index],clusters.aic,bic.aic.matrix.nonames[3,clusters.aic.index], lty=2,col="blue")
axis(1, tick=T,at=clusters.aic, label=clusters.aic,col.axis="blue",col="blue",cex.axis=1,cex=1)


#add legend
legend(110,104000,c("BIC","AIC"),ncol=2,lty=c(1,1),pch=c(1,16),col=c("red","blue"))

dev.off()
tools::texi2dvi("aic_bic.tex", pdf=T)