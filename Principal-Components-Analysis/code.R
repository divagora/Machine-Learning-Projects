### PCA & K-Nearest Neighbours ##

#Import data
require(readr)
faces_test_labels <- read_csv("~/path/to/Faces_Test_Labels.csv", col_names = FALSE)
faces_test_inputs <- as.matrix(read_csv("~/path/to/Faces_Test_Inputs.csv", col_names = FALSE))
faces_train_inputs <- as.matrix(read_csv("~/path/to/Faces_Train_Inputs.csv", col_names = FALSE))
faces_train_labels <- read_csv("~/path/to/Faces_Train_Labels.csv", col_names = FALSE)


### Q1 ###

## Compute and Plot Average Face ##

#define function to rotate images so they are correctly displayed by 'image'
rotate <- function(m){
  t(m)[,nrow(m):1]
}

#(optional) define colours for images
shades <- colorRampPalette(c("grey2","grey95"))

#face plotting function
plot.faces <- function(face_vector) {
  face_matrix <- rotate(matrix(as.numeric(face_vector),112,92))
  image(face_matrix,col=gray.colors(20),asp=1,axes=F)#col=shades(50))
}

par(mfrow=c(1,1))
require(tikzDevice)
tikz("face21.tex", standAlone = TRUE)
plot.faces(faces_train_inputs[21,])
dev.off()
tools::texi2dvi('face21.tex',pdf=T)

require(tikzDevice)
tikz("face108.tex", standAlone = TRUE)
plot.faces(faces_train_inputs[108,])
dev.off()
tools::texi2dvi('face108.tex',pdf=T)

require(tikzDevice)
tikz("face319.tex", standAlone = TRUE)
plot.faces(faces_train_inputs[319,])
dev.off()
tools::texi2dvi('face319.tex',pdf=T)

#calculate average face of training set
average_face <- colMeans(faces_train_inputs) 

#plot average face
par(mfrow=c(1,1))
require(tikzDevice)
tikz("average_face.tex", standAlone = TRUE)
plot.faces(average_face)
dev.off()
tools::texi2dvi('average_face.tex',pdf=T)


## Function to Find PCA Basis of Size M ##

#define function to normalise vectors
norm <- function(x) {x / sqrt(sum(x^2))} 


#define PCA basis function
pca <- function(M,X) { #inputs are M (size of basis) and X (matrix containing training set)
  X <- scale(X,center=T,scale=FALSE) #mean center rows of X 
  eigen_basis <- eigen(X%*%t(X))$vectors[,1:M] #compute eigenbasis for XX^T (dim 320x320)
  eigen_basis <- t(X)%*%eigen_basis #recover required basis for X^TX (dim 10304x10304)
  eigen_basis <- apply(eigen_basis,2,norm) #normalise basis
  return(eigen_basis)
}

#compute first five 'eigenfaces'
five_faces <- pca(5,faces_train_inputs) 

#plot first five eigenfaces
par(mfrow=c(2,3))
for (i in 1:5){
  plot.faces(five_faces[,i])
}

### - ###


### Q2 ###

##Project a single face into the PCA basis ##

top_dim <- dim(faces_train_inputs)[1] #max possible no. of eigenfaces (with non-zero eigenvalue)
basis_dim <- c(5,10,50) #dimensions of interest

#compute all eigenfaces
projection.bases <- pca(top_dim,faces_train_inputs) 
V <- as.matrix(projection.bases)

#specify which face to project
face_q2 <- 15

#define matrix of projections 
#ith column to contain projection using PCA basis of dimension i
projection <- matrix(0,ncol=top_dim,nrow=10304)

#similarly define matrix of errors 
#ith column to contain squared difference between dimension i projection and original image
errors.2 <- matrix(0,ncol=top_dim,nrow=10304)


#NB: following loop may be slow if running for all dimensions (1:320)

for (i in 1:top_dim){
  #project into basis space (multiply by A = matrix of M eigenfaces)
  #return to image space (multiply by A^T)
  #chosen face is mean centered before projecting; mean is then re-added
  projection[,i] <- (faces_train_inputs[face_q2,]-average_face)%*%V[,1:i]%*%t(V[,1:i])+average_face
  
  #compute squared errors
  errors.2[,i] <- (projection[,i]-faces_train_inputs[face_q2,])^2 
}

#plot the results in dimensions of interest (5,10,50)
par(mfrow=c(1,3))
for (i in basis_dim){
  plot.faces(projection[,i])
}

### - ###


### Q3 ###

## Plot A Graph of Mean Squared Error, and Discuss Optimal Choice of M ##

#compute MSE using 'errors' matrix
MSE <- colMeans(errors.2) 

#extension: compute MSE for various faces and average MSE

#plot errors for all possible PCA basis dimensions (1,...,320)
par(mfrow=c(1,1))
plot(1:top_dim,MSE[1:top_dim],type="l")
points(basis_dim,MSE[basis_dim]) #highlight points at dimensions of interest

#require(tikzDevice)
#tikz('MSE.tex', standAlone = TRUE, width=5, height=5); par(mar=c(5,5,5,5))
#par(mfrow=c(1,1)); plot(1:top_dim,MSE[1:top_dim],type="p",col="blue",xlab="M",ylab="MSE",pch=16,cex.lab=2,cex.axis=2,cex=.45)
#points(basis_dim,MSE[basis_dim],col="red",pch=16,cex=1.5) #highlight points at dimensions of interest
#abline(a=0.005,b=0,lty=2); abline(a=0.0025,b=0,lty=2); abline(v=c(52,104),lty=c(2,2))
#dev.off()
#tools::texi2dvi('MSE.tex',pdf=T)

#compute finite difference of f(M)=MSE(M) 
MSE_diff <- abs(MSE[2:length(MSE)]-MSE[1:(length(MSE)-1)])
plot(1:(top_dim),MSE_diff[1:(top_dim)],type="l")

#require(tikzDevice)
#tikz('deltaMSE.tex', standAlone = TRUE, width=5, height=5); par(mar=c(6,6,6,6))
#MSE_diff <- abs(MSE[2:length(MSE)]-MSE[1:(length(MSE)-1)])
#plot(1:(top_dim),MSE_diff[1:(top_dim)],type="p",col='blue',xlab="M", ylab="$\\Delta$(MSE)",pch=16,cex.lab=2,cex.axis=2,cex=.75)
#points(basis_dim,MSE_diff[basis_dim],type="p",col='red',pch=16,cex=1.5)
#abline(a=0.0005,b=0,lty=2); abline(a=0.00025,b=0,lty=2); abline(v=c(65,76),lty=c(2,2))
#dev.off()
#tools::texi2dvi('deltaMSE.tex',pdf=T)

#compute proportion of variance explained by dimension M

#PCA eigen-values function
pca_evalues <- function(X) { #input is just X (matrix containing training set)
  X <- scale(X,center=T,scale=FALSE) #mean center rows of X 
  eigen_values <- eigen(X%*%t(X))$values #compute eigenbasis for XX^T (dim 320x320)
  return(eigen_values)
}

#compute all eigenvalues
eigen.values <- pca_evalues(faces_train_inputs)

#proportion of variance = sum of first M eigenvalues / sum of all eigenvalues
prop.of.variance=matrix(c(eigen.values, rep(0,320)),nrow=320,ncol=2)
for (i in 1:320){
  prop.of.variance[i,2] <- sum(prop.of.variance[1:i,1])/sum(prop.of.variance[1:320,1])
}
plot(1:320,prop.of.variance[,2],type="l")

#require(tikzDevice)
#tikz('prop_of_variance.tex', standAlone = TRUE, width=5, height=5); par(mar=rep(6,4))
#plot(1:320,prop.of.variance[,2],type="p",col="blue",xlab="M", ylab="$\\lambda(M)$",pch=46,cex.lab=2,cex.axis=2,cex=1)
#points(basis_dim,prop.of.variance[basis_dim,2],type="p",col="red",pch=16,cex=1.5)
#abline(a=0.8,b=0,lty=2);abline(a=0.9,b=0,lty=2); abline(v=c(41,97),lty=c(2,2))
#dev.off()
#tools::texi2dvi('prop_of_variance.tex',pdf=T)

#determine points which meet chosen thresholds
c(min(which((MSE<0.005) == TRUE)),min(which((MSE<0.0025) == TRUE))) #MSE
c(max(which((MSE_diff>=0.0005) == TRUE))+1,max(which((MSE_diff>=0.00025) == TRUE))+1) #deltaMSE
c(min(which((prop.of.variance[,2]>0.8) == TRUE)),min(which((prop.of.variance[,2]>0.9) == TRUE))) #MSE

### - ###



### Q4 ###

##Function Implementing K-Nearest Neighbour Classifier ##

#NB: we proceed only using M.optimal from Q3

#define distance measure function (Euclidean)
#euclid.distance <- function(x1,x2){ 
#  distance <- sqrt(sum((x1 - x2) ^ 2))
#  return(as.numeric(distance))
#}

#define mode function 
mode <- function(x) {
  unique_val <- unique(x)
  counts <- vector()
  for (i in 1:length(unique_val)) {
    counts[i] <- length(which(x==unique_val[i]))
  }
  position <- c(which(counts==max(counts)))
  mode_x <- unique_val[position]
  return(mode_x)
}

#define classification rate function
class.rate <- function(x1,x2) sum(x1==x2)/length(x1)


##knn function
#standard inputs are faces_test_inputs, faces_train_inputs, faces_train_labels, faces_test_labels
#standard input for basis is V
knn1 <- function(test, train, train.class, test.class, k, m, basis, dist.power){
  prediction <- c()  #empty predictions vector
  
  #project onto m-dim subspace
  test <- (test-colMeans(train))%*%basis[,1:m]
  train <- (train-colMeans(train))%*%basis[,1:m]
  
  #concatenate training labels and inputs
  train <- cbind(t(train.class),train)

  for(i in c(1:nrow(test))){ #loop over rows in test data
    
    #compute dist. between ith test point and all training points
    distances <- as.numeric(apply(train[,2:ncol(train)],1,function(x) as.numeric(dist(rbind(x,test[i,]),method="minkowski",p=dist.power)))) 
    classes <- train[,1] #classes
    
    neighbours <- data.frame(classes, distances) #combine 'distances' and 'classes' 
    neighbours <- neighbours[order(neighbours$distances),] #sort in asc. order of dist.
    neighbours <- neighbours[1:k,] #k-nearest neighbours
    
    class.pred <- mode(neighbours[,"classes"]) #most common class in knn
    
    #while loop to deal with multiple modes
    #decrease value of k until there is a unique mode
    while (length(class.pred)>1) { 
      neighbours <- neighbours[-(dim(neighbours)[1]),]
      class.pred <-  mode(neighbours[,"classes"])
    }
    prediction <- c(prediction,class.pred) #append predictions vector
  }
  
  #return classification rate; alternatively could return actual predictions
  return(class.rate(as.numeric(prediction),test.class)) 
}

#trial with various values of k, M and p
knn_basis_dim=seq(10,320,10)
k_values=c(1:5,10)
p_values=1:3
knn.results <-array(rep(0, length(knn_basis_dim)*length(k_values)*length(p_values)), dim=c(length(knn_basis_dim), length(k_values), length(p_values)))

for (i in 1:length(knn_basis_dim)){
  for (j in 1:length(k_values)){
    for (k in 1:length(p_values)){
      knn.results[i,j,k]=knn1(faces_test_inputs,faces_train_inputs,faces_train_labels,faces_test_labels,k_values[j],knn_basis_dim[i],V,p_values[k])
    }
  }
}

knn.results.names <- c("~/Documents/Imperial/Spring Term/Machine Learning (M5MS10)/Coursework/Coursework 1/knn_results1",
                       "~/Documents/Imperial/Spring Term/Machine Learning (M5MS10)/Coursework/Coursework 1/knn_results2",
                       "~/Documents/Imperial/Spring Term/Machine Learning (M5MS10)/Coursework/Coursework 1/knn_results3")
for (i in 1:length(p_values)){
  write.table(round(t(knn.results[,,i]),2),file=knn.results.names[i],sep="&")
}

knn_basis_dim_matrix <- matrix(knn_basis_dim,nrow=length(knn_basis_dim),ncol=length(k_values))

require(tikzDevice)
names <- c("knn1.tex","knn2.tex","knn3.tex")
legend.pos <- c("topright","bottomright","bottomright")
legend.names <- c("$K=1$","$K=2$","$K=3$","$K=4$","$K=5$","$K=10$")
for (i in 1:length(p_values)){
  tikz(names[i], standAlone = TRUE, width=9, height=9); par(mar=rep(7,4))
  matplot(knn_basis_dim_matrix,knn.results[,,i],type="b",pch=16,lty=1,ylab="Classification Rate",xlab="$M$",col=c(2,2,'orange',3,4,1),cex.lab=3,cex.axis=3,cex=1.5,ylim=c(0.7,1),lwd=3)
  legend(legend.pos[i], inset=.02, legend=legend.names, pch=16, col=c(2,2,'orange',3,4,1),ncol=3,cex=2.5)
  dev.off()
  tools::texi2dvi(names[i],pdf=T)
}

#Cross-Validation
faces_inputs <- rbind(faces_train_inputs,faces_test_inputs) #combine training inputs/labels
faces_labels <- cbind(faces_train_labels,faces_test_labels) #combine test inputs/labels

block_size <- 10 #define block size; should be a factor of 400
blocks_matrix <- matrix(sample.int(400, size = 400, replace = FALSE), nrow = dim(faces_inputs) 
                        [1]/block_size, ncol = block_size) #each row a random test set
basis_dimensions <- seq(45,100,5); k_values <- c(1,2,3,4)
knn_CV <- matrix(0,nrow=length(k_values),ncol=length(basis_dimensions)) #initialise CV matrix


for (l in 1:length(k_values)){
  for (j in 1:length(basis_dimensions)){
    knn_CV_k=c()
    for (i in 1:dim(blocks_matrix)[1]){
      training_input <- faces_inputs[-(blocks_matrix[i,]),] #def. ith training input
      training_labels <- faces_labels[,-(blocks_matrix[i,])] #def. ith training labels
      test_input <- faces_inputs[blocks_matrix[i,],] #def. ith test input
      test_labels <- faces_labels[,blocks_matrix[i,]] #def. ith test labels
      #compute new basis from training set
      pca.basis <- pca(dim(training_input)[1],training_input); B <- as.matrix(pca.basis)
      #update k-specific CV vector, using knn classifier function
      knn_CV_k <- c(knn_CV_k,knn1(test_input,training_input,training_labels,
                                  test_labels,k_values[l],basis_dimensions[j],B,2))
    }
    knn_CV[l,j] <- mean(knn_CV_k) #update CV vector with mean ouputs for each k
  }
}

knn_CV_with_names <- knn_CV
rownames(knn_CV_with_names) <- c("K=1","K=2","K=3")
colnames(knn_CV_with_names) <- c("B=45","B=50","B=55","B=60","B=65","B=70","B=75","B=80","B=85","B=90","B=95","B=100")


#results
knn_CV <- matrix(c(0.98,0.975,0.9725,0.9725,0.9725,0.9725,0.9700,0.9700,0.9725,0.975,0.975,0.9775,0.955,0.9525,0.9575,0.9575,0.9575,0.9575,0.9575,0.955,0.9575,0.9625,0.96,0.965,0.9475,0.945,0.95,0.9525,0.9525,0.9525,0.95,0.9525,0.9525,0.955,0.955,0.9575),nrow=3,ncol=12,byrow=T)
knn_CV <- rbind(knn_CV[1,],knn_CV[1,],knn_CV[2:3,]) #include K=2

basis_dimensions_matrix <- matrix(basis_dimensions, ncol=length(k_values)+1,nrow=length(basis_dimensions))
legend.text <- c("$K=1$","$K=2$","$K=3$","$K=4$")

require(tikzDevice)
tikz("CV_plot.tex", standAlone = TRUE, width=20, height=5); par(mar=c(4.5,8,4,5))
matplot(basis_dimensions_matrix,t(knn_CV),type="b",pch=16,lty=1,ylab="Classification Rate",xlab="$M$",col=c(2,2,3:4),cex.lab=3,cex.axis=3,cex=2,lwd=3,ylim=c(0.937,0.98))
legend(76,0.95, legend=legend.text, pch=16, col=c(2,2,3:4),ncol=4,cex=3)#,horiz=TRUE)
dev.off()
tools::texi2dvi("CV_plot.tex",pdf=T)

