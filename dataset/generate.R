#set directory

setwd("/Users/immac/Desktop/DataSet_Paper")

#set the seed, parameter g in the code



g=1

# Simulation grid for Y:
# Step sizes:
dt <- 0.05
# dx = c*dt.
# Ambit linear factor:
c <- 1
# Grid coordinates:
xval <- seq(0, 10, by = c*dt); tval <- seq(0, 50000, by = dt)
nx <- length(xval); nt <- length(tval)
# Odd numbers.
# Note that area covered by Y depends on c to have the same number of datapoints.

# Levy basis:
L <- "NIG"
# Parameter vector: 
#pv<-c(0,0.5)  #parameter Gaussian seed with zero mean
pv <- c(5,0,0.2,0) #parameter NIG seed with zero mean
# Gau: (mu, tau); IG: (d, g); Gamma: (a, l); NIG: (a, b, d, u).

#To run Gau:(0, 0.1) such to have seed that are zero mean
#To run "NIG": c(5,0.0.2,0) such to have seed that are zero mean

# Definition of the kernel in terms of x-xi = u, t-s = w:
# Generalisable to g(|t-s|). Currently c|t-s| with A = 1,4
h <- function(c, u, w){
  if ((abs(u) <= c*w) & (w >= 0)){
    exp(-w)
  }
  else{0}
}

# Discretisation of kernel:
p <- 300; q <- 300
# Even numbers.

wvect <- dt*seq(0, p, by = 1)
uvect <- c*dt*seq(-q, q, by = 1)
hmat <- matrix(0, nrow = length(uvect), ncol = length(wvect))
for (i in (1:length(uvect))){
  for (j in (1:length(wvect))){
    k <- i + j
    if (k %% 2 == 0){
      hmat[i, j] <- h(c, uvect[i], wvect[j])
    }
  }
}

# List of rows:
htab <- split(hmat, row(hmat))
# kernel is symmetrical in u.
dimhx <- dim(hmat)[1]; dimht <- dim(hmat)[2]

# Dimensions of extended simulation grid for W:
nr <- nx + dimhx - 1; nc <- nt + dimht - 1

## Generate datasets ##

library(MASS); library(statmod); library(fBasics); library(GeneralizedHyperbolic)

  
  # Generate data.
  set.seed(g)
  Time <- proc.time()[3]
  Wmat <- matrix(0, nrow = nr, ncol = nc)
  for (i in 1:nr){
    for (j in 1:nc){
      k <- i + j
      if (k %% 2 == 0){
        if (L == "Gau"){
          Wmat[i, j] <- rnorm(1, mean = 2*c*(dt^2)*pv[1], sd = sqrt(2*c*(dt^2)*(pv[2]^2)))
        }
        else if (L == "IG"){
          Wmat[i, j] <- rinvgauss(1, mu = 2*c*(dt^2)*(pv[1]/pv[2]), lambda = (2*c*(dt^2)*pv[1])^2)
        }
        else if (L == "NIG"){
          Wmat[i, j] <- rnig(1, alpha = pv[1], beta = pv[2], delta = 2*c*(dt^2)*pv[3], mu = 2*c*(dt^2)*pv[4])
        }
        else if (L == "Gamma"){
          Wmat[i, j] <- rgamma(1, shape = 2*c*(dt^2)*pv[1], rate = pv[2])
        }		
      }
    }
  }
  
  # List of rows:
  Wtab <- split(Wmat, row(Wmat))
  
  # Perform discrete convolution by row:
  Y <- rep(NA, nt)
  for (i in 1:nx){
    yvect <- rep(0, nt)
    # i = 1 (x[0]):
    padstart <- i 
    for (k in 1:dimhx){
      j <- padstart + k - 1
      yvect <- yvect + convolve(Wtab[[j]], htab[[k]], type = 'f')
    }
    Y <- rbind(Y, rev(yvect))
    # Time was reversed.
  }
  # Remove dummy first row:
  Y <- Y[-1, ]
  # Insert missing values:
  for (i in 1:nx){
    for (j in 1:nt){
      k <- i + j
      if (k %% 2 != 0){
        Y[i, j] <- NA
      }	
    }
  }
  # Save computational time and dataset in files:
  comptimes <- data.frame("Time" = NA)
  comptimes$Time <- proc.time()[3] - Time
  write.table(comptimes, file = paste(L, "DGdatacomptimes.txt", sep = ""), sep = "\t", col.names = FALSE,
              row.names = FALSE, append = TRUE)
  fname <- paste(L, "diamonddatanew", g, ".txt", sep = "")
  write.matrix(Y, file = fname)


# MM inference: 

fname <- paste(L, "diamonddatanew", g, ".txt", sep = "")
Y <- matrix(scan(fname, n = nx*nt), nx, nt, byrow = TRUE)
nrY <- nrow(Y); ncY <- ncol(Y);

# Estimate kernel parameters
ndata <- ((nrY-1)/2)*ncY + ceiling(ncY/2)
# Sample variance
s1 <- sum(Y, na.rm = TRUE); s2 <- sum(Y^2, na.rm = TRUE); 
s3 <- sum(Y^3, na.rm = TRUE); s4 <- sum(Y^4, na.rm = TRUE); 
k2 <- (1/(ndata*(ndata-1)))*(ndata*s2 - s1^{2})
# Empirical variograms gamma(2*c*dx, 0) and gamma(0,2*c*dt)
d01 <- Y; d01[, 1:(ncY-2)] <- d01[, 3:ncY]; d01[, (ncY-2+1):ncY] <- NA
g01 <- mean((Y - d01)^2, na.rm = TRUE)/k2
d10 <- Y; d10[1:(nrY - 2),] <- d10[3:nrY,]; d10[(nrY-2+1):nrY, ] <- NA
g10 <- mean((Y - d10)^2, na.rm = TRUE)/k2

# Estimated lambdas
# Change if necessary
dt <- 0.05
# dx = c*dt. (known beforehand) 
# Changes with simulation grid.
hatA <- -log(1 - g01/2)/(2*dt)
hatc  <- -hatA*(2*c*dt)/log(1 - g10/2)

# Estimate Levy basis parameters (Non-parametric method)
k1 <- (1/ndata)*s1
k3 <- (1/(ndata*(ndata-1)*(ndata-2)))*((ndata^2)*s3 - (3*ndata*s2*s1)  
                                       + 2*(s1^3))
k4 <- (1/(ndata*(ndata-1)*(ndata-2)*(ndata-3)))*((ndata^3 + ndata^2)*s4 
                                                 - 4*(ndata^2 + ndata)*s3*s1 - 3*(ndata^2 - ndata)*(s2^2) 
                                                 + 12*ndata*s2*(s1^2) - 6*(s1^4))
intkn <- function(c, n, l){(2*c)/((n^2)*(l^2))}
kw1 <- k1/intkn(hatc, 1, hatA); kw2 <- k2/intkn(hatc, 2, hatA)
kw3 <- k3/intkn(hatc, 3, hatA); 
kw4 <- k4/intkn(hatc, 4, hatA)

if (L == "Gau"){
  hatmu <- kw1; hattau <- sqrt(kw2)
  # Add results	
  result <- data.frame("Trial" = NA, "g10" = NA, "g01" = NA, "hatA" = NA, "hatc" = NA, "kw3" = NA, "hatmu" = NA, "hattau" = NA)
  result$Trial <- g
  result$g10 <- g10; result$g01 <- g01; result$hatA <- hatA
  result$hatc <- hatc; result$kw3 <- kw3; result$hatmu <- hatmu; 
  result$hattau <- hattau
  write.table(result, file = "diamonddataGauMM.txt", sep = "\t", col.names = FALSE, 
              row.names = FALSE, append = TRUE)
} else if (L == "IG"){
  hatgamma <- sqrt(kw1/kw2); hatdelta <- kw1*hatgamma
  # Add results	
  result <- data.frame("Trial" = NA, "g10" = NA, "g01" = NA, "hatA" = NA, "hatc" = NA, "kw3" = NA, "hatd" = NA, "hatg" = NA)
  result$Trial <- g
  result$g10 <- g10; result$g01 <- g01; result$hatA <- hatA
  result$hatc <- hatc; result$kw3 <- kw3; result$hatd <- hatdelta; 
  result$hatg<- hatgamma
  write.table(result, file = "diamonddataIGMM.txt", sep = "\t", col.names = FALSE, 
              row.names = FALSE, append = TRUE)
}	else if (L == "Gamma"){
  hatbeta <- kw1/kw2
  hatalpha <- hatbeta*kw1
  # Add results	
  result <- data.frame("Trial" = NA, "g10" = NA, "g01" = NA, "hatA" = NA, "hatc" = NA, "kw3" = NA, "hata" = NA, "hatbeta" = NA)
  result$Trial <- g
  result$g10 <- g10; result$g01 <- g01; result$hatA <- hatA
  result$hatc <- hatc; result$kw3 <- kw3; result$hata <- hatalpha
  result$hatbeta<- hatbeta
  write.table(result, file = "diamonddataGammaMM.txt", sep = "\t", col.names = FALSE, 
              row.names = FALSE, append = TRUE)
}	else if (L == "NIG"){
  hatb <- (3*kw2*kw3)/(3*kw2*kw4- 5*(kw3^2))
  a <- 3*hatb*kw2/kw3
  hatm <- kw1 - (kw4*(a^3)*hatb)/(3*(5*(hatb^2) + a)*((hatb^2) + a))
  hatd <- kw4*(a^(7/2))/(3*(5*(hatb^2) + a)*(hatb^2 + a))
  hatalp <- sqrt(hatb^2 + a)
  # Add results	
  result <- data.frame("Trial" = NA, "g10" = NA, "g01" = NA, "hatA" = NA, "hatc" = NA, "kw3" = NA, "hatalp" = NA, "hatb" = NA, "hatd"=NA, "hatm" = NA)
  result$Trial <- g
  result$g10 <- g10; result$g01 <- g01; result$hatA <- hatA
  result$hatc <- hatc; result$kw3 <- kw3; result$hatalp <- hatalp; 
  result$hatb<- hatb; result$hatd <- hatd; result$hatm <- hatm;
  write.table(result, file = "diamonddataNIGMM.txt", sep = "\t", col.names = FALSE, 
              row.names = FALSE, append = TRUE)
}	
