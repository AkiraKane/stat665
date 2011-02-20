# Crime and Community data
# Dataset taken from UCI's public ML data base
# Girish Sastry

# Read in data
library (lattice)
crime.data <- read.csv (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/communities.data", head = TRUE, sep=",")
test.index <- read.table (file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/test.index", sep = " ")
summary (crime.data)
ncols <- ncol (crime.data)
nrows <- nrow (crime.data)
#ViolentCrimesPerPop <- crime.data $ ViolentCrimesPerPop
#racepctblack <- crime.data $ racepctblack
#plot (ViolentCrimesPerPop, racepctblack)

crime.mat <- as.matrix (crime.data)
#splom (~ crime.mat[,1:14], cex = 0.5, pscales = 0, main = "Scatter Plot Matrix for Crime Data")
test.mat <- t (as.matrix (test.index)) 
test.data <- data.frame ()

for (i in test.mat[,1]) {
  test.data <- rbind (test.data, crime.data[i,])  # construct test data
  crime.data <- crime.data[-i,]
}

write.csv (crime.data, file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/crime.data.csv")
write.csv (test.data, file = "/home/gyratortron/Desktop/cs-yale/stat665/HW2/test.data.csv")