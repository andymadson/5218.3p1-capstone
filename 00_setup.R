# ============================================
# IS 5213 Capstone Day — Package Setup
# ============================================
# Run this script ONCE at the start of class
# to install and load all required packages.
# ============================================

# Install all packages needed for today
install.packages(c(
  "ggplot2", "corrplot", "rpart", "rpart.plot",
  "randomForest", "gbm", "caret", "e1071",
  "Rtsne", "cluster", "factoextra", "reshape2"
), repos = "https://cloud.r-project.org")

# Load every library we will use today
library(ggplot2)
library(corrplot)
library(reshape2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(caret)
library(e1071)
library(Rtsne)
library(cluster)
library(factoextra)

cat("All packages installed and loaded!\n")
