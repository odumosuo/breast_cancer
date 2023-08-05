##################### BINF6970_ASSIGNMENT_4

# Authors:
#     Daniel Gyamfi Amoako
#     Olusegun Odumosu
# Date: April 23, 2023
# 
# This script explores the datasets (trainset.csv) and (testset.csv) to predict tumor status (benign or malignant) using elastic net, a fully grown classification/regression tree (CART), Support Vector Machine (SVM), a bagged version of CART, and Random Forest.
#This is  classification problem to classify breast tumors as malign or benign.
# 
# Problem: Predicting Cancer related Malignant Cell Growth

# Required input documents:
# trainset.csv (in the present working directory)
# testset.csv (in the present working directory)


####################################### CODE FOR ANALYSIS ############################
#-------------------------------------------------------------------------------------------------------------------------------------.
###### SETTING UP #####
# Install the packages 
# install.packages("caret")
# install.packages("glmnet")
# install.packages("rpart")
# install.packages("e1071")
# install.packages("ROCR")
# install.packages("pROC")
# install.packages("ggfortify")
# installed.packages("kernlab")
# install.packages("gridExtra")
# install.packages("cowplot")

# Load the libraries
library(readr)
library(dplyr)
library(purrr)
library(ggplot2)
library(caret)
library(glmnet)
library(rpart)
library(e1071)
library(ROCR)
library(pROC)
library(rpart)
library(ggfortify)
library(kernlab)
library(gridExtra)
library(cowplot)
library(grid)
library(ggpubr)
library(GGally)



############# DATA EXPLORATION ###########
#import data 
df <- read_csv("trainset.csv")
# Check for appropriate class on each of the variable.  
glimpse(df)
# set status as factor
df$status <- as.factor(df$status)
# check for missing values
map_int(df, function(.x) sum(is.na(.x)))
# There are no missing values. In the case that there would be many missing values, we would go on the transforming data for some appropriate imputation.


# check for class imbalance
round(prop.table(table(df$status)), 2)
# The response variable is slightly unbalanced.


#### Check for multicollinearity
#Let's look for correlation in the variables.  Most ML algorithms assumed that the predictor variables are  independent from each others.
#Let's check for correlations.  For an analysis to be robust it is good to remove multicollinearity (remove highly correlated predictors) 
df_corr <- cor(df %>% select(-status))
corrplot::corrplot(df_corr, order = "hclust", tl.cex = 1, addrect = 8)
title(main = "Correlation plot of all covariates")
# Shows that indeed there are quite a few variables that are correlated.  On the next step, we will remove the highly correlated ones using the `caret` package.  



## Remove highly correlated features
# The findcorrelation() function from caret package remove highly correlated predictors
# based on whose correlation is above 0.9. This function uses a heuristic algorithm 
# to determine which variable should be removed instead of selecting blindly
df2 <- df %>% select(-findCorrelation(df_corr, cutoff = 0.9))
#Number of columns for our new data frame
ncol(df2)
#Extract the names of the removed columns
removed_cols <- setdiff(names(df), names(df2))
#Create a new data frame containing the removed columns
df_removed <- df[, removed_cols]
#Print the removed columns
print(removed_cols)
# our new dataframe is 10 variables shorter


#### PCA analysis
## PCA with all features
# Let's first go on an unsupervised analysis with a PCA analysis.  
# To do so, we will remove the `status` variable, then we will also scale and center the variables.
preproc_pca_df <- prcomp(df %>% select(-status), scale = TRUE, center = TRUE)
summary(preproc_pca_df)

# Calculate the proportion of variance explained
pca_df_var <- preproc_pca_df$sdev^2
pve_df <- pca_df_var / sum(pca_df_var)
cum_pve <- cumsum(pve_df)
pve_table <- tibble(comp = seq(1:ncol(df %>% select(-status))), pve_df, cum_pve)

# plot the proportion of variance explained
cumulative_PCA <- ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0) + 
  labs(x = "Number of components", y = "Cumulative Variance", title = "Cumulative plot of explained variance along principal components")
# With the original dataset, 95% of the variance is explained with 10 PC's.
#Let's check on the most influential variables for the first 2 components 
# Plot PCA
pca_df <- as_tibble(preproc_pca_df$x)
ggplot(pca_df, aes(x = PC1, y = PC2, col = df$status)) + geom_point()
# It does look like the first 2 components managed to separate the status quite well.    

# We can use the ggfortify library to get a more detailed analysis of what variables are the most influential in the first 2 components 
PCA1 <- autoplot(preproc_pca_df, data = df,  colour = 'status',
                 loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue") + ggtitle("With all features")

# Let's visualize the first 3 components.
df_pcs <- cbind(as_tibble(df$status), as_tibble(preproc_pca_df$x))
GGally::ggpairs(df_pcs, columns = 2:4, ggplot2::aes(color = value))
# First two components still explains majority of the variance in the dataset.


## Perform PCA with selected features
preproc_pca_df2 <- prcomp(df2, scale = TRUE, center = TRUE)
summary(preproc_pca_df2)
pca_df2_var <- preproc_pca_df2$sdev^2
# proportion of variance explained
pve_df2 <- pca_df2_var / sum(pca_df2_var)
cum_pve_df2 <- cumsum(pve_df2)
pve_table_df2 <- tibble(comp = seq(1:ncol(df2)), pve_df2, cum_pve_df2)
#cumulative variance plot
cumulative_PCA2 <- ggplot(pve_table_df2, aes(x = comp, y = cum_pve_df2)) +
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0) + 
  labs(x = "Number of components", y = "Cumulative Variance", title = "Cumulative plot of explained variance along principal components")
plot(cumulative_PCA2)
# Shows that in this case, around 8 PC's explained 95% of the variance.




#################### CREATE CLASSIFICATION MODELS ####################
#### CREAT TRAINING AND TESTING DATASET WITH GIVEN VARIABLES
set.seed(1815)
df3 <- cbind(status = df$status, df2)
# Load the test set data from the CSV file
df4 <- read_csv("testset.csv")
# Extract the names of the removed columns
removed_cols <- setdiff(names(df), names(df2))
# Remove "status" from the list of removed columns
removed_cols <- removed_cols[removed_cols != "status"]
# Create a new data frame containing only the columns that are not in removed_cols
df5 <- df4[, c(setdiff(names(df4), removed_cols))]
# Assign df3 to df_training
df_training <- df3
# Assign df5 to df_testing
df_testing <-  df5
# Set up cross-validation method and control parameters for train function
df_control <- trainControl(
  method="repeatedcv",  # Use repeated cross-validation method
  number = 10,  # Number of folds for cross-validation
  repeats = 3,  # Number of repeats for cross-validation
  classProbs = TRUE,  # Compute class probabilities
  summaryFunction = twoClassSummary  # Set up summary function for two-class problems
)


###### CREATE TRAINING AND TESTING DATASET WITH QUADRATIC VARIABLES 
### create training dataset with quadratic variables
# Explore the quadratic effects of all covariates and highlight the most important/influential ones in the trained logistic regression model;
# The code below gets the columns in "df3"
colnames(df3)
# Extract all column names except the "status" column
predictor_vars_logreg <- colnames(df3)
predictor_vars_logreg <- predictor_vars_logreg[-1]
# Square each predictor variable
predictors_sq_logreg <- sapply(predictor_vars_logreg, function(x) df3[[x]]^2)
# Combine the squared predictor variables into a data frame
predictors_sq_df_logreg <- as.data.frame(predictors_sq_logreg)
colnames(predictors_sq_df_logreg) <- paste(predictor_vars_logreg, "_sq", sep = "")
# Combine the original and squared predictor variables into a new data frame
df_training_sq <- cbind(df3[predictor_vars_logreg], predictors_sq_df_logreg)
df_training_sq$status <- df3$status


### create testing dataset with quadratic variables
# Square each predictor variable
testing_predictors_sq_logreg <- sapply(predictor_vars_logreg, function(x) df_testing[[x]]^2)
# Combine the squared predictor variables into a data frame
testing_predictors_sq_logreg <- as.data.frame(testing_predictors_sq_logreg)
colnames(testing_predictors_sq_logreg) <- paste(predictor_vars_logreg, "_sq", sep = "")
# Combine the original and squared predictor variables into a new data frame
df_testing_sq <- cbind(df_testing[predictor_vars_logreg], testing_predictors_sq_logreg)
df_testing_sq$status <- df_testing$status


#### CREATE MODEL FOR ELASTIC NET #####
# Create elastic net model with both Lasso and Ridge regularization
# Create control object for repeated cross-validation
set.seed(1815)
df_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)
# Set up search grid
grid <- expand.grid(alpha = seq(0, 1, length.out = 11), lambda = 10^seq(-5, 5, length.out = 11))
# Fit model using elastic net with Lasso and Ridge regularization
model_logreg <- train(status ~., data = df_training, method = "glmnet", 
                      metric = "ROC", preProcess = c("scale", "center"), 
                      trControl = df_control, tuneGrid = grid)
# Plot the logistic regression model
plot(model_logreg)
# Make predictions on the testing data
prediction_logreg_df <- predict(model_logreg, newdata = df_testing)
# Convert df_testing$status to a factor with the same levels as prediction_logreg_df
df_testing$status <- factor(df_testing$status, levels = levels(prediction_logreg_df))
# Compute confusion matrix
cm_logreg_df <- confusionMatrix(prediction_logreg_df, df_testing$status, positive = "M")
cm_logreg_df


### Compute evaluation metrics 
# Compute the accuracy, sensitivity, specificity, precision, recall, F1-score, and AUC-ROC of the model
accuracy_lr <- cm_logreg_df$overall[1]
sensitivity_lr <- cm_logreg_df$byClass[1]
specificity_lr <- cm_logreg_df$byClass[2]
precision_lr <- cm_logreg_df$byClass[3]
recall_lr <- cm_logreg_df$byClass[1]
f1_score_lr <- (2 * precision_lr * recall_lr) / (precision_lr + recall_lr)
# Compute AUC-ROC
probabilities_lr <- predict(model_logreg, newdata = df_testing, type = "prob")
auc_roc_lr <- roc(df_testing$status, probabilities_lr[, "M"])$auc


## Print the evaluation metrics
cat(paste0("Logistic regression model_Accuracy: ", round(accuracy_lr, 4), "\n"))
cat(paste0("Logistic regression model_Sensitivity: ", round(sensitivity_lr, 4), "\n"))
cat(paste0("Logistic regression model_Specificity: ", round(specificity_lr, 4), "\n"))
cat(paste0("Logistic regression model_Precision: ", round(precision_lr, 4), "\n"))
cat(paste0("Logistic regression model_Recall: ", round(recall_lr, 4), "\n"))
cat(paste0("Logistic regression model_F1-score: ", round(f1_score_lr, 4), "\n"))
cat(paste0("Logistic regression model_AUC-ROC: ", round(auc_roc_lr, 4), "\n"))




### With quadratic dataset
# Fit a glmnet model on the new data frame using cross-validation
model_logreg_sq_fs <- train(status ~., data = df_training_sq, method = "glmnet", 
                            metric = "ROC", preProcess = c("scale", "center"), 
                            trControl = df_control, tuneGrid = grid)
# Make predictions on the testing data
prediction_logreg_df_sq <- predict(model_logreg_sq_fs, newdata = df_testing_sq)
# Convert df_testing_sq$status to a factor with the same levels as prediction_logreg_df_sq
df_testing_sq$status <- factor(df_testing_sq$status, levels = levels(prediction_logreg_df_sq))
# Compute confusion matrix
cm_logreg_df_sq <- confusionMatrix(prediction_logreg_df_sq, df_testing_sq$status, positive = "M")
cm_logreg_df_sq
# Compute the accuracy, sensitivity, specificity, precision, recall, F1-score, and AUC-ROC of the model
accuracy_lr_sq <- cm_logreg_df_sq$overall[1]
sensitivity_lr_sq <- cm_logreg_df_sq$byClass[1]
specificity_lr_sq <- cm_logreg_df_sq$byClass[2]
precision_lr_sq <- cm_logreg_df_sq$byClass[3]
recall_lr_sq <- cm_logreg_df_sq$byClass[1]
f1_score_lr_sq <- (2 * precision_lr_sq * recall_lr_sq) / (precision_lr_sq + recall_lr_sq)
# Compute AUC-ROC
probabilities_lr_sq <- predict(model_logreg_sq_fs, newdata = df_testing_sq, type = "prob")
auc_roc_lr_sq <- roc(df_testing_sq$status, probabilities_lr_sq[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Logistic regression quadratic model_Accuracy: ", round(accuracy_lr_sq, 4), "\n"))
cat(paste0("Logistic regression quadratic model_Sensitivity: ", round(sensitivity_lr_sq, 4), "\n"))
cat(paste0("Logistic regression quadratic model_Specificity: ", round(specificity_lr_sq, 4), "\n"))
cat(paste0("Logistic regression quadratic model_Precision: ", round(precision_lr_sq, 4), "\n"))
cat(paste0("Logistic regression quadratic model_Recall: ", round(recall_lr_sq, 4), "\n"))
cat(paste0("Logistic regression quadratic model_F1-score: ", round(f1_score_lr_sq, 4), "\n"))
cat(paste0("Logistic regression quadratic model_AUC-ROC: ", round(auc_roc_lr_sq, 4), "\n"))


### Calculate variable importance scores
# Calculate variable importance scores for the glmnet model with squared predictors
var_imp_sq_logreg_sq <- varImp(model_logreg_sq_fs)
# Extract the variable importance scores and sort in descending order
print(var_imp_sq_logreg_sq)
# Calculate variable importance scores for the original glmnet model
var_imp_logreg <- varImp(model_logreg)
# Print out the 10 most influential variables for the original model
print(var_imp_logreg)






##### CREATE MODEL FOR K-NEAREST NEIGHBORS (KNN) #####
set.seed(1815)
model_knn_df <- train(status ~., data = df_training, 
                      method = "knn", 
                      metric = "ROC", 
                      preProcess = c("scale", "center"), 
                      trControl = df_control, 
                      tuneLength =31)
# Plot the logistic regression model
plot(model_knn_df)
# Make predictions on the testing data
prediction_knn_df <- predict(model_knn_df, newdata = df_testing)
# Convert df_testing$status to a factor with the same levels as prediction_knn_df
df_testing$status <- factor(df_testing$status, levels = levels(prediction_knn_df))
# Compute confusion matrix
cm_knn_df <- confusionMatrix(prediction_knn_df, df_testing$status, positive = "M")
cm_knn_df

# Compute the accuracy, sensitivity, specificity, precision, recall, F1-score, and AUC-ROC of the model
accuracy_knn <- cm_knn_df$overall[1]
sensitivity_knn <- cm_knn_df$byClass[1]
specificity_knn <- cm_knn_df$byClass[2]
precision_knn <- cm_knn_df$byClass[3]
recall_knn <- cm_knn_df$byClass[1]
f1_score_knn <- (2 * precision_knn * recall_knn) / (precision_knn + recall_knn)
# Compute AUC-ROC
probabilities_knn <- predict(model_knn_df, newdata = df_testing, type = "prob")
auc_roc_knn <- roc(df_testing$status, probabilities_knn[, "M"])$auc

# Print the evaluation metrics
cat(paste0("KNN model_Accuracy: ", round(accuracy_knn, 4), "\n"))
cat(paste0("KNN model_Sensitivity: ", round(sensitivity_knn, 4), "\n"))
cat(paste0("KNN model_Specificity: ", round(specificity_knn, 4), "\n"))
cat(paste0("KNN model_Precision: ", round(precision_knn, 4), "\n"))
cat(paste0("KNN model_Recall: ", round(recall_knn, 4), "\n"))
cat(paste0("KNN model_F1-score: ", round(f1_score_knn, 4), "\n"))
cat(paste0("KNN model_AUC-ROC: ", round(auc_roc_knn, 4), "\n"))


#### Explore the quadratic effects of all covariates 
# Fit a glmnet model on the new data frame using cross-validation
model_knn_df_sq_fs <- train(status ~., data = df_training_sq, method = "glmnet", 
                            metric = "ROC", preProcess = c("scale", "center"), 
                            trControl = df_control, tuneGrid = grid)
# Make predictions on the testing data
prediction_knn_df_sq <- predict(model_knn_df_sq_fs, newdata = df_testing_sq)
# Convert df_testing_sq$status to a factor with the same levels as prediction_knn_df_sq
df_testing_sq$status <- factor(df_testing_sq$status, levels = levels(prediction_knn_df_sq))
# Compute confusion matrix
cm_knn_df_sq <- confusionMatrix(prediction_knn_df_sq, df_testing_sq$status, positive = "M")
cm_knn_df_sq

# Compute the accuracy, sensitivity, specificity, precision, recall, F1-score, and AUC-ROC of the model
accuracy_knn_sq <- cm_knn_df_sq$overall[1]
sensitivity_knn_sq <- cm_knn_df_sq$byClass[1]
specificity_knn_sq <- cm_knn_df_sq$byClass[2]
precision_knn_sq <- cm_knn_df_sq$byClass[3]
recall_knn_sq <- cm_knn_df_sq$byClass[1]
f1_score_knn_sq <- (2 * precision_knn_sq * recall_knn_sq) / (precision_knn_sq + recall_knn_sq)
# Compute AUC-ROC
probabilities_knn_sq <- predict(model_knn_df_sq_fs, newdata = df_testing_sq, type = "prob")
auc_roc_knn_sq <- roc(df_testing_sq$status, probabilities_knn_sq[, "M"])$auc

# Print the evaluation metrics
cat(paste0("KNN quadratic model_Accuracy: ", round(accuracy_knn_sq, 4), "\n"))
cat(paste0("KNN quadratic model_Sensitivity: ", round(sensitivity_knn_sq, 4), "\n"))
cat(paste0("KNN quadratic model_Specificity: ", round(specificity_knn_sq, 4), "\n"))
cat(paste0("KNN quadratic model_Precision: ", round(precision_knn_sq, 4), "\n"))
cat(paste0("KNN quadratic model_Recall: ", round(recall_knn_sq, 4), "\n"))
cat(paste0("KNN quadratic model_F1-score: ", round(f1_score_knn_sq, 4), "\n"))
cat(paste0("KNN quadratic model_AUC-ROC: ", round(auc_roc_knn_sq, 4), "\n"))

### Calculate variable importance scores
# Calculate variable importance scores for the glmnet model with squared predictors
var_imp_sq_knn <- varImp(model_knn_df_sq_fs)
# Extract the variable importance scores and sort in descending order
print(var_imp_sq_knn)
# Calculate variable importance scores for the original glmnet model
var_imp_knn <- varImp(model_knn_df)
# Print out the 10 most influential variables for the original model
print(var_imp_knn)




##### CREATE MODEL FOR CART #####
set.seed(1815)
# Fit model using decision tree with recursive partitioning
model_rpart_df <- train(status ~ ., data = df_training, method = "rpart", metric = "ROC", preProcess = c("scale", "center"), trControl = df_control, tuneLength = 10)
#Plot the CART model
plot(model_rpart_df)
# Make predictions on the testing data
prediction_rpart_df <- predict(model_rpart_df, newdata = df_testing)
# Convert df_testing$status to a factor with the same levels as prediction_rpart_df
df_testing$status <- factor(df_testing$status, levels = levels(prediction_rpart_df))
# Calculate confusion matrix
cm_rpart_df <- confusionMatrix(prediction_rpart_df, df_testing$status, positive = "M")

# Compute evaluation metrics
accuracy_dt <- cm_rpart_df$overall[1]
sensitivity_dt <- cm_rpart_df$byClass[1]
specificity_dt <- cm_rpart_df$byClass[2]
precision_dt <- cm_rpart_df$byClass[3]
recall_dt <- cm_rpart_df$byClass[1]
f1_score_dt <- (2 * precision_dt * recall_dt) / (precision_dt + recall_dt)
# Compute AUC-ROC
probabilities_dt <- predict(model_rpart_df, newdata = df_testing, type = "prob")
auc_roc_dt <- roc(df_testing$status, probabilities_dt[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Fully grown classification tree (CART) model_Accuracy: ", round(accuracy_dt, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) model_Sensitivity: ", round(sensitivity_dt, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) model_Specificity: ", round(specificity_dt, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) model_Precision: ", round(precision_dt, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) model_Recall: ", round(recall_dt, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) model_F1-score: ", round(f1_score_dt, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) model_AUC-ROC: ", round(auc_roc_dt, 4), "\n"))


#### Explore the quadratic effects
# Fit a glmnet model on the new data frame using cross-validation
model_rpart_df_sq_fs <- train(status ~., data = df_training_sq, method = "glmnet", 
                              metric = "ROC", preProcess = c("scale", "center"), 
                              trControl = df_control, tuneGrid = grid)
# Make predictions on the testing data
prediction_rpart_df_sq <- predict(model_rpart_df_sq_fs, newdata = df_testing_sq)
# Convert df_testing_sq$status to a factor with the same levels as prediction_rpart_df_sq
df_testing_sq$status <- factor(df_testing_sq$status, levels = levels(prediction_rpart_df_sq))
# Calculate confusion matrix
cm_rpart_df_sq <- confusionMatrix(prediction_rpart_df_sq, df_testing_sq$status, positive = "M")

# Compute evaluation metrics
accuracy_dt_sq <- cm_rpart_df_sq$overall[1]
sensitivity_dt_sq <- cm_rpart_df_sq$byClass[1]
specificity_dt_sq <- cm_rpart_df_sq$byClass[2]
precision_dt_sq <- cm_rpart_df_sq$byClass[3]
recall_dt_sq <- cm_rpart_df_sq$byClass[1]
f1_score_dt_sq <- (2 * precision_dt_sq * recall_dt_sq) / (precision_dt_sq + recall_dt_sq)
# Compute AUC-ROC
probabilities_dt_sq <- predict(model_rpart_df_sq_fs, newdata = df_testing_sq, type = "prob")
auc_roc_dt_sq <- roc(df_testing_sq$status, probabilities_dt_sq[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Fully grown classification tree (CART) quadratic model_Accuracy: ", round(accuracy_dt_sq, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) quadratic model_Sensitivity: ", round(sensitivity_dt_sq, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) quadratic model_Specificity: ", round(specificity_dt_sq, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) quadratic model_Precision: ", round(precision_dt_sq, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) quadratic model_Recall: ", round(recall_dt_sq, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) quadratic model_F1-score: ", round(f1_score_dt_sq, 4), "\n"))
cat(paste0("Fully grown classification tree (CART) quadratic model_AUC-ROC: ", round(auc_roc_dt_sq, 4), "\n"))

# Calculate variable importance scores for the glmnet model with squared predictors
var_imp_sq_rpart <- varImp(model_rpart_df_sq_fs)
# Extract the variable importance scores and sort in descending order
print(var_imp_sq_rpart)
# Calculate variable importance scores for the original glmnet model
var_imp_rpart <- varImp(model_rpart_df)
# Print out the 10 most influential variables for the original model
print(var_imp_rpart)




##### CREATE MODEL FOR SVM ######
set.seed(1815)
# Fit model using decision tree with recursive partitioning
model_svm_df <- train(status~.,
                      df_training, method = "svmLinear", metric = "ROC", 
                      preProcess = c('center', 'scale'), 
                      trControl = df_control)

# Make predictions on the testing data
prediction_svm_df <- predict(model_svm_df, newdata = df_testing)
# Convert df_testing$status to a factor with the same levels as prediction_svm_df
df_testing$status <- factor(df_testing$status, levels = levels(prediction_svm_df))
# Calculate confusion matrix
cm_svm_df <- confusionMatrix(prediction_svm_df, df_testing$status, positive = "M")

# Compute evaluation metrics
accuracy_svm <- cm_svm_df$overall[1]
sensitivity_svm <- cm_svm_df$byClass[1]
specificity_svm <- cm_svm_df$byClass[2]
precision_svm <- cm_svm_df$byClass[3]
recall_svm <- cm_svm_df$byClass[1]
f1_score_svm <- (2 * precision_svm * recall_svm) / (precision_svm + recall_svm)
# Compute AUC-ROC
probabilities_svm <- predict(model_svm_df, newdata = df_testing, type = "prob")
auc_roc_svm <- roc(df_testing$status, probabilities_svm[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Support Vector Machine model_Accuracy: ", round(accuracy_svm, 4), "\n"))
cat(paste0("Support Vector Machine model_Sensitivity: ", round(sensitivity_svm, 4), "\n"))
cat(paste0("Support Vector Machine model_Specificity: ", round(specificity_svm, 4), "\n"))
cat(paste0("Support Vector Machine model_Precision: ", round(precision_svm, 4), "\n"))
cat(paste0("Support Vector Machine model_Recall: ", round(recall_svm, 4), "\n"))
cat(paste0("Support Vector Machine model_F1-score: ", round(f1_score_svm, 4), "\n"))
cat(paste0("Support Vector Machine model_AUC-ROC: ", round(auc_roc_svm, 4), "\n"))


### Explore the quadratic effects of all covariates 
# Fit a glmnet model on the new data frame using cross-validation
model_svm_df_sq_fs <- train(status ~., data = df_training_sq, method = "glmnet", 
                            metric = "ROC", preProcess = c("scale", "center"), 
                            trControl = df_control, tuneGrid = grid)
# Make predictions on the testing data
prediction_svm_df_sq <- predict(model_svm_df_sq_fs, newdata = df_testing_sq)
# Convert df_testing_sq$status to a factor with the same levels as prediction_svm_df_sq
df_testing_sq$status <- factor(df_testing_sq$status, levels = levels(prediction_svm_df_sq))
# Calculate confusion matrix
cm_svm_df_sq <- confusionMatrix(prediction_svm_df_sq, df_testing_sq$status, positive = "M")

# Compute evaluation metrics
accuracy_svm_sq <- cm_svm_df_sq$overall[1]
sensitivity_svm_sq <- cm_svm_df_sq$byClass[1]
specificity_svm_sq <- cm_svm_df_sq$byClass[2]
precision_svm_sq <- cm_svm_df_sq$byClass[3]
recall_svm_sq <- cm_svm_df_sq$byClass[1]
f1_score_svm_sq <- (2 * precision_svm_sq * recall_svm_sq) / (precision_svm_sq + recall_svm_sq)
# Compute AUC-ROC
probabilities_svm_sq <- predict(model_svm_df_sq_fs, newdata = df_testing_sq, type = "prob")
auc_roc_svm_sq <- roc(df_testing_sq$status, probabilities_svm_sq[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Support Vector Machine quadratic model_Accuracy: ", round(accuracy_svm_sq, 4), "\n"))
cat(paste0("Support Vector Machine quadratic model_Sensitivity: ", round(sensitivity_svm_sq, 4), "\n"))
cat(paste0("Support Vector Machine quadratic model_Specificity: ", round(specificity_svm_sq, 4), "\n"))
cat(paste0("Support Vector Machine quadratic model_Precision: ", round(precision_svm_sq, 4), "\n"))
cat(paste0("Support Vector Machine quadratic model_Recall: ", round(recall_svm_sq, 4), "\n"))
cat(paste0("Support Vector Machine quadratic model_F1-score: ", round(f1_score_svm_sq, 4), "\n"))
cat(paste0("Support Vector Machine quadratic model_AUC-ROC: ", round(auc_roc_svm_sq, 4), "\n"))


# Calculate variable importance scores for the glmnet model with squared predictors
var_imp_sq_svm <- varImp(model_svm_df_sq_fs)
# Extract the variable importance scores and sort in descending order
print(var_imp_sq_svm)
# Calculate variable importance scores for the original glmnet model
var_imp_svm <- varImp(model_svm_df)
# Print out the 10 most influential variables for the original model
print(var_imp_svm)




##### CREATE MODEL FOR BAGGED CART ######
set.seed(1815)
# Fit model using bagged CART
model_bagged_cart_df <- train(status ~ ., data = df_training, method = "treebag", metric = "ROC", preProcess = c("scale", "center"), trControl = df_control, tuneLength = 10, ntree = 50)
# Make predictions on the testing data
prediction_bagged_cart_df <- predict(model_bagged_cart_df, newdata = df_testing)
# Convert df_testing$status to a factor with the same levels as prediction_bagged_cart_df
df_testing$status <- factor(df_testing$status, levels = levels(prediction_bagged_cart_df))
# Calculate confusion matrix
cm_bagged_cart_df <- confusionMatrix(prediction_bagged_cart_df, df_testing$status, positive = "M")

# Compute evaluation metrics
accuracy_bc <- cm_bagged_cart_df$overall[1]
sensitivity_bc <- cm_bagged_cart_df$byClass[1]
specificity_bc <- cm_bagged_cart_df$byClass[2]
precision_bc <- cm_bagged_cart_df$byClass[3]
recall_bc <- cm_bagged_cart_df$byClass[1]
f1_score_bc <- (2 * precision_bc * recall_bc) / (precision_bc + recall_bc)
# Compute AUC-ROC
probabilities_bc <- predict(model_bagged_cart_df, newdata = df_testing, type = "prob")
auc_roc_bc <- roc(df_testing$status, probabilities_bc[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Bagged CART model_Accuracy: ", round(accuracy_bc, 4), "\n"))
cat(paste0("Bagged CART model_Sensitivity: ", round(sensitivity_bc, 4), "\n"))
cat(paste0("Bagged CART model_Specificity: ", round(specificity_bc, 4), "\n"))
cat(paste0("Bagged CART model_Precision: ", round(precision_bc, 4), "\n"))
cat(paste0("Bagged CART model_Recall: ", round(recall_bc, 4), "\n"))
cat(paste0("Bagged CART model_F1-score: ", round(f1_score_bc, 4), "\n"))
cat(paste0("Bagged CART model_AUC-ROC: ", round(auc_roc_bc, 4), "\n"))


#### Explore the quadratic effects of all covariates 
# Fit a glmnet model on the new data frame using cross-validation
model_bc_df_sq_fs <- train(status ~., data = df_training_sq, method = "glmnet", 
                           metric = "ROC", preProcess = c("scale", "center"), 
                           trControl = df_control, tuneGrid = grid)
# Make predictions on the testing data
prediction_bagged_cart_df_sq <- predict(model_bc_df_sq_fs, newdata = df_testing_sq)
# Convert df_testing_sq$status to a factor with the same levels as prediction_bagged_cart_df_sq
df_testing_sq$status <- factor(df_testing_sq$status, levels = levels(prediction_bagged_cart_df_sq))
# Calculate confusion matrix
cm_bagged_cart_df_sq <- confusionMatrix(prediction_bagged_cart_df_sq, df_testing_sq$status, positive = "M")

# Compute evaluation metrics
accuracy_bc_sq <- cm_bagged_cart_df_sq$overall[1]
sensitivity_bc_sq <- cm_bagged_cart_df_sq$byClass[1]
specificity_bc_sq <- cm_bagged_cart_df_sq$byClass[2]
precision_bc_sq <- cm_bagged_cart_df_sq$byClass[3]
recall_bc_sq <- cm_bagged_cart_df_sq$byClass[1]
f1_score_bc_sq <- (2 * precision_bc_sq * recall_bc_sq) / (precision_bc_sq + recall_bc_sq)
# Compute AUC-ROC
probabilities_bc_sq <- predict(model_bc_df_sq_fs, newdata = df_testing_sq, type = "prob")
auc_roc_bc_sq <- roc(df_testing_sq$status, probabilities_bc_sq[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Bagged CART quadratic model_Accuracy: ", round(accuracy_bc_sq, 4), "\n"))
cat(paste0("Bagged CART quadratic model_Sensitivity: ", round(sensitivity_bc_sq, 4), "\n"))
cat(paste0("Bagged CART quadratic model_Specificity: ", round(specificity_bc_sq, 4), "\n"))
cat(paste0("Bagged CART quadratic model_Precision: ", round(precision_bc_sq, 4), "\n"))
cat(paste0("Bagged CART quadratic model_Recall: ", round(recall_bc_sq, 4), "\n"))
cat(paste0("Bagged CART quadratic model_F1-score: ", round(f1_score_bc_sq, 4), "\n"))
cat(paste0("Bagged CART quadratic model_AUC-ROC: ", round(auc_roc_bc_sq, 4), "\n"))


# Calculate variable importance scores for the glmnet model with squared predictors
var_imp_sq_bc <- varImp(model_bc_df_sq_fs)
# Extract the variable importance scores and sort in descending order
print(var_imp_sq_bc)
# Calculate variable importance scores for the original glmnet model
var_imp_bc <- varImp(model_bagged_cart_df)
# Print out the 10 most influential variables for the original model
print(var_imp_bc)



##### CREATE MODEL FOR RANDOM FOREST #####
set.seed(1815)
model_rf_df <- train(status ~., data = df_training,
                     method = "rf", 
                     metric = 'ROC', 
                     trControl = df_control)
# Plot the logistic regression model
plot(model_rf_df)
# Make predictions on the testing data
prediction_rf_df <- predict(model_rf_df, newdata = df_testing)
# Convert df_testing$status to a factor with the same levels as prediction_rf_df
df_testing$status <- factor(df_testing$status, levels = levels(prediction_rf_df))
# Compute confusion matrix
cm_rf_df <- confusionMatrix(prediction_rf_df, df_testing$status, positive = "M")

# Compute the accuracy, sensitivity, specificity, precision, recall, F1-score, and AUC-ROC of the model
accuracy_rf <- cm_rf_df$overall[1]
sensitivity_rf <- cm_rf_df$byClass[1]
specificity_rf <- cm_rf_df$byClass[2]
precision_rf <- cm_rf_df$byClass[3]
recall_rf <- cm_rf_df$byClass[1]
f1_score_rf <- (2 * precision_rf * recall_rf) / (precision_rf + recall_rf)
# Compute AUC-ROC
probabilities_rf <- predict(model_rf_df, newdata = df_testing, type = "prob")
auc_roc_rf <- roc(df_testing$status, probabilities_rf[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Random forest model_Accuracy: ", round(accuracy_rf, 4), "\n"))
cat(paste0("Random forest model_Sensitivity: ", round(sensitivity_rf, 4), "\n"))
cat(paste0("Random forest model_Specificity: ", round(specificity_rf, 4), "\n"))
cat(paste0("Random forest model_Precision: ", round(precision_rf, 4), "\n"))
cat(paste0("Random forest model_Recall: ", round(recall_rf, 4), "\n"))
cat(paste0("Random forest model_F1-score: ", round(f1_score_rf, 4), "\n"))
cat(paste0("Random forest model_AUC-ROC: ", round(auc_roc_rf, 4), "\n"))


#### Explore the quadratic effects of all covariates 
# Fit a glmnet model on the new data frame using cross-validation
model_rf_df_sq_fs <- train(status ~., data = df_training_sq, method = "glmnet", 
                           metric = "ROC", preProcess = c("scale", "center"), 
                           trControl = df_control, tuneGrid = grid)
# Make predictions on the testing data
prediction_rf_df_sq <- predict(model_rf_df_sq_fs, newdata = df_testing_sq)
# Convert df_testing_sq$status to a factor with the same levels as prediction_rf_df_sq
df_testing_sq$status <- factor(df_testing_sq$status, levels = levels(prediction_rf_df_sq))
# Compute confusion matrix
cm_rf_df_sq <- confusionMatrix(prediction_rf_df_sq, df_testing_sq$status, positive = "M")
cm_rf_df_sq

# Compute the accuracy, sensitivity, specificity, precision, recall, F1-score, and AUC-ROC of the model
accuracy_rf_sq <- cm_rf_df_sq$overall[1]
sensitivity_rf_sq <- cm_rf_df_sq$byClass[1]
specificity_rf_sq <- cm_rf_df_sq$byClass[2]
precision_rf_sq <- cm_rf_df_sq$byClass[3]
recall_rf_sq <- cm_rf_df_sq$byClass[1]
f1_score_rf_sq <- (2 * precision_rf_sq * recall_rf_sq) / (precision_rf_sq + recall_rf_sq)
# Compute AUC-ROC
probabilities_rf_sq <- predict(model_rf_df_sq_fs, newdata = df_testing_sq, type = "prob")
auc_roc_rf_sq <- roc(df_testing_sq$status, probabilities_rf_sq[, "M"])$auc

# Print the evaluation metrics
cat(paste0("Random forest quadratic model_Accuracy: ", round(accuracy_rf_sq, 4), "\n"))
cat(paste0("Random forest quadratic model_Sensitivity: ", round(sensitivity_rf_sq, 4), "\n"))
cat(paste0("Random forest quadratic model_Specificity: ", round(specificity_rf_sq, 4), "\n"))
cat(paste0("Random forest quadratic model_Precision: ", round(precision_rf_sq, 4), "\n"))
cat(paste0("Random forest quadratic model_Recall: ", round(recall_rf_sq, 4), "\n"))
cat(paste0("Random forest quadratic model_F1-score: ", round(f1_score_rf_sq, 4), "\n"))
cat(paste0("Random forest quadratic model_AUC-ROC: ", round(auc_roc_rf_sq, 4), "\n"))


# Calculate variable importance scores for the glmnet model with squared predictors
var_imp_sq_rf <- varImp(model_rf_df_sq_fs)
# Extract the variable importance scores and sort in descending order
print(var_imp_sq_rf)
# Calculate variable importance scores for the original glmnet model
var_imp_rf <- varImp(model_rf_df)
# Print out the 10 most influential variables for the original model
print(var_imp_rf)


##### SUMMARY DATAFRAME ####
### Create a summary datafame for all the models
accuracy <- c(accuracy_lr, accuracy_lr_sq, accuracy_knn, accuracy_knn_sq, accuracy_dt, accuracy_dt_sq, accuracy_svm, accuracy_svm_sq, accuracy_bc, accuracy_bc_sq, accuracy_rf, accuracy_rf_sq)

precision <- c(precision_lr, precision_lr_sq, precision_knn, precision_knn_sq, precision_dt, precision_dt_sq, precision_svm, precision_svm_sq, precision_bc, precision_bc_sq, precision_rf, precision_rf_sq)

recall <- c(recall_lr, recall_lr_sq, recall_knn, recall_knn_sq, recall_dt, recall_dt_sq, recall_svm, recall_svm_sq, recall_bc, recall_bc_sq, recall_rf, recall_rf_sq)

f1_score <- c(f1_score_lr, f1_score_lr_sq, f1_score_knn, f1_score_knn_sq, f1_score_dt, f1_score_dt_sq, f1_score_svm, f1_score_svm_sq, f1_score_bc, f1_score_bc_sq, f1_score_rf, f1_score_rf_sq)

auc_roc <- c(auc_roc_lr, auc_roc_lr_sq, auc_roc_knn, auc_roc_knn_sq, auc_roc_dt, auc_roc_dt_sq, auc_roc_svm, auc_roc_svm_sq, auc_roc_bc, auc_roc_bc_sq, auc_roc_rf, auc_roc_rf_sq)

model_name <- c("Elastic_net", "Elastic_net_quadratic", "KNN", "KNN_quadratic", "CART", "CART_quadratic", "SVM", "SVM_quadratic", "Bagged_CART", "Bagged_CART_quadratic", "Random_Forest", "Random_Forest_quadratic")

summary_df <- as.data.frame(cbind(model_name, accuracy, precision, recall, f1_score, auc_roc))

rownames(summary_df) <- model_name
print(summary)

#Table 1: Classification results of the models shows the results of the evaluation metrics for both the actual models and the quadratic covariate effects models for further comparative analysis

#######################################  FIGURES ANALYSED ################
#The different figures used for the manuscript have been interpreted below ----------------------------------------------------------------.

#### Class distribution
class_training <- ggplot(df_training, aes(x = status, fill = status)) +
  geom_bar() +
  labs(title = "Training dataset")

class_testing <- ggplot(df_testing, aes(x = status, fill = status)) +
  geom_bar() +
  labs(title = "Testing dataset")

grid.arrange(class_training, class_testing, ncol=2, nrow =1, top = textGrob("Bar plots showing class distribution of cancer status",gp=gpar(fontsize=15,font=3)))
## Figure 1:Bar plots showing class distribution of cancer status.
## Interpretation: The class distribution in both the training and testing dataset shows an imbalanced distribution of the response variable cancer status, where the majority of the samples are classified as Benign (61%) and the remaining samples as Malignant (39%).


#### Correlation plot
corrplot::corrplot(df_corr, order = "hclust", tl.cex = 1, addrect = 8)
title(main = "Correlation plot of all covariates")
## Figure 2 Correlation matrix plot between features.
## Interpretation: Figure 2 shows that indeed there are quite a few variables that are correlated.


#### PCA plot
PCA2 <- autoplot(preproc_pca_df2, data = df_training,  colour = 'status',
                 loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue")+ ggtitle("PCA plot showing loading vectors")
grid.arrange(cumulative_PCA2, PCA2, ncol=2, nrow =1, top = textGrob("Principal Component Analysis",gp=gpar(fontsize=15,font=3)))
## Figure 3: a: Cumulative plot of explained variance along principal components. b: PCA plot showing loading vectors with influential predictors.
## Interpretation: PCA analysis was performed to reduce the dimensionality of the dataset and identify the most important predictors that explain the variation in the data. The result of the PCA analysis indicates that only 8 principal components were needed to explain 95% of the variance in the data, suggesting that the dataset can be effectively represented with a lower number of dimensions (Figure 3a). The finding that the first 2 principal components (PC1 and PC2) managed to separate the cancer status well compared to the other components suggests that these components contain the most relevant information for predicting the cancer status (Figure b3). A detailed analysis of what variables are the most influential in the first 2 components showed that some variables may be associated with the cancer status prediction (Figure 3b). However, it is important to note that the interpretation of the principal components can be complex and may not always be straightforward.

#### Confusion Matrices
# elastic net
df_cm_logreg_df <- as.data.frame(cm_logreg_df$table)
colnames(df_cm_logreg_df) <- c("Predicted", "Actual", "Count")
# plot the confusion matrix with colors
confusion_logreg <- ggplot(data = df_cm_logreg_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "green") +
  geom_text(aes(label = Count)) +
  theme_minimal() +
  ggtitle("Elastic net")

# knn
df_cm_knn <- as.data.frame(cm_knn_df$table)
colnames(df_cm_knn) <- c("Predicted", "Actual", "Count")
# plot the confusion matrix with colors
confusion_knn <- ggplot(data = df_cm_knn, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "green") +
  geom_text(aes(label = Count)) +
  theme_minimal() +
  ggtitle("KNN")

# CART
df_cm_rpart <- as.data.frame(cm_rpart_df$table)
colnames(df_cm_rpart) <- c("Predicted", "Actual", "Count")
# plot the confusion matrix with colors
confusion_rpart <- ggplot(data = df_cm_rpart, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "green") +
  geom_text(aes(label = Count)) +
  theme_minimal() +
  ggtitle("CART")

# SVM
df_cm_svm <- as.data.frame(cm_svm_df$table)
colnames(df_cm_svm) <- c("Predicted", "Actual", "Count")
# plot the confusion matrix with colors
confusion_svm <- ggplot(data = df_cm_svm, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "green") +
  geom_text(aes(label = Count)) +
  theme_minimal() +
  ggtitle("SVM")

# Bagged CART
df_cm_bagged_cart <- as.data.frame(cm_bagged_cart_df$table)
colnames(df_cm_bagged_cart) <- c("Predicted", "Actual", "Count")
# plot the confusion matrix with colors
confusion_bagged_cart <- ggplot(data = df_cm_bagged_cart, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "green") +
  geom_text(aes(label = Count)) +
  theme_minimal() +
  ggtitle("Bagged CART")

# Random Forest
df_cm_rf <- as.data.frame(cm_rf_df$table)
colnames(df_cm_rf) <- c("Predicted", "Actual", "Count")
# plot the confusion matrix with colors
confusion_rf <- ggplot(data = df_cm_rf, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "green") +
  geom_text(aes(label = Count)) +
  theme_minimal() +
  ggtitle("Random Forest")


grid.arrange(confusion_logreg, confusion_knn, confusion_rpart, confusion_svm, confusion_bagged_cart, confusion_rf,  ncol=3, nrow =2, top = textGrob("Confusion matrices for classification models",gp=gpar(fontsize=20,font=3)))

## Figure 4: Confusion matrix for Cancer status prediction using all models.
## Interpretation From the results, we can see that all models have a high number of true positives, indicating that they are able to predict malignant cases accurately (Figure 4). However, some models have a higher number of false positives than others. For instance, KNN have a higher number of false positives than the other models, indicating that it may wrongly predict benign cases as malignant. On the other hand, CART, Bagged CART, Random Forest and SVM have a lower number of false positives, indicating that they are better at predicting benign cases correctly.Overall, the results suggest that Random Forest and SVM may be the better models as they have a high number of true positives and a low number of false positives, indicating that they are better at predicting both malignant and benign cases correctly. However, the final decision should be made by considering other performance metrics, such as accuracy, precision, recall, and F1 score, in addition to the confusion matrix to make an informed decision about the best model for making prediction. 




#### Learning curves
# elastic net
control_lc <- trainControl(method="repeatedcv", number = 10,repeats = 3, classProbs = TRUE)

set.seed(1815)
learn_lg <- learning_curve_dat(dat =  df_training,
                               outcome = "status",
                               test_prop = 1/10,
                               method = "glmnet",
                               preProc = c("center", "scale"),
                               metric = "Accuracy",
                               trControl = control_lc)

lc_lg <- ggplot(learn_lg, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw() +
  labs(title = "Elastic net")

#knn
set.seed(1815)
learn_knn <- learning_curve_dat(dat =  df_training,
                                outcome = "status",
                                test_prop = 1/10,
                                method = "knn",
                                preProc = c("center", "scale"),
                                metric = "Accuracy",
                                trControl = control_lc)

lc_knn <- ggplot(learn_knn, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw() +
  labs(title = "KNN")

## CART
set.seed(1815)
learn_CART <- learning_curve_dat(dat =  df_training,
                                 outcome = "status",
                                 test_prop = 1/10,
                                 method = "rpart",
                                 preProc = c("center", "scale"),
                                 metric = "Accuracy",
                                 trControl = control_lc)

lc_CART <- ggplot(learn_CART, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw() +
  labs(title = "CART")


## bagged CART
set.seed(1815)
learn_bc <- learning_curve_dat(dat =  df_training,
                               outcome = "status",
                               test_prop = 1/10,
                               method = "treebag",
                               preProc = c("center", "scale"),
                               metric = "Accuracy",
                               trControl = control_lc)

lc_bc <- ggplot(learn_bc, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw() +
  labs(title = "Bagged CART")

## Random forest
set.seed(1815)
learn_rf <- learning_curve_dat(dat =  df_training,
                               outcome = "status",
                               test_prop = 1/10,
                               method = "rf",
                               preProc = c("center", "scale"),
                               metric = "Accuracy",
                               trControl = control_lc)

lc_rf <- ggplot(learn_rf, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw() +
  labs(title = "Random Forest")

## SVM
set.seed(1815)

learn_SVM <- learning_curve_dat(dat = df_training,
                                outcome = "status",
                                test_prop = 1/10,
                                preProc = c("center", "scale"),
                                method = "svmLinear",
                                metric = "Accuracy",
                                trcontrol = control_lc)

lc_SVM <- ggplot(learn_SVM, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw() +
  labs(title = "SVM")

grid.arrange(lc_lg, lc_knn, lc_CART,  ncol=1, nrow =3, top = textGrob("Learning curves for Elastic net, KNN and CART",gp=gpar(fontsize=15,font=3)))

grid.arrange(lc_SVM, lc_bc, lc_rf,ncol=1, nrow =3, top = textGrob("Learning curves for SVM, bagged CART and Random Forest",gp=gpar(fontsize=15,font=3)))

## Figure 6: Learning curves for SVM, bagged CART AND Random Forest.
## Figure 7: Learning curves for Elastic net, KNN and CART.
## Interpretation: A learning curve is a graphical representation of the model's performance as a function of the training set size or the number of iterations. It is used to diagnose if the model is under-fitting (high bias) or over-fitting (high variance) and to determine if adding more data or iterations will improve the model's performance. In this study, the performance of six classification models has been evaluated based on their training score, testing score, and resampling score. The training score indicates how well the model fits the training data, while the testing score indicates how well the model performs on unseen data. The resampling score represents the mean cross-validation score of the model, indicating how well the model generalizes to new data. According to the learning curve, the SVM, Bagged CART, Random Forest, Elastic net, KNN, and CART models all perform relatively well on this classification task (Figures 6 and 7). However, the SVM and Elastic net had the highest resampling scores (Elastic net=0.9675, SVM=0.96) which were not far from the training and testing score indicating that both models generalize well to new data compared to the others (Figures 6 and 7).



#### Model evaluation plot
set.seed(1815)
# Create a list of models that were trained previously
model_list <- list(
  elastic_net = model_logreg,  # Logistic regression model
  rf = model_rf_df,  # Random forest model
  svm = model_svm_df,  # Support vector machine model
  knn = model_knn_df,  # K-nearest neighbors model
  CART = model_rpart_df,  # Classification and regression trees model
  Bagged_CART = model_bagged_cart_df  # Bagged classification and regression trees model
)
# Calculate resamples for each model in the list
results <- resamples(model_list)
# Summarize the resampled results
summary(results)
# Generate a box-and-whisker plot of the resampled results
bwplot(results, metric = "ROC", main = "Model evalution (ROC scores) for classification models")

##Figure 5: Model performance for all models for Cancer status prediction. 
## Interpretation: The Receiver Operating Characteristic (ROC) curve is a graphical representation of the performance of a binary classification model at different thresholds. The area under the curve (AUC) is a measure of the performance of the model, where a higher AUC indicates a better performance. The ROC-AUC score measures the ability of the model to distinguish between positive and negative classes. In this study, the SVM model had the highest ROC-AUC score of 0.9933, indicating that it has the best performance in predicting breast cancer status. This was followed by Elastic_net and Random Forest with a score of 0.988 and 0.9866 respectively (Table 1 and Figure 5). However, the CART model had the lowest AUC score of 0.9413, indicating that it may not be the best model for this problem (Table 1 and Figure 5). Overall, the ROC curve and AUC analysis results support the findings of the previous sections, indicating that the SVM and Elastic net are more suitable for this cancer status classification problem, with SVM showing the best performance. It is worth noting that the ROC curve and AUC analysis provide an important performance evaluation for binary classification models (Gigliarano et al., 2014), and can provide additional insight beyond the traditional performance metrics such as accuracy, precision, recall, and F1-score.


## References
#1 Bauer, E., & Kohavi, R. (1999). An empirical comparison of voting classification algorithms: Bagging, boosting, and variants. Machine learning, 36, 105-139.
#2 Bray, F., Jemal, A., Grey, N., Ferlay, J., & Forman, D. (2021). Global cancer transitions according to the Human Development Index (2008–2030): a population-based study. The Lancet Oncology, 22(2), 205-215.
#3 Chowdhury, M. Z. I., & Turin, T. C. (2020). Variable selection strategies and its importance in clinical prediction modelling. Family medicine and community health, 8(1). 
#4 Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2019). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.
#5 Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. O'Reilly Media.
#6 Gigliarano, C., Figini, S., & Muliere, P. (2014). Making classifier performance comparisons when ROC curves intersect. Computational Statistics & Data Analysis, 77, 300-312. 
#7 Guo, G., Wang, H., Bell, D., Bi, Y., & Greer, K. (2003). KNN model-based approach in classification. Lecture Notes in Computer Science, 2888, 986–996
#8 Hicks, S. A., Strümke, I., Thambawita, V., Hammou, M., Riegler, M. A., Halvorsen, P., & Parasa, S. (2022). On evaluation metrics for medical applications of artificial intelligence. Scientific Reports, 12(1), 5979.  
#9 Lewis, R. J. (2000). An introduction to classification and regression tree (CART) analysis. Annual meeting of the society for academic emergency medicine in San Francisco, California (Vol. 14). California: Department of Emergency Medicine Harbor-UCLA Medical Center Torrance.
#10 Noble, W. S. (2006). What is a support vector machine? Nature Biotechnology, 24(12), 1565–1567.
#11 Owen, A. B. (2007). A robust hybrid of lasso and ridge regression. Contemporary Mathematics, 443(7), 59-72.
#12 Ozsahin, D. U., Mustapha, M. T., Mubarak, A. S., Ameen, Z. S., & Uzun, B. (2022). Impact of feature scaling on machine learning models for the diagnosis of diabetes. 2022 International Conference on Artificial Intelligence in Everything (AIE) (pp. 87-94). IEEE.
#13 Prabhakaran, S. (2018). Caret Package – A Practical Guide to Machine Learning in R. Machinelearningplus. https://www.machinelearningplus.com/machine-learning/caret-package/
#14 Rodriguez-Galiano, V. F., Ghimire, B., Rogan, J., Chica-Olmo, M., & Rigol-Sanchez, J. P. (2012). An assessment of the effectiveness of a random forest classifier for land-cover classification. ISPRS journal of photogrammetry and remote sensing, 67, 93-104.
#15 Wang, J., Yang, X., Cai, H., Tan, W., Jin, C., Zhou, J., & Cheng, L. (2020). Development and validation of a deep learning algorithm for improving Gleason scoring of prostate cancer. Nature Communications, 11(1), 1-9.
#16 Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net (vol B 67, pg 301, 2005). Journal of the Royal Statistical Society. Series B, Statistical Methodology, 67, 768–768. 

#THANK YOU
