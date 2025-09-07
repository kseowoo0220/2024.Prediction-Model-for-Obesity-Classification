# load libraries
library(tidyverse)
library(caret)
library(glmnet)
library(pROC)
library(randomForest)

set.seed(123)

# load the data
df <- read.csv('/Users/kseowoo/Library/CloudStorage/OneDrive-Personal/UNC CH MS/2024 spring/BISO992/project data/obesity.csv')

# remove cols
df <- df[ , -1] # remove ID
df <- df[ , !(names(df) %in% c("BMIz"))]

# convert categorical variables to factors
cat_vars <- c("gender","race","Family.income.to.poverty.ratio",
              "education","energy","obesity")
df[cat_vars] <- lapply(df[cat_vars], factor)

str(df)


# -------------------------------------------------------
# Exploratory Data Analysis

# Histogram for numeric variables
num_cols <- sapply(df, is.numeric)
numeric_df <- df[, num_cols]

# Number of numeric columns
n <- ncol(numeric_df)
cat("Number of numeric variables:", n, "\n")

# Set plotting layout: 4 columns
pdf('/Users/kseowoo/Library/CloudStorage/OneDrive-Personal/resume project data and code/Obesity/final/histogram.pdf',width=8)
par(mfrow = c(ceiling(n/4), 4), mar=c(4,4,2,1))

# Plot histograms

for(colname in names(numeric_df)) {
  hist(numeric_df[[colname]],
       main = colname,
       xlab = colname,
       col = "skyblue",
       border = "black")
}
dev.off()

# check correlation bewteen features to check multicollinearity
num_cols <- sapply(df, is.numeric)
cor_mat <- cor(df[, num_cols], use = "pairwise.complete.obs")

# check the correlation matrix
print(cor_mat)

# set the heatmap color
col_yr <- colorRampPalette(c("pink", 'white','skyblue'))(200)

# corrplot
corrplot(cor_mat,
         method = "color",
         type = "upper",
         col = col_yr,   
         tl.col = "black",
         tl.cex = 0.8,
         addCoef.col = "black") 


# split the data into train and test set
train.idx <- createDataPartition(df$obesity, p = 0.8, list = FALSE)
train <- df[train.idx, ]
test  <- df[-train.idx, ]

table(train$obesity); table(test$obesity)


# -------------------------------------------------------
# Baseline Logistic Regression
# -------------------------------------------------------
logit_model = glm(obesity~.,data=train,family = 'binomial')
summary(logit_model)
logit_model
library(pROC)

# check multicollinearity
vif_values <- vif(logit_model)

# Get predicted probabilities (for the positive class)
prob <- predict(logit_model, newdata = test, type = "response")

# True labels
true <- test$obesity   # make sure this is 0/1 or factor with levels

# Compute ROC
roc_obj <- roc(true, prob)

# AUC value
auc(roc_obj)


# -------------------------------------------------------
# LASSO Logistic Regression
# -------------------------------------------------------
library(glmnet)
library(pROC)

# Prepare data (convert to matrix, remove intercept column)
x_train <- model.matrix(obesity ~ ., data = train)[, -1]
y_train <- train$obesity
x_test  <- model.matrix(obesity ~ ., data = test)[, -1]
y_test  <- test$obesity

# Cross-validated LASSO (alpha=1)
set.seed(123)
cv_lasso <- cv.glmnet(
  x_train, y_train,
  family = "binomial",
  alpha = 1,              # LASSO
  type.measure = "auc",   # maximize ROC-AUC
  nfolds = 10
)

# Best lambda (gives highest AUC)
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Fit final model at best lambda
lasso_model <- glmnet(
  x_train, y_train,
  family = "binomial",
  alpha = 1,
  lambda = best_lambda
)

# Predict probabilities on test data
prob_lasso <- predict(lasso_model, newx = x_test, type = "response")

# ROC & AUC
roc_lasso <- roc(y_test, as.numeric(prob_lasso))
auc_lasso <- auc(roc_lasso)
cat("LASSO Test AUC:", auc_lasso, "\n")

# Show selected predictors
coef(lasso_model)


# -------------------------------------------------------
# Random Forest: all features vs top-10,15,20,25 by importance
# -------------------------------------------------------
library(randomForest)
library(pROC)
set.seed(123)

# Prepare data: Random Forest requires a factor target,
# but AUC calculation is easier with numeric 0/1.
train_rf <- train
test_rf  <- test

# Create both factor (for RF) and numeric (for AUC) versions of target
train_rf$obesity_f <- factor(ifelse(train_rf$obesity == 'yes', 1, 0), levels = c(0,1))
test_rf$obesity_f  <- factor(ifelse(test_rf$obesity == 'yes', 1, 0), levels = c(0,1))
y_test_num <- as.numeric(as.character(test_rf$obesity_f))  # numeric 0/1 labels

# Train RF model with all predictors
p <- ncol(train_rf) - 2  # subtract target columns
mtry_default <- floor(sqrt(p))

rf_all <- randomForest(
  obesity_f ~ . - obesity,   # remove duplicate target column
  data = train_rf,
  ntree = 1000,
  mtry = mtry_default,
  importance = TRUE
)

# Extract feature importance (MeanDecreaseAccuracy)
imp <- importance(rf_all, type = 1)
imp_tbl <- data.frame(feature = rownames(imp), MDA = imp[,1], row.names = NULL)
imp_tbl <- imp_tbl[order(-imp_tbl$MDA), ]

# Define top-k feature sizes
top_sizes <- c(10,15,20,25)

# Evaluate AUC on test data - function
eval_auc <- function(model, newdata, y_true_num) {
  prob <- predict(model, newdata = newdata, type = "prob")[, "1"]
  auc(roc(y_true_num, prob, quiet = TRUE))
}

# Evaluate AUC with all predictors
auc_all <- eval_auc(rf_all, test_rf, y_test_num)

# Store results
results <- data.frame(model = "RF_all", n_features = p, auc = as.numeric(auc_all))

# Retrain RF models using top-k features and evaluate AUC
for (k in top_sizes) {
  feats_k <- imp_tbl$feature[1:min(k, nrow(imp_tbl))]
  
  # Subset training and test data to top-k features
  train_k <- train_rf[, c("obesity_f", feats_k), drop = FALSE]
  test_k  <- test_rf[,  c("obesity_f", feats_k), drop = FALSE]
  
  rf_k <- randomForest(
    obesity_f ~ .,
    data = train_k,
    ntree = 1000,
    mtry = floor(sqrt(length(feats_k))),
    importance = FALSE
  )
  
  auc_k <- eval_auc(rf_k, test_k, y_test_num)
  results <- rbind(results, data.frame(model = paste0("RF_top", k), n_features = length(feats_k), auc = as.numeric(auc_k)))
}

print(results)


# =======================================================
# Baseline vs LASSO vs RandomForest (BEST) ROC–AUC Compare
# =======================================================
library(pROC)

## Ensure consistent numeric labels for ROC
if (!exists("y_test_num")) {
  y_test_num <- ifelse(test$obesity == "yes", 1, 0)
}

## Baseline logistic ROC (uses `prob` from your baseline code)
roc_logit <- roc(y_test_num, as.numeric(prob), quiet = TRUE)
auc_logit <- as.numeric(auc(roc_logit))

## LASSO ROC (uses `prob_lasso` from your LASSO code)
roc_lasso <- roc(y_test_num, as.numeric(prob_lasso), quiet = TRUE)
auc_lasso <- as.numeric(auc(roc_lasso))

## RF BEST model selection among {all, top10, top20, top30}
# We will recompute AUCs & pick the best one, then keep its ROC for plotting.
# Start with "all features"
prob_all <- predict(rf_all, newdata = test_rf, type = "prob")[, "1"]
roc_all  <- roc(y_test_num, prob_all, quiet = TRUE)
auc_all  <- as.numeric(auc(roc_all))
best_name <- "RF_all"
best_auc  <- auc_all
best_roc  <- roc_all

# Evaluate each top-k
for (k in top_sizes) {
  feats_k <- imp_tbl$feature[1:min(k, nrow(imp_tbl))]
  train_k <- train_rf[, c("obesity_f", feats_k), drop = FALSE]
  test_k  <- test_rf[,  c("obesity_f", feats_k), drop = FALSE]
  
  rf_k <- randomForest(
    obesity_f ~ ., data = train_k,
    ntree = 1000, mtry = floor(sqrt(length(feats_k)))
  )
  
  prob_k <- predict(rf_k, newdata = test_k, type = "prob")[, "1"]
  roc_k  <- roc(y_test_num, prob_k, quiet = TRUE)
  auc_k  <- as.numeric(auc(roc_k))
  
  if (auc_k > best_auc) {
    best_auc <- auc_k
    best_roc <- roc_k
    best_name <- paste0("RF_top", k)
  }
}


## Single ROC plot (baseline vs LASSO vs RF_best)
plot(roc_logit, lwd = 2, main = "ROC: Baseline vs LASSO vs RF (best)",col='steelblue')
plot(roc_lasso, add = TRUE, lwd = 2, col = "firebrick")
plot(best_roc, add = TRUE, lwd = 2, col = "darkgreen")
legend("bottomright",
       legend = c(
         paste0("Logistic (AUC=", round(auc_logit, 3), ")"),
         paste0("LASSO (AUC=",   round(auc_lasso, 3), ")"),
         paste0(best_name, " (AUC=", round(best_auc, 3), ")")
       ),
       col = c("steelblue", "firebrick", "darkgreen"), lwd = 2, bty = "n")

## tidy summary table
comparison_tbl <- tibble::tibble(
  model = c("Logistic (all feats)", "LASSO (λ* = lambda.min)", best_name),
  auc   = c(auc_logit, auc_lasso, best_auc)
)
print(comparison_tbl)
