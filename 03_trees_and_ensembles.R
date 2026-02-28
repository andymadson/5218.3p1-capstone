# ============================================
# Session 3: Trees & Ensembles
# Time: 11:00 AM - 12:00 PM
# ============================================
# Topics: Week 3 (decision trees),
#         Week 4 (model validation),
#         Week 5 (random forests, boosting)
# Prerequisite: Run 01 and 02 scripts first
# ============================================

library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)

# ── Reusable Evaluation Function ─────────────

evaluate_model <- function(model, test,
                           model_name,
                           type = "class") {
  preds <- predict(model, test, type = type)
  if (is.matrix(preds) || is.data.frame(preds)) {
    preds <- as.factor(
      ifelse(preds[, 2] > 0.5, 1, 0))
  }
  cm <- table(Predicted = preds,
              Actual = test$target)
  acc <- sum(diag(cm)) / sum(cm)
  cat("\n==", model_name, "==", "\n")
  print(cm)
  cat("Accuracy:", round(acc * 100, 2), "%\n")
  return(acc)
}


# ── Part A: Decision Tree Baseline ───────────

# Step 1: Build a classification tree
tree_model <- rpart(
  target ~ .,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.01,        # Complexity parameter
    minsplit = 10,     # Min obs to attempt split
    maxdepth = 5       # Max tree depth
  )
)

# Visualize the tree
rpart.plot(tree_model,
           main = "Heart Disease Decision Tree",
           extra = 104,   # Show % and count
           roundint = FALSE)

# Step 2: Evaluate on the test set
acc_tree <- evaluate_model(tree_model,
                           test_data, "Decision Tree")


# ── Part B: Random Forest ────────────────────

# Step 3: Build a random forest
set.seed(2026)
rf_model <- randomForest(
  target ~ .,
  data = train_data,
  ntree = 500,
  mtry = 3,  # Features sampled per split
  importance = TRUE
)

print(rf_model)

# Variable importance plot
varImpPlot(rf_model,
           main = "Random Forest Variable Importance")

# Step 4: Evaluate random forest
acc_rf <- evaluate_model(rf_model,
                         test_data, "Random Forest")


# ── Part C: Gradient Boosting ────────────────

# Step 5: Build a gradient boosting model
# GBM needs numeric target (0/1)
train_gbm <- train_data
train_gbm$target <- as.numeric(
  as.character(train_data$target))

set.seed(2026)
gbm_model <- gbm(
  target ~ .,
  data = train_gbm,
  distribution = "bernoulli",
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  cv.folds = 5
)

# Find optimal number of trees
best_trees <- gbm.perf(gbm_model,
                        method = "cv")
cat("Optimal trees:", best_trees, "\n")

# Step 6: Evaluate gradient boosting
gbm_probs <- predict(gbm_model,
                      newdata = test_data,
                      n.trees = best_trees,
                      type = "response")

gbm_preds <- as.factor(
  ifelse(gbm_probs > 0.5, 1, 0))

cm_gbm <- table(Predicted = gbm_preds,
                 Actual = test_data$target)
acc_gbm <- sum(diag(cm_gbm)) / sum(cm_gbm)
cat("\n== Gradient Boosting ==\n")
print(cm_gbm)
cat("Accuracy:", round(acc_gbm * 100, 2),
    "%\n")


# ── Tuning Ideas (for Team Tournament) ──────
#
# Decision Tree:
#   - Change cp to 0.001 (more complex trees)
#   - Adjust maxdepth to 8 or 10
#   - Use printcp(tree_model) and prune to optimal cp
#
# Random Forest:
#   - Try mtry values from 2 to 6
#   - Use tuneRF() for automated search
#   - Increase ntree to 1000
#
# Gradient Boosting:
#   - Try shrinkage = 0.01 with n.trees = 2000
#   - Adjust interaction.depth to 4 or 5
