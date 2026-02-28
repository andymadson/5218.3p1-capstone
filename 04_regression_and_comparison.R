# ============================================
# Session 4: Regression & Model Comparison
# Time: 1:00 - 2:15 PM
# ============================================
# Topics: Week 6 (logistic regression,
#         variable selection)
# Prerequisite: Run 01, 02, and 03 scripts first
# ============================================

library(caret)
library(ggplot2)

# ── Part A: Logistic Regression ──────────────

# Step 1: Fit logistic regression on numeric features
logit_model <- glm(
  target ~ .,
  data = train_num,
  family = binomial(link = "logit")
)

summary(logit_model)

# Step 2: Stepwise variable selection
logit_step <- step(logit_model,
                   direction = "both",
                   trace = 0)  # Suppress output

cat("\nFull model AIC:",
    AIC(logit_model), "\n")
cat("Stepwise model AIC:",
    AIC(logit_step), "\n")
cat("\nSelected variables:\n")
cat(names(coef(logit_step)), sep = "\n")

# Step 3: Evaluate logistic regression
logit_probs <- predict(logit_step,
                        newdata = test_num,
                        type = "response")

logit_preds <- as.factor(
  ifelse(logit_probs > 0.5, 1, 0))

cm_logit <- table(Predicted = logit_preds,
                   Actual = test_num$target)
acc_logit <- sum(diag(cm_logit)) / sum(cm_logit)
cat("\n== Logistic Regression (Stepwise) ==\n")
print(cm_logit)
cat("Accuracy:", round(acc_logit * 100, 2),
    "%\n")


# ── Part B: ROC Curves & AUC ────────────────

# Function to compute ROC data
roc_data <- function(actual, probs, model_name) {
  actual_num <- as.numeric(
    as.character(actual))
  thresholds <- seq(0, 1, by = 0.01)
  tpr <- sapply(thresholds, function(t) {
    preds <- ifelse(probs >= t, 1, 0)
    sum(preds == 1 & actual_num == 1) /
      max(sum(actual_num == 1), 1)
  })
  fpr <- sapply(thresholds, function(t) {
    preds <- ifelse(probs >= t, 1, 0)
    sum(preds == 1 & actual_num == 0) /
      max(sum(actual_num == 0), 1)
  })
  data.frame(FPR = fpr, TPR = tpr,
             Model = model_name)
}

# Get probabilities for each model
tree_probs <- predict(tree_model,
                      test_data, type = "prob")[, 2]
rf_probs <- predict(rf_model,
                    test_data, type = "prob")[, 2]
# gbm_probs already computed in 03 script
# logit_probs already computed above

# Build combined ROC data frame
roc_all <- rbind(
  roc_data(test_data$target, tree_probs,
           "Decision Tree"),
  roc_data(test_data$target, rf_probs,
           "Random Forest"),
  roc_data(test_data$target, gbm_probs,
           "Gradient Boosting"),
  roc_data(test_num$target, logit_probs,
           "Logistic Regression")
)

# Plot all ROC curves together
ggplot(roc_all, aes(x = FPR, y = TPR,
                    color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed",
              color = "gray50") +
  labs(title = "ROC Curves: All Models Compared",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal() +
  theme(legend.position = "bottom")


# ── Part C: Model Comparison Summary ─────────

results <- data.frame(
  Model = c("Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression"),
  Accuracy = round(c(acc_tree, acc_rf,
                     acc_gbm, acc_logit) * 100, 2)
)
results <- results[order(-results$Accuracy), ]

# Bar chart comparison
ggplot(results,
       aes(x = reorder(Model, Accuracy),
           y = Accuracy, fill = Model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(Accuracy, "%")),
            hjust = -0.1) +
  coord_flip() +
  ylim(0, 100) +
  labs(title = "Model Accuracy Comparison",
       x = "", y = "Test Accuracy (%)") +
  theme_minimal()
