# ============================================
# Session 7: Build Your First Neural Network
# Final Exercise
# ============================================
# Topics: Neural networks, backpropagation,
#         architecture visualization,
#         Classical ML vs Neural Networks
# Prerequisite: Run scripts 01–04 first
#               (needs train_num, test_num,
#                test_data, evaluate_model,
#                acc_tree, acc_rf, acc_gbm,
#                acc_logit)
# ============================================


# ── Verify environment from earlier sessions ─

cat("=== Environment Check ===\n")
cat("train_num exists (Session 2):",
    exists("train_num"), "\n")
cat("test_num exists  (Session 2):",
    exists("test_num"), "\n")
cat("train_data exists (Session 2):",
    exists("train_data"), "\n")
cat("test_data exists  (Session 2):",
    exists("test_data"), "\n")
cat("evaluate_model exists (Session 3):",
    exists("evaluate_model"), "\n")
cat("acc_tree exists (Session 3):",
    exists("acc_tree"), "\n")
cat("acc_rf exists   (Session 3):",
    exists("acc_rf"), "\n")
cat("acc_gbm exists  (Session 3):",
    exists("acc_gbm"), "\n")
cat("acc_logit exists (Session 4):",
    exists("acc_logit"), "\n")

# Quick data shape check
cat("\ntrain_num:", nrow(train_num), "rows,",
    ncol(train_num), "columns\n")
cat("test_num: ", nrow(test_num), "rows,",
    ncol(test_num), "columns\n")
cat("Target levels:", levels(test_data$target),
    "\n")
cat("=========================\n")


# ── Install & load new packages ──────────────

install.packages(c(
  "neuralnet",       # Neural network engine
  "NeuralNetTools"   # Architecture visualization
), repos = "https://cloud.r-project.org")

library(neuralnet)
library(NeuralNetTools)
library(ggplot2)  # Already loaded, but be safe


# ── Neural network evaluation helper ─────────
# The evaluate_model() from Session 3 expects a
# model object and calls predict() internally.
# Neural networks use compute(), so we need a
# wrapper that accepts pre-computed predictions.

evaluate_nn <- function(preds, actuals,
                        model_name) {
  cm <- table(Predicted = preds,
              Actual = actuals)
  acc <- sum(diag(cm)) / sum(cm)
  cat("\n==", model_name, "==", "\n")
  print(cm)
  cat("Accuracy:", round(acc * 100, 2), "%\n")
  return(acc)
}


# ═══════════════════════════════════════════════
# PART A: Your First Neural Network
# (Instructor-Led, 25 min)
# ═══════════════════════════════════════════════


# ── Step 1: Normalize features to 0–1 range ──

feature_cols <- setdiff(names(train_num), "target")
cat("Input features:", length(feature_cols), "\n")
cat("First few:", head(feature_cols, 8), "...\n")

# Compute min and max from TRAINING data only.
# Never use test data statistics — that is leakage!
train_mins <- sapply(train_num[, feature_cols], min)
train_maxs <- sapply(train_num[, feature_cols], max)
train_ranges <- train_maxs - train_mins

# Handle zero-range columns (constant features)
train_ranges[train_ranges == 0] <- 1

# Normalize training features
train_nn <- as.data.frame(scale(
  train_num[, feature_cols],
  center = train_mins,
  scale = train_ranges
))

# Normalize test features using TRAINING min/max
test_nn <- as.data.frame(scale(
  test_num[, feature_cols],
  center = train_mins,
  scale = train_ranges
))

# Add target as numeric 0/1
train_nn$target <- as.numeric(
  as.character(train_num$target))
test_nn$target <- as.numeric(
  as.character(test_num$target))

# Verify
cat("\nTraining set: ", nrow(train_nn), "rows,",
    ncol(train_nn), "cols\n")
cat("Test set:     ", nrow(test_nn), "rows,",
    ncol(test_nn), "cols\n")
cat("Feature range: [",
    round(min(train_nn[, feature_cols]), 3), ",",
    round(max(train_nn[, feature_cols]), 3), "]\n")
cat("Target values:", sort(unique(train_nn$target)),
    "\n")


# ── Step 2: Build a simple neural network ────

# Build the formula dynamically
nn_formula <- as.formula(paste(
  "target ~",
  paste(feature_cols, collapse = " + ")
))

cat("Formula (first 80 chars):\n")
cat(substr(deparse(nn_formula, width.cutoff = 500),
           1, 80), "...\n")
cat("\nTotal input features:", length(feature_cols),
    "\n")

# Train the neural network
set.seed(2026)
nn_model <- neuralnet(
  nn_formula,
  data = train_nn,
  hidden = c(5),         # 1 hidden layer, 5 neurons
  linear.output = FALSE, # Classification (sigmoid)
  threshold = 0.05,      # Stop when error change < this
  stepmax = 1e5,         # Max training iterations
  lifesign = "full",     # Show training progress
  lifesign.step = 1000,  # Print every 1000 steps
  act.fct = "logistic"   # Sigmoid activation
)

cat("\nTraining complete!\n")
cat("Steps to convergence:",
    nn_model$result.matrix["steps", ], "\n")
cat("Final error:",
    round(nn_model$result.matrix["error", ], 6),
    "\n")


# ── Step 3: Visualize the network architecture ─

# Base plot: every neuron and connection weight
plot(nn_model,
     rep = "best",
     col.entry = "#2E86AB",
     col.hidden = "#E74C3C",
     col.out = "#27AE60",
     show.weights = TRUE,
     information = TRUE
)

# NeuralNetTools: cleaner, publication-quality
plotnet(nn_model,
        pos_col = "#E74C3C",  # Positive weights = red
        neg_col = "#3498DB",  # Negative weights = blue
        alpha_val = 0.7,
        circle_cex = 4,
        cex_val = 0.6,
        max_sp = TRUE
)
title("Neural Network Architecture: 5-Neuron Hidden Layer",
      line = 2.5, cex.main = 1.2)


# ── Step 4: Evaluate the neural network ──────

nn_output <- compute(nn_model,
                     test_nn[, feature_cols])

nn_probs <- nn_output$net.result[, 1]

cat("Predicted probability range:",
    round(min(nn_probs), 4), "to",
    round(max(nn_probs), 4), "\n")

# Convert to class predictions at the 0.5 threshold
nn_preds <- as.factor(ifelse(nn_probs > 0.5, 1, 0))

acc_nn <- evaluate_nn(
  nn_preds,
  test_data$target,
  "Neural Network (1 layer, 5 neurons)"
)


# ── Step 5: Variable importance ──────────────

# Garson's algorithm
importance_nn <- garson(nn_model) +
  ggtitle(
    "Neural Network: Which Features Matter Most?"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(
    angle = 45, hjust = 1, size = 9))

print(importance_nn)

# Olden's method (signed importance)
olden_plot <- olden(nn_model) +
  ggtitle("Signed Feature Importance (Olden)") +
  theme_minimal() +
  theme(axis.text.x = element_text(
    angle = 45, hjust = 1, size = 9))

print(olden_plot)


# ── Step 6: Quick leaderboard ────────────────

leaderboard <- data.frame(
  Model = c(
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
    "Neural Network (5)"
  ),
  Accuracy = round(c(
    acc_tree, acc_rf, acc_gbm,
    acc_logit, acc_nn
  ) * 100, 2)
)

leaderboard <- leaderboard[
  order(-leaderboard$Accuracy), ]

cat("\n=== UPDATED TOURNAMENT LEADERBOARD ===\n")
print(leaderboard, row.names = FALSE)
cat("======================================\n")


# ═══════════════════════════════════════════════
# PART B: Team Architecture Challenge (25 min)
# Try as many as time allows!
# ═══════════════════════════════════════════════


# ── Challenge 1: Two Hidden Layers (8-4) ─────

set.seed(2026)
nn_deep <- neuralnet(
  nn_formula,
  data = train_nn,
  hidden = c(8, 4),      # Layer 1: 8, Layer 2: 4
  linear.output = FALSE,
  threshold = 0.05,
  stepmax = 1e5,
  lifesign = "full",
  lifesign.step = 2000,
  act.fct = "logistic"
)

plotnet(nn_deep,
        pos_col = "#E74C3C",
        neg_col = "#3498DB",
        alpha_val = 0.7,
        circle_cex = 3,
        cex_val = 0.5)
title("Deep Neural Network: 8-4 Architecture",
      line = 2.5)

nn_deep_out <- compute(nn_deep,
                       test_nn[, feature_cols])
nn_deep_preds <- as.factor(
  ifelse(nn_deep_out$net.result[, 1] > 0.5, 1, 0))

acc_nn_deep <- evaluate_nn(
  nn_deep_preds,
  test_data$target,
  "Neural Network (8-4, two layers)"
)


# ── Challenge 2: One Wide Layer (15 neurons) ─

set.seed(2026)
nn_wide <- neuralnet(
  nn_formula,
  data = train_nn,
  hidden = c(15),        # 1 layer, 15 neurons
  linear.output = FALSE,
  threshold = 0.05,
  stepmax = 1e5,
  lifesign = "full",
  lifesign.step = 2000,
  act.fct = "logistic"
)

plotnet(nn_wide,
        pos_col = "#E74C3C",
        neg_col = "#3498DB",
        alpha_val = 0.5,
        circle_cex = 2.5,
        cex_val = 0.4)
title("Wide Neural Network: 15 Neurons", line = 2.5)

nn_wide_out <- compute(nn_wide,
                       test_nn[, feature_cols])
nn_wide_preds <- as.factor(
  ifelse(nn_wide_out$net.result[, 1] > 0.5, 1, 0))

acc_nn_wide <- evaluate_nn(
  nn_wide_preds,
  test_data$target,
  "Neural Network (15 neurons, one layer)"
)


# ── Challenge 3: Three Hidden Layers (10-6-3) ─

set.seed(2026)
nn_3layer <- neuralnet(
  nn_formula,
  data = train_nn,
  hidden = c(10, 6, 3),  # Three hidden layers
  linear.output = FALSE,
  threshold = 0.08,       # Relaxed for convergence
  stepmax = 2e5,          # More steps needed
  lifesign = "full",
  lifesign.step = 5000,
  act.fct = "logistic"
)

plotnet(nn_3layer,
        pos_col = "#E74C3C",
        neg_col = "#3498DB",
        alpha_val = 0.5,
        circle_cex = 2.5,
        cex_val = 0.4)
title("Deep Network: 10-6-3 Architecture", line = 2.5)

nn_3layer_out <- compute(nn_3layer,
                         test_nn[, feature_cols])
nn_3layer_preds <- as.factor(
  ifelse(nn_3layer_out$net.result[, 1] > 0.5, 1, 0))

acc_nn_3layer <- evaluate_nn(
  nn_3layer_preds,
  test_data$target,
  "Neural Network (10-6-3, three layers)"
)


# ── Challenge 4: Multiple Training Runs ──────

set.seed(2026)
nn_multi <- neuralnet(
  nn_formula,
  data = train_nn,
  hidden = c(8, 4),
  linear.output = FALSE,
  threshold = 0.05,
  stepmax = 1e5,
  rep = 5,                # Train 5 times, keep best
  lifesign = "minimal",
  act.fct = "logistic"
)

cat("\nErrors across 5 repetitions:\n")
for (i in 1:5) {
  err <- nn_multi$result.matrix["error", i]
  cat(sprintf("  Rep %d: %.6f\n", i, err))
}

nn_multi_out <- compute(nn_multi,
                        test_nn[, feature_cols])
nn_multi_preds <- as.factor(
  ifelse(nn_multi_out$net.result[, 1] > 0.5, 1, 0))

acc_nn_multi <- evaluate_nn(
  nn_multi_preds,
  test_data$target,
  "Neural Network (8-4, best of 5 runs)"
)


# ── Challenge 5: Ensemble Your Neural Networks ─

# Average probabilities from each architecture
p1 <- compute(nn_model,
              test_nn[, feature_cols])$net.result[, 1]
p2 <- compute(nn_deep,
              test_nn[, feature_cols])$net.result[, 1]
p3 <- compute(nn_wide,
              test_nn[, feature_cols])$net.result[, 1]

avg_probs <- (p1 + p2 + p3) / 3

nn_ensemble_preds <- as.factor(
  ifelse(avg_probs > 0.5, 1, 0))

acc_nn_ensemble <- evaluate_nn(
  nn_ensemble_preds,
  test_data$target,
  "Neural Network Ensemble (3 architectures)"
)


# ═══════════════════════════════════════════════
# FINAL LEADERBOARD: Classical ML vs Neural Nets
# ═══════════════════════════════════════════════

# Start with the four classical models
model_names <- c(
  "Decision Tree", "Random Forest",
  "Gradient Boosting", "Logistic Regression"
)
model_accs <- c(
  acc_tree, acc_rf, acc_gbm, acc_logit
)
model_types <- rep("Classical ML", 4)

# Add baseline neural network (Part A)
model_names <- c(model_names,
                 "Neural Net (5 neurons)")
model_accs  <- c(model_accs, acc_nn)
model_types <- c(model_types, "Neural Network")

# Add team challenge models if they exist
if (exists("acc_nn_deep")) {
  model_names <- c(model_names,
                   "Neural Net (8-4 deep)")
  model_accs  <- c(model_accs, acc_nn_deep)
  model_types <- c(model_types, "Neural Network")
}
if (exists("acc_nn_wide")) {
  model_names <- c(model_names,
                   "Neural Net (15 wide)")
  model_accs  <- c(model_accs, acc_nn_wide)
  model_types <- c(model_types, "Neural Network")
}
if (exists("acc_nn_3layer")) {
  model_names <- c(model_names,
                   "Neural Net (10-6-3)")
  model_accs  <- c(model_accs, acc_nn_3layer)
  model_types <- c(model_types, "Neural Network")
}
if (exists("acc_nn_multi")) {
  model_names <- c(model_names,
                   "Neural Net (8-4 x5 runs)")
  model_accs  <- c(model_accs, acc_nn_multi)
  model_types <- c(model_types, "Neural Network")
}
if (exists("acc_nn_ensemble")) {
  model_names <- c(model_names,
                   "NN Ensemble (3 nets)")
  model_accs  <- c(model_accs, acc_nn_ensemble)
  model_types <- c(model_types, "Neural Network")
}

# Build and sort the final leaderboard
all_results <- data.frame(
  Model = model_names,
  Accuracy = round(model_accs * 100, 2),
  Type = model_types
)
all_results <- all_results[
  order(-all_results$Accuracy), ]

cat("\n")
cat(paste(rep("=", 55), collapse = ""), "\n")
cat("   ULTIMATE MODEL LEADERBOARD\n")
cat("   Classical ML vs Neural Networks\n")
cat(paste(rep("=", 55), collapse = ""), "\n")
print(all_results, row.names = FALSE)
cat(paste(rep("=", 55), collapse = ""), "\n")

# Color-coded bar chart
ggplot(all_results,
       aes(x = reorder(Model, Accuracy),
           y = Accuracy,
           fill = Type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = paste0(Accuracy, "%")),
    hjust = -0.1, size = 3.5
  ) +
  coord_flip() +
  ylim(0, 100) +
  labs(
    title = "Ultimate Tournament: Classical ML vs Neural Networks",
    subtitle = paste("All models on the same",
                     nrow(test_data), "test patients"),
    x = "", y = "Test Accuracy (%)",
    fill = "Model Type"
  ) +
  scale_fill_manual(values = c(
    "Classical ML"    = "#2E86AB",
    "Neural Network"  = "#E74C3C"
  )) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 11),
    legend.position = "bottom"
  )
