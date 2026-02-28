# ═══════════════════════════════════════════════
# IS 5213 Capstone Day — Complete Script
# ═══════════════════════════════════════════════
# Run this top-to-bottom in a clean RStudio
# session to reproduce the entire day's work.
# ═══════════════════════════════════════════════


# ── Package Installation ─────────────────────

install.packages(c(
  "ggplot2", "corrplot", "rpart", "rpart.plot",
  "randomForest", "gbm", "caret", "e1071",
  "Rtsne", "cluster", "factoextra", "reshape2"
), repos = "https://cloud.r-project.org")

library(ggplot2); library(corrplot)
library(reshape2); library(rpart)
library(rpart.plot); library(randomForest)
library(gbm); library(caret)
library(Rtsne); library(cluster)
library(factoextra)


# ═══════════════════════════════════════════════
# SESSION 1: Data Loading, EDA & Visualization
# ═══════════════════════════════════════════════

# ── Data Loading ─────────────────────────────

url <- paste0(
  "https://archive.ics.uci.edu/ml/",
  "machine-learning-databases/heart-disease/",
  "processed.cleveland.data"
)

col_names <- c(
  "age", "sex", "cp", "trestbps", "chol",
  "fbs", "restecg", "thalach", "exang",
  "oldpeak", "slope", "ca", "thal", "target"
)

heart <- read.csv(url, header = FALSE,
                  col.names = col_names,
                  na.strings = "?")

cat("Rows:", nrow(heart), "  Columns:", ncol(heart), "\n")

# Inspect
str(heart)
head(heart)
summary(heart)

# Convert target to binary factor
heart$target <- ifelse(heart$target > 0, 1, 0)
heart$target <- as.factor(heart$target)

table(heart$target)
cat("Disease prevalence:",
    round(mean(as.numeric(as.character(heart$target))) * 100, 1),
    "%\n")

# ── EDA Visualizations ──────────────────────

numeric_cols <- names(heart)[sapply(heart, is.numeric)]

heart_long <- melt(heart, id.vars = "target",
                   measure.vars = numeric_cols)

ggplot(heart_long,
       aes(x = value, fill = target)) +
  geom_histogram(bins = 25, alpha = 0.6,
                 position = "identity") +
  facet_wrap(~ variable, scales = "free") +
  scale_fill_manual(
    values = c("0" = "#3498DB", "1" = "#E74C3C"),
    labels = c("No Disease", "Disease")
  ) +
  labs(title = "Feature Distributions by Heart Disease Status",
       x = "Value", y = "Count", fill = "Diagnosis") +
  theme_minimal()

cor_matrix <- cor(heart[, numeric_cols],
                  use = "complete.obs")

corrplot(cor_matrix, method = "color",
         type = "upper", order = "hclust",
         tl.cex = 0.8,
         addCoef.col = "black",
         number.cex = 0.6,
         title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))

# Box plots
features_to_plot <- c("age", "trestbps",
                      "chol", "thalach")
par(mfrow = c(2, 2))
for (feat in features_to_plot) {
  boxplot(heart[[feat]] ~ heart$target,
          col = c("#3498DB", "#E74C3C"),
          main = paste(feat, "by Diagnosis"),
          xlab = "0 = No Disease, 1 = Disease",
          ylab = feat)
}
par(mfrow = c(1, 1))

# Chest pain type
ggplot(heart, aes(x = as.factor(cp),
                  fill = target)) +
  geom_bar(position = "fill") +
  scale_fill_manual(
    values = c("0" = "#3498DB", "1" = "#E74C3C"),
    labels = c("No Disease", "Disease")
  ) +
  labs(title = "Heart Disease Rate by Chest Pain Type",
       x = "Chest Pain Type (0-3)",
       y = "Proportion",
       fill = "Diagnosis") +
  theme_minimal()

# Scatter plot
ggplot(heart, aes(x = age, y = thalach,
                  color = target)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(
    values = c("0" = "#3498DB", "1" = "#E74C3C"),
    labels = c("No Disease", "Disease")
  ) +
  labs(title = "Age vs Max Heart Rate by Diagnosis",
       x = "Age", y = "Max Heart Rate") +
  theme_minimal()


# ═══════════════════════════════════════════════
# SESSION 2: Data Scrubbing & Feature Engineering
# ═══════════════════════════════════════════════

# ── Missing Values ───────────────────────────

na_counts <- colSums(is.na(heart))
na_counts[na_counts > 0]

cat("Total missing values:", sum(is.na(heart)), "\n")
cat("Complete cases:", sum(complete.cases(heart)),
    "out of", nrow(heart), "\n")

get_mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

heart$ca[is.na(heart$ca)] <-
  median(heart$ca, na.rm = TRUE)
heart$thal[is.na(heart$thal)] <-
  get_mode(heart$thal)

cat("Remaining NAs:", sum(is.na(heart)), "\n")

# ── Feature Engineering ──────────────────────

cat_cols <- c("sex", "cp", "fbs", "restecg",
              "exang", "slope", "ca", "thal")

for (col in cat_cols) {
  heart[[col]] <- as.factor(heart[[col]])
}

heart$age_group <- cut(
  heart$age,
  breaks = c(0, 40, 55, 70, 100),
  labels = c("Young", "Middle",
             "Senior", "Elderly")
)

heart$high_chol <- ifelse(heart$chol > 240, 1, 0)
heart$high_chol <- as.factor(heart$high_chol)

heart$hr_reserve <- (220 - heart$age) - heart$thalach

cat("New columns added. Total columns:",
    ncol(heart), "\n")

# ── Modeling Dataset ─────────────────────────

heart_model <- heart[, !(names(heart) %in%
                           c("age_group"))]

dummy_matrix <- model.matrix(
  ~ . - target - 1, data = heart_model
)
heart_numeric <- as.data.frame(dummy_matrix)
heart_numeric$target <- heart_model$target

cat("Factor dataset:", ncol(heart_model),
    "columns\n")
cat("Numeric dataset:", ncol(heart_numeric),
    "columns\n")

# ── Train/Test Split ─────────────────────────

set.seed(2026)

train_idx <- sample(1:nrow(heart_model),
                    size = 0.7 * nrow(heart_model))

train_data <- heart_model[train_idx, ]
test_data  <- heart_model[-train_idx, ]

train_num <- heart_numeric[train_idx, ]
test_num  <- heart_numeric[-train_idx, ]

cat("Training set:", nrow(train_data), "rows\n")
cat("Test set:",     nrow(test_data),  "rows\n")


# ═══════════════════════════════════════════════
# SESSION 3: Trees & Ensembles
# ═══════════════════════════════════════════════

# ── Evaluation Function ──────────────────────

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

# ── Decision Tree ────────────────────────────

tree_model <- rpart(
  target ~ .,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.01,
    minsplit = 10,
    maxdepth = 5
  )
)

rpart.plot(tree_model,
           main = "Heart Disease Decision Tree",
           extra = 104,
           roundint = FALSE)

acc_tree <- evaluate_model(tree_model,
                           test_data, "Decision Tree")

# ── Random Forest ────────────────────────────

set.seed(2026)
rf_model <- randomForest(
  target ~ .,
  data = train_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

print(rf_model)
varImpPlot(rf_model,
           main = "Random Forest Variable Importance")

acc_rf <- evaluate_model(rf_model,
                         test_data, "Random Forest")

# ── Gradient Boosting ────────────────────────

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

best_trees <- gbm.perf(gbm_model, method = "cv")
cat("Optimal trees:", best_trees, "\n")

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
cat("Accuracy:", round(acc_gbm * 100, 2), "%\n")


# ═══════════════════════════════════════════════
# SESSION 4: Regression & Model Comparison
# ═══════════════════════════════════════════════

# ── Logistic Regression ──────────────────────

logit_model <- glm(
  target ~ .,
  data = train_num,
  family = binomial(link = "logit")
)

summary(logit_model)

logit_step <- step(logit_model,
                   direction = "both",
                   trace = 0)

cat("\nFull model AIC:",
    AIC(logit_model), "\n")
cat("Stepwise model AIC:",
    AIC(logit_step), "\n")
cat("\nSelected variables:\n")
cat(names(coef(logit_step)), sep = "\n")

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
cat("Accuracy:", round(acc_logit * 100, 2), "%\n")

# ── ROC Curves ───────────────────────────────

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

tree_probs <- predict(tree_model,
                      test_data, type = "prob")[, 2]
rf_probs <- predict(rf_model,
                    test_data, type = "prob")[, 2]

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

# ── Model Comparison ─────────────────────────

results <- data.frame(
  Model = c("Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression"),
  Accuracy = round(c(acc_tree, acc_rf,
                     acc_gbm, acc_logit) * 100, 2)
)
results <- results[order(-results$Accuracy), ]

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


# ═══════════════════════════════════════════════
# SESSION 5: PCA, tSNE & Clustering
# ═══════════════════════════════════════════════

# ── PCA ──────────────────────────────────────

features_num <- heart_numeric[,
  !(names(heart_numeric) %in% c("target"))]

pca_result <- prcomp(features_num,
                     center = TRUE,
                     scale. = TRUE)

variance_pct <- (pca_result$sdev^2 /
  sum(pca_result$sdev^2)) * 100

plot(variance_pct,
     type = "b", pch = 19,
     xlab = "Principal Component",
     ylab = "% Variance Explained",
     main = "PCA Scree Plot",
     col = "#2E86AB")
abline(h = 5, lty = 2, col = "red")

pca_df <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Disease = heart_numeric$target
)

ggplot(pca_df, aes(x = PC1, y = PC2,
                   color = Disease)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(
    values = c("0" = "#3498DB",
               "1" = "#E74C3C"),
    labels = c("No Disease", "Disease")
  ) +
  labs(title = "PCA: First Two Components",
       x = paste0("PC1 (",
                   round(variance_pct[1], 1), "%)"),
       y = paste0("PC2 (",
                   round(variance_pct[2], 1), "%)")) +
  theme_minimal()

# ── tSNE ─────────────────────────────────────

set.seed(2026)
features_unique <- unique(features_num)
target_unique <- heart_numeric$target[
  !duplicated(features_num)]

tsne_result <- Rtsne(
  as.matrix(features_unique),
  dims = 2,
  perplexity = 30,
  verbose = FALSE,
  max_iter = 1000
)

tsne_df <- data.frame(
  tSNE1 = tsne_result$Y[, 1],
  tSNE2 = tsne_result$Y[, 2],
  Disease = target_unique
)

ggplot(tsne_df, aes(x = tSNE1, y = tSNE2,
                    color = Disease)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(
    values = c("0" = "#3498DB",
               "1" = "#E74C3C"),
    labels = c("No Disease", "Disease")
  ) +
  labs(title = "tSNE: Nonlinear 2D Projection",
       x = "tSNE Dimension 1",
       y = "tSNE Dimension 2") +
  theme_minimal()

# ── K-Means Clustering ──────────────────────

features_scaled <- scale(features_num)

fviz_nbclust(features_scaled,
             kmeans,
             method = "wss",
             k.max = 10) +
  labs(title = "Elbow Method for Optimal k") +
  theme_minimal()

set.seed(2026)
km_result <- kmeans(features_scaled,
                    centers = 2,
                    nstart = 25)

fviz_cluster(km_result,
             data = features_scaled,
             geom = "point",
             ellipse = TRUE,
             palette = c("#3498DB", "#E74C3C"),
             main = "K-Means Clustering (k=2)")

cluster_vs_actual <- table(
  Cluster = km_result$cluster,
  Actual = heart_numeric$target
)
print(cluster_vs_actual)

agree_1 <- sum(diag(cluster_vs_actual)) /
  sum(cluster_vs_actual)
agree_2 <- sum(diag(cluster_vs_actual[2:1, ])) /
  sum(cluster_vs_actual)
cat("Best cluster-to-label agreement:",
    round(max(agree_1, agree_2) * 100, 1),
    "%\n")

# Cluster profiling
cluster_profiles <- aggregate(
  features_num,
  by = list(Cluster = km_result$cluster),
  FUN = mean
)
print(t(round(cluster_profiles, 2)))

cat("\n\nCapstone Day complete!\n")
