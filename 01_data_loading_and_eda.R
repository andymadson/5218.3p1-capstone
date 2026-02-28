# ============================================
# Session 1: Data Loading, EDA & Visualization
# Time: 8:00 - 9:15 AM
# ============================================
# Topics: Week 1 (EDA, R basics), Week 4 (ggplot2)
# Dataset: Cleveland Heart Disease (UCI)
# ============================================

library(ggplot2)
library(corrplot)
library(reshape2)

# ── Part A: Setup and Data Loading ───────────

# Step 1: Load the Cleveland Heart Disease dataset
url <- paste0(
  "https://archive.ics.uci.edu/ml/",
  "machine-learning-databases/heart-disease/",
  "processed.cleveland.data"
)

# Define column names based on UCI documentation
col_names <- c(
  "age", "sex", "cp", "trestbps", "chol",
  "fbs", "restecg", "thalach", "exang",
  "oldpeak", "slope", "ca", "thal", "target"
)

heart <- read.csv(url, header = FALSE,
                  col.names = col_names,
                  na.strings = "?")

# Quick check
cat("Rows:", nrow(heart), "  Columns:", ncol(heart), "\n")

# Step 2: Inspect the structure
str(heart)
head(heart)
summary(heart)

# Step 3: Convert the target to a binary factor
# 0 = No Disease, 1+ = Disease
heart$target <- ifelse(heart$target > 0, 1, 0)
heart$target <- as.factor(heart$target)

# Check class balance
table(heart$target)
cat("Disease prevalence:",
    round(mean(as.numeric(as.character(heart$target))) * 100, 1),
    "%\n")


# ── Part B: Exploratory Data Analysis ────────

# Step 4: Faceted histograms by target
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

# Step 5: Correlation heatmap
cor_matrix <- cor(heart[, numeric_cols],
                  use = "complete.obs")

corrplot(cor_matrix, method = "color",
         type = "upper", order = "hclust",
         tl.cex = 0.8,
         addCoef.col = "black",
         number.cex = 0.6,
         title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))


# ── Part C: Team EDA Challenge ───────────────

# Challenge 1: Box plots for key numeric features
features_to_plot <- c("age", "trestbps",
                      "chol", "thalach")

par(mfrow = c(2, 2))  # 2x2 grid
for (feat in features_to_plot) {
  boxplot(heart[[feat]] ~ heart$target,
          col = c("#3498DB", "#E74C3C"),
          main = paste(feat, "by Diagnosis"),
          xlab = "0 = No Disease, 1 = Disease",
          ylab = feat)
}
par(mfrow = c(1, 1))  # Reset layout

# Challenge 2: Chest pain type vs target (proportion)
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

# Challenge 3: Scatter plot — Age vs Max Heart Rate
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
