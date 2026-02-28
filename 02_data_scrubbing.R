# ============================================
# Session 2: Data Scrubbing & Feature Engineering
# Time: 9:30 - 10:45 AM
# ============================================
# Topics: Week 2 (data scrubbing, imputation,
#         one-hot encoding)
# Prerequisite: Run 01_data_loading_and_eda.R
#               first (needs the 'heart' data frame)
# ============================================

# ── Part A: Handling Missing Values ──────────

# Step 1: Identify missing data
na_counts <- colSums(is.na(heart))
na_counts[na_counts > 0]

cat("Total missing values:", sum(is.na(heart)), "\n")
cat("Complete cases:", sum(complete.cases(heart)),
    "out of", nrow(heart), "\n")

# Step 2: Impute missing values
# Mode function for categorical imputation
get_mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Impute ca with median
heart$ca[is.na(heart$ca)] <-
  median(heart$ca, na.rm = TRUE)

# Impute thal with mode
heart$thal[is.na(heart$thal)] <-
  get_mode(heart$thal)

# Verify: no more NAs
cat("Remaining NAs:", sum(is.na(heart)), "\n")


# ── Part B: Feature Engineering ──────────────

# Step 3: Convert categorical columns to factors
cat_cols <- c("sex", "cp", "fbs", "restecg",
              "exang", "slope", "ca", "thal")

for (col in cat_cols) {
  heart[[col]] <- as.factor(heart[[col]])
}

str(heart)

# Step 4: Create new engineered features

# Age group bins
heart$age_group <- cut(
  heart$age,
  breaks = c(0, 40, 55, 70, 100),
  labels = c("Young", "Middle",
             "Senior", "Elderly")
)

# High cholesterol flag (>240 mg/dL is high)
heart$high_chol <- ifelse(
  heart$chol > 240, 1, 0
)
heart$high_chol <- as.factor(heart$high_chol)

# Heart rate reserve (proxy for fitness)
# Max predicted HR = 220 - age
heart$hr_reserve <- (220 - heart$age) - heart$thalach

cat("New columns added. Total columns:",
    ncol(heart), "\n")


# ── Part C: Preparing the Modeling Dataset ───

# Step 5: One-hot encode categorical variables
heart_model <- heart[, !(names(heart) %in%
                           c("age_group"))]

# One-hot encode using model.matrix
# -1 removes the intercept column
dummy_matrix <- model.matrix(
  ~ . - target - 1, data = heart_model
)
heart_numeric <- as.data.frame(dummy_matrix)
heart_numeric$target <- heart_model$target

cat("Factor dataset:", ncol(heart_model),
    "columns\n")
cat("Numeric dataset:", ncol(heart_numeric),
    "columns\n")

# Step 6: Train/Test split (70/30)
set.seed(2026)  # Reproducibility

train_idx <- sample(1:nrow(heart_model),
                    size = 0.7 * nrow(heart_model))

train_data <- heart_model[train_idx, ]
test_data  <- heart_model[-train_idx, ]

# Same split for the numeric version
train_num <- heart_numeric[train_idx, ]
test_num  <- heart_numeric[-train_idx, ]

cat("Training set:", nrow(train_data), "rows\n")
cat("Test set:",     nrow(test_data),  "rows\n")
