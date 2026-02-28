# ============================================
# Session 5: PCA, tSNE & Clustering
# Time: 2:30 - 3:30 PM
# ============================================
# Topics: Week 7 (PCA, tSNE),
#         Week 8 (clustering, segmentation)
# Prerequisite: Run 01 and 02 scripts first
#               (needs heart_numeric)
# ============================================

library(Rtsne)
library(cluster)
library(factoextra)
library(ggplot2)

# ── Part A: Principal Component Analysis ─────

# Step 1: Run PCA on numeric features
features_num <- heart_numeric[,
  !(names(heart_numeric) %in% c("target"))]

pca_result <- prcomp(features_num,
                     center = TRUE,
                     scale. = TRUE)

# Scree plot
variance_pct <- (pca_result$sdev^2 /
  sum(pca_result$sdev^2)) * 100

plot(variance_pct,
     type = "b", pch = 19,
     xlab = "Principal Component",
     ylab = "% Variance Explained",
     main = "PCA Scree Plot",
     col = "#2E86AB")
abline(h = 5, lty = 2, col = "red")

# Step 2: Visualize PCA in 2D
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


# ── Part B: tSNE Visualization ───────────────

# Step 3: Run tSNE
# Remove duplicate rows (tSNE requirement)
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

# Plot tSNE
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


# ── Part C: K-Means Clustering ───────────────

# Step 4: Elbow method for optimal k
features_scaled <- scale(features_num)

fviz_nbclust(features_scaled,
             kmeans,
             method = "wss",
             k.max = 10) +
  labs(title = "Elbow Method for Optimal k") +
  theme_minimal()

# Step 5: Run K-Means (k = 2)
set.seed(2026)
km_result <- kmeans(features_scaled,
                    centers = 2,
                    nstart = 25)

# Visualize clusters on PCA axes
fviz_cluster(km_result,
             data = features_scaled,
             geom = "point",
             ellipse = TRUE,
             palette = c("#3498DB", "#E74C3C"),
             main = "K-Means Clustering (k=2)")

# Compare clusters to actual disease labels
cluster_vs_actual <- table(
  Cluster = km_result$cluster,
  Actual = heart_numeric$target
)
print(cluster_vs_actual)

# Agreement rate
agree_1 <- sum(diag(cluster_vs_actual)) /
  sum(cluster_vs_actual)
agree_2 <- sum(diag(cluster_vs_actual[2:1, ])) /
  sum(cluster_vs_actual)
cat("Best cluster-to-label agreement:",
    round(max(agree_1, agree_2) * 100, 1),
    "%\n")


# ── Team Challenge: Cluster Profiling ────────

# Profile clusters by feature means
cluster_profiles <- aggregate(
  features_num,
  by = list(Cluster = km_result$cluster),
  FUN = mean
)
# Transpose for easier reading
print(t(round(cluster_profiles, 2)))
