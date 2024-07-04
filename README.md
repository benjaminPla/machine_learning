# Machine Learning

## Supervised Learning

You have the output column and you train the model with it.

- **Classification**: Predict a class label or category (e.g., true/false, spam/ham, cat/dog).

  - Logistic Regression (binary or multi-class)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Gradient Boosting Machines (GBM)
  - Neural Networks

- **Regression**: Predict a continuous value (e.g., price, temperature, age).

  - Linear Regression
  - Decision Tree Regression
  - Random Forest Regression
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors Regression
  - Ridge Regression
  - Lasso Regression
  - Polynomial Regression
  - Neural Networks (e.g., Multilayer Perceptron)

## Unsupervised Learning

You don't have the output column and you train the model without it.

- **Dimensionality Reduction**: Reduce the number of features while preserving important information.

  - Principal Component Analysis (PCA)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Singular Value Decomposition (SVD)
  - Linear Discriminant Analysis (LDA)
  - Independent Component Analysis (ICA)

- **Density Estimation**: Estimate the probability distribution of data.

  - Gaussian Mixture Models (GMM)
  - Kernel Density Estimation (KDE)

- **Market Basket Analysis**: Identify associations between items.

  - Apriori Algorithm
  - Eclat Algorithm

- **Clustering**: Group similar data points together.

  - K-Means
  - Hierarchical Clustering
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - Mean Shift

# Metrics

## Summary:

- **Accuracy** gives an overall measure of correct predictions.
- **Confusion Matrix** breaks down the types of correct and incorrect predictions.
- **Recall** (or sensitivity) focuses on correctly identifying positive cases.
- **Specificity** (or true negative rate) focuses on correctly identifying negative cases.
- **F1 Score** balances precision and recall into a single metric, useful when there's an uneven class distribution.

### Accuracy Score:

**Definition**: Accuracy measures the proportion of correctly predicted instances (both true positives and true negatives) out of the total number of instances.
**Interpretation**: It provides an overall assessment of how well the model predicts the correct class across all classes.
Formula: _Accuracy=Number of Correct PredictionsTotal Number of PredictionsAccuracy=Total Number of PredictionsNumber of Correct Predictions_

### Confusion Matrix:

**Definition**: A table that summarizes the performance of a classification model.
**Interpretation**: Shows the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

Annotations:

- TP: Correctly predicted positives.
- TN: Correctly predicted negatives.
- FP: Incorrectly predicted positives (Type I error).
- FN: Incorrectly predicted negatives (Type II error).

>[[3452  294]
> [ 900  304]]

|               | Predicted Stays | Predicted Leaves |
| ------------- | --------------- | ---------------- |
| Actual Stays  | 3452 (TP)       | 294 (FP)         |
| Actual Leaves | 900 (FN)        | 304 (TN)         |

### Recall (Sensitivity, True Positive Rate):

**Definition**: Recall measures the proportion of actual positives that are correctly identified by the model.
**Interpretation**: Indicates how well the model identifies positive instances.
Formula: _Recall=TPTP+FNRecall=TP+FNTP_

### Specificity (True Negative Rate):

**Definition**: Specificity measures the proportion of actual negatives that are correctly identified by the model.
**Interpretation**: Indicates how well the model identifies negative instances.
Formula: _Specificity=TNTN+FPSpecificity=TN+FPTN_

### F1 Score:

**Definition**: F1 score is the harmonic mean of precision and recall. It provides a balance between precision and recall.
**Interpretation**: A higher F1 score indicates better overall performance of the model.
Formula: _F1 Score=2⋅Precision⋅RecallPrecision+RecallF1 Score=2⋅Precision+RecallPrecision⋅Recall_
