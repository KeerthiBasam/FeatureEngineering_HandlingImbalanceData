# FeatureEngineering_HandlingImbalanceData

# Handling Class Imbalance with Upsampling & Downsampling

## Overview

This project demonstrates how to handle **class imbalance** in a binary classification dataset using **downsampling** and **upsampling** techniques in Python. A synthetic dataset is generated to simulate a real-world imbalanced scenario.

Class imbalance is a common problem in machine learning tasks such as fraud detection, churn prediction, and medical diagnosis, where one class significantly outnumbers the other.

---

## Problem Statement

Imbalanced datasets can cause machine learning models to become biased toward the majority class, resulting in poor predictive performance on the minority class.

Example imbalance:

* Majority class (0): 90%
* Minority class (1): 10%

---

## Dataset Creation

A synthetic dataset is created using NumPy and Pandas with the following characteristics:

* Total samples: 1000
* Features: `feature_1`, `feature_2`
* Target variable: `target` (binary)

Class distributions:

* Class `0`: Generated from a normal distribution with mean 0
* Class `1`: Generated from a normal distribution with mean 2

A fixed random seed is used to ensure reproducibility.

---

## Initial Class Distribution

Before applying any resampling technique, the dataset shows a strong imbalance between the two target classes.

---

## Downsampling

### Description

Downsampling reduces the number of samples in the majority class to match the minority class.

### When to Use

* Dataset is large
* Faster training is required

### Trade-offs

* Possible loss of important information from the majority class

---

## Upsampling

### Description

Upsampling increases the number of samples in the minority class by **randomly resampling with replacement** until it matches the majority class size.

In this project, upsampling is performed using `sklearn.utils.resample`, which is a simple and widely used approach for balancing classes.

### Implementation

The dataset is first split into majority and minority classes. The minority class is then resampled with replacement to match the number of majority class samples.

```
from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df['target'] == 0]
df_minority = df[df['target'] == 1]

# Upsample minority class
minority_upsampled = resample(
    df_minority,
    replace=True,          # Sample with replacement
    n_samples=len(df_majority),
    random_state=42        # For reproducibility
)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, minority_upsampled])
```

After upsampling, the dataset becomes **balanced**, with both classes having an equal number of samples.

### When to Use

* Dataset is small or moderately sized
* Minority class is critical to model performance
* You want to avoid losing majority-class information

### Trade-offs

* Increased risk of overfitting due to duplicated samples
* No new synthetic data is generated (unlike SMOTE or ADASYN)

---

## Best Practices

* Always split the dataset into **train and test sets before resampling**
* Apply resampling techniques **only on the training data**
* Use appropriate evaluation metrics:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Confusion Matrix

---

## Requirements

Install the required Python libraries:

```
pip install pandas numpy scikit-learn
```

(Optional for advanced resampling):

```
pip install imbalanced-learn
```

---


## Key Takeaway

Proper handling of class imbalance is critical for building fair and effective machine learning models. The choice between upsampling and downsampling depends on dataset size, performance goals, and risk of overfitting.
