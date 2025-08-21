# Spam Classifier

This repository contains a project for building and evaluating machine learning models to classify SMS messages as **spam** or **ham** (non-spam). The aim is to create an effective spam detection system using various text classification techniques and machine learning algorithms.

***

## Project Overview

- The project involves preprocessing text message data, implementing different feature extraction methods such as Bag of Words, TF-IDF, and Word2Vec.
- Several machine learning models are trained and compared, including:
  - Multinomial Naive Bayes
  - SGDClassifier
  - Random Forest Classifier
  - Logistic Regression
  - AdaBoost
  - XGBoost
- Model performance is evaluated based on metrics like Accuracy, Precision, Recall, and F1-Score.
- Different feature extraction methods are analyzed using metrics such as accuracy, precision, and recall.

***

## Dataset

- The dataset used in this repository is sourced from **Kaggle**.
- It contains labeled SMS messages tagged as either spam or ham.
- The data files included:
  - `spam.csv` — A collection of SMS messages with the category label.
  - `combined_data.csv`, `spam_ham_dataset.csv`, and `email.csv` — Additional datasets with labeled message data used for analysis and model training.
  
**Note:** The dataset is publicly available on Kaggle and widely used for spam detection tasks.

***

## Key Features

- Data exploration and visualization of message categories (spam vs. ham).
- Text preprocessing and feature engineering using multiple methods.
- Training and comparison of various classical ML models.
- Performance comparison with detailed metrics.
- Code is implemented in Python using libraries like Pandas, NumPy, and Scikit-learn.

***

## How to Use

1. Clone the repository.
2. Ensure the datasets (`spam.csv` and others) are in the designated folder.
3. Run the Jupyter notebooks or scripts to preprocess data, train models, and evaluate performance.
4. Review accuracy and other performance metrics to select the best model for your needs.

***

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn (for visualization)

***

## Acknowledgements

- The SMS spam dataset used in this project was obtained from **Kaggle**. It is a publicly available dataset commonly used for benchmarking spam filtering techniques.

***

This repository serves as a comprehensive example of applying machine learning for natural language processing (NLP) classification tasks, specifically for spam detection in SMS messages.


