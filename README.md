# Game Review Helpfulness Prediction

This project is a replication of the study by Müller et al. (2016) aimed at predicting the helpfulness of online customer reviews using a dataset from the video game industry. The Jupyter Notebook provided includes all steps necessary to understand, preprocess, and model the data to predict review helpfulness.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
- [Results](#results)
- [References](#references)
- [Author](#author)

## Overview

The goal of this project is to predict the helpfulness of online customer reviews in the video game industry by applying various machine learning techniques. This project closely follows the methodology outlined by Müller et al. (2016).

## Dataset

The dataset used in this project is sourced from the Amazon Customer Review Dataset, specifically the `reviews_Video_Games_5.json.gz` subset. This dataset contains 231,780 reviews, each having a minimum of five reviews per user or product.

### Dataset Features

- `reviewerID`: ID of the reviewer
- `asin`: ID of the product
- `reviewerName`: Name of the reviewer
- `helpful`: Helpfulness rating of the review
- `reviewText`: Text of the review
- `overall`: Rating of the product 
- `summary`: Summary of the review
- `unixReviewTime`: Time of the review (unix time)
- `reviewTime`: Time of the review (raw)

You can download the dataset from [here](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz).

## Project Structure

- **Data Import and Exploration**: Initial data loading and basic exploratory analysis.
- **Data Preprocessing**: Cleaning and preparing the data for modeling.
- **Text Preprocessing**: Applying text processing techniques to prepare textual data.
- **LDA Topics Generation and Feature Engineering**: Generating topics using Latent Dirichlet Allocation (LDA) and creating features from the topics.
- **Modeling with RandomForest**: Training, evaluating, and predicting review helpfulness using a RandomForest classifier.
- **Metrics Plotting**: Visualizing the performance metrics of the model.

## Installation

To run this project locally, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- gensim
- json

You can install these dependencies using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk gensim json
```

## Usage

To use this project, download the dataset as described in the [Dataset](#dataset) section and place it in your working directory. Then, open the Jupyter Notebook (`predictHelpfulnessReview.ipynb`) and run the cells sequentially to replicate the study and evaluate the model.

## Methods

### Data Import and Exploration

- Loading the dataset and performing initial exploratory data analysis (EDA) to understand its structure and content.

### Data Preprocessing

- Cleaning the data by handling missing values and filtering out irrelevant entries.

### Text Preprocessing

- Processing the review text to convert it into a format suitable for modeling, including tokenization, stop word removal, and stemming.

### LDA Topics Generation and Feature Engineering

- Applying LDA for topic modeling and creating features based on document-topic probabilities.

### Modeling with RandomForest

- Training a RandomForest classifier on the engineered features and evaluating its performance on the test set.

### Metrics Plotting

- Visualizing performance metrics such as accuracy, precision, recall, and F1-score to assess the model's effectiveness.

## Results

The model's performance is evaluated based on several metrics, including accuracy, precision, recall, and F1-score. These metrics are plotted to provide a clear visual representation of how well the model predicts the helpfulness of reviews.

## References

- Müller, J., Gurevych, I., & Zubiaga, A. (2016). "Predicting the Helpfulness of Online Reviews." In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 314–319, Beijing, China.

## Author

This project was created by Vaishob Anand. If you have any questions or feedback, feel free to reach out.
