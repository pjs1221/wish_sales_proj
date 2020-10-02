# Wish Sales Projection: Project Overview

* 

## Code and Resources Used
<strong>Python Version</strong>: 3.7.6

<strong>Packages</strong>: pandas, numpy, scikit-learn, matplotlib, seaborn

<strong>Kaggle Dataset</strong>: https://www.kaggle.com/jmmvutu/summer-products-and-sales-in-ecommerce-wish

## Data Cleaning

After acquiring the data, I needed to clean it so that it was usable for the model. I made the following changes:

* Add column with number of other listings the merchant has in the data
* Cleaned

## Exploratory Data Analysis



## Model Building

A train-test split was performed on the dataset with a test size of 20%. Furthermore, k-fold cross validation was utilized as a means of estimating the in-sample accuracy with k = 10. 

At this stage, performance was evaluated by a simple accuracy metric.

Four machine learning algorithms were considered for this data including:

* K Nearest Neighbor - Utilized as the dataset was not extremely large and thus computationally expensive
* Decision Trees - Used due to the class nature of many independent and dependent variables
* Random Forest - Ensemble method for decision trees
* XGBoost - Ensemble method more optimized for performance

Gradient boosted classifier and XGboost were found to be the best models. I used GridSearchCV to tune the hyperparameters of the model.

## Model Performance

### Initial Model Performance

I tested various models 

* K Nearest Neighbor: Train Accuracy = 46%
* SVM with Linear Kernel: Train Accuracy = 40.86%
* Decision Trees: Train Accuracy = 48.25%
* Random Forest: Train Accuracy = 52.70%
* XGBoost: Train Accuracy = 46.98%

The Random Forest clearly performed the best of the all the models tested.


