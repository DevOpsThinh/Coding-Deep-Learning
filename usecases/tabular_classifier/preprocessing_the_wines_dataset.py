# Author/ Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Create a classification model that can be used to classify the quality of wines based on their chemical
# composition. The model will be trained on the popular UCI ML wine dataset and export the dataset to CSV files.

import numpy as np
import matplotlib
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

from fundamentals.custom_functions import make_the_graph

wine_dataset = datasets.load_wine()

df_wine = pd.DataFrame(data=np.c_[wine_dataset['data'],
    wine_dataset['target']], columns=wine_dataset['feature_names'] + ['target'])

# examine the shape of the dataset
print('df_wine has ' + str(df_wine.shape[0]) + ' rows and ' + str(df_wine.shape[1]) + ' columns')
df_wine.info()
# Change the datatype of the target column to unsigned int8
df_wine[['target']] = df_wine[['target']].astype(np.uint8)
# How many missing values?
df_wine.isnull().sum()
# Histogram of columns
df_wine['target'].hist(figsize=(7, 4))

make_the_graph(df_wine, "The distribution of the data of the UCL ML Wine Dataset",
               "Min values", "Max values")
make_the_graph(df_wine['target'], "The distribution of the data in the target attribute",
               "Max values", "Min values")

# Split the df_wine dataframe into 2, 1 with the features and the other with the target
df_wine_target = df_wine.loc[:, ['target']]
df_wine_features = df_wine.drop(['target'], axis=1)

# Standardize the columns of df_wine_features
df_wine_features = (df_wine_features - df_wine_features.min()) / (df_wine_features.max() - df_wine_features.min())

df_wine_features.head()

# Create a training & test sets
wine_dataset_split = train_test_split(df_wine_features, df_wine_target,
                                      test_size=0.25, random_state=17, stratify=df_wine_target['target'])

df_wine_features_train = wine_dataset_split[0]
df_wine_features_test = wine_dataset_split[1]
df_wine_target_train = wine_dataset_split[2]
df_wine_target_test = wine_dataset_split[3]

# Combine  train - test datasets into a single dataframe & export it to a CSV file.
df_wine_train = pd.concat([df_wine_features_train, df_wine_target_train], axis=1)
df_wine_train.to_csv('./datasets_export/wine_train.csv', header=True, index=None)
df_wine_test = pd.concat([df_wine_features_test, df_wine_target_test], axis=1)
df_wine_test.to_csv('./datasets_export/wine_test.csv', header=True, index=None)