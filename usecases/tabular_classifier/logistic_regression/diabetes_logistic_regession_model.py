# Author/ Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Create a Logistic Regression model that can be used to convert into
# the Core ML Format via CoreML Tool .
# The model will be trained & converted on the popular UCI ML Pima Indians Diabetes dataset.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

# Load the contents of a file into a pandas Dataframe
INPUT_FILE = "./diabetes.csv"
df_diabetes = pd.read_csv(INPUT_FILE)

# Examine the shape of the dataset
print('df_diabetes has ' + str(df_diabetes.shape[0]) + ' rows and ' + str(df_diabetes.shape[1]) + ' columns')
# View the first 5 row of the dataFrame
pd.set_option('display.max_columns', None)
df_diabetes.head().info()

# How many missing values?
df_diabetes.isnull().sum()

# Examine the  statistical characteristics of the dataframe
df_diabetes.describe()

# Histogram of columns
df_diabetes.hist(figsize=(12, 12))
df_diabetes['BloodPressure'].hist(figsize=(12, 4))

# Split the df_diabetes dataframe into two, one with the features & the other with
# the features and the other with the target.
df_diabetes_target = df_diabetes.loc[:, ['Outcome']]
df_diabetes_features = df_diabetes.drop(['Outcome'], axis=1)

# Inspect the number of items in each dataframe
print('df_diabetes_target has ' + str(df_diabetes_target.shape[0]) +
      ' rows and ' + str(df_diabetes_target.shape[1]) + ' columns')
print('df_diabetes_features has ' + str(df_diabetes_features.shape[0]) +
      ' rows and ' + str(df_diabetes_features.shape[1]) + ' columns')

# Create a training & test sets.
diabetes_split = train_test_split(df_diabetes_features, df_diabetes_target,
                                  test_size=0.25, random_state=17, stratify=df_diabetes_target['Outcome'])
df_diabetes_features_train = diabetes_split[0]
df_diabetes_features_test = diabetes_split[1]
df_diabetes_target_train = diabetes_split[2]
df_diabetes_target_test = diabetes_split[3]

# Visualize the distribution of target values in the original dataset & the test/ train sets
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].set_title('df_diabetes_target')
df_diabetes_target['Outcome'].value_counts(dropna=False).plot.bar(grid=True, ax=axes[0])

axes[1].set_title('df_diabetes_target_train')
df_diabetes_target_train['Outcome'].value_counts(dropna=False).plot.bar(grid=True, ax=axes[1])

axes[2].set_title('df_diabetes_target_test')
df_diabetes_target_test['Outcome'].value_counts(dropna=False).plot.bar(grid=True, ax=axes[2])

# plt.show()


