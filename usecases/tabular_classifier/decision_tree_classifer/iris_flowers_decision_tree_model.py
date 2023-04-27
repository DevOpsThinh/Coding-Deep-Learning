import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from IPython.display import Image
from six import StringIO

# Load the iris flowers dataset
iris_dataset = datasets.load_iris()

df_iris = pd.DataFrame(data=np.c_[iris_dataset['data'], iris_dataset['target']],
                       columns=iris_dataset['feature_names'] + ['target'])
# Verify the shape of the dataset
print('df_iris has ' + str(df_iris.shape[0]) + ' rows and ' + str(df_iris.shape[1]) + ' columns')
# Inspect the first few rows (5) of the dataframe
pd.set_option('display.max_columns', None)
df_iris.head()
df_iris.info()
# Histogram of target attribute
# df_iris['target'].hist(figsize=(7, 4))

# Change the data type of the target column to string
df_iris[['target']] = df_iris[['target']].astype(np.uint8)
df_iris['target'] = df_iris['target'].apply(str)

df_iris.isnull().sum()
# Split the df_iris dataframe into two, one with the features and the other with the target
df_iris_target = df_iris.loc[:, ['target']]
df_iris_features = df_iris.drop(['target'], axis=1)
# Create training &  test sets
iris_split = train_test_split(df_iris_features, df_iris_target, test_size=0.25, random_state=17, stratify=df_iris_target['target'])
df_iris_features_train = iris_split[0]
df_iris_features_test = iris_split[1]
df_iris_target_train = iris_split[2]
df_iris_target_test = iris_split[3]
# Create the Decision Tree Classification Model (DTM) with Scikit-learn
# https:/scikit-learn.org/stable/modules;generated/sklearn.tree.DecisionTreeClassifier.html
# Train a DTM
model = DecisionTreeClassifier(random_state=17)
model.fit(df_iris_features_train, df_iris_target_train.values.ravel())
print(model.feature_importances_)
# Get predictions from model, and compute accuracy
predictions = model.predict(df_iris_features_test)
print(predictions)
accuracy = accuracy_score(df_iris_target_test, predictions)
print(accuracy)
# Visualize the Decision Tree
dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names=df_iris_features.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("iris_flow_dtm.png")
Image(graph.create_png())



