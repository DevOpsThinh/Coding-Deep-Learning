# Author/ Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Create a Logistic Regression model that can be used to convert into
# the Core ML Format via CoreML Tool .
# The model will be trained & converted on the popular UCI ML Pima Indians Diabetes dataset.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from diabetes_logistic_regession_model \
    import df_diabetes_features, df_diabetes_target, df_diabetes_features_train, \
    df_diabetes_target_train, df_diabetes_features_test, df_diabetes_target_test

# Train a logistic regression model on the diabetes dataset.
lr_model = LogisticRegression(penalty='l2', fit_intercept=True, solver='liblinear', multi_class='ovr')
trained_model = lr_model.fit(df_diabetes_features_train, df_diabetes_target_train.values.ravel())

# Get predictions
lr_predictions = lr_model.predict(df_diabetes_features_test)
print(lr_predictions)

# Access class-wise probabilities
lr_probabilities = lr_model.predict_proba(df_diabetes_features_test)
print(lr_probabilities)

# Implement custom thresholding logic
dfProbabilities = pd.DataFrame(lr_probabilities[:, 0])
predictions_with_custom_threshold = dfProbabilities.applymap(lambda x: 0 if x > 0.8 else 1)
print(predictions_with_custom_threshold.values.ravel())

cm = confusion_matrix(df_diabetes_target_test, lr_predictions)
accuracy = (cm[0][0] + cm[1][1]) / df_diabetes_target_test.shape[0]
print('accuracy: ' + str(accuracy))  # 79,69%

# Perform 10-fold cross-validation & compute the average accuracy (ac) across the 10 folds.
kf = StratifiedKFold(n_splits=10, shuffle=True)
f_number = 1
f_accuracy = []

for train_indices, test_indices in kf.split(df_diabetes_features, y = df_diabetes_target['Outcome']):
    df_diabetes_features_train = df_diabetes_features.iloc[train_indices]
    df_diabetes_target_train = df_diabetes_target.iloc[train_indices]

    df_diabetes_features_test = df_diabetes_features.iloc[test_indices]
    df_diabetes_target_test = df_diabetes_target.iloc[test_indices]

    # Train a logistic regression model
    model = LogisticRegression(penalty='l2', fit_intercept=True, solver="liblinear", multi_class='ovr')
    trained_model_kfcv = model.fit(df_diabetes_features_train, df_diabetes_target_train.values.ravel())

    # Get predictions
    predictions = model.predict(df_diabetes_features_test)

    # Compute fold accuracy, append the value into the f_accuracy array
    cm = confusion_matrix(df_diabetes_target_test, predictions)
    accuracy = (cm[0][0] + cm[1][1]) / df_diabetes_target_test.shape[0]
    f_accuracy.append(accuracy)

    print("Fold number:", f_number)
    print("Fold accuracy: " + str(accuracy))

    f_number += 1

# Compute ac of the model across the folds.
print("Average 10-fold accuracy: " + str(np.mean(f_accuracy)))  # 77, 35%




