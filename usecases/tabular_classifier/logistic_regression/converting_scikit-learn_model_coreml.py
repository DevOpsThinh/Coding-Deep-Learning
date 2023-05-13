# Author/ Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Create a Logistic Regression model that can be used to convert into
# the Core ML Format via CoreML Tool .
# The model will be trained & converted on the popular UCI ML Pima Indians Diabetes dataset.

import coremltools

from usecases.tabular_classifier.logistic_regression.logistic_regression_classifier import trained_model

coreml_model = coremltools.converters.sklearn.convert(trained_model, ['Pregnancies', 'Glucose',
                                                                      'BloodPressure', 'SkinThickness', 'Insulin',
                                                                      'BMI', 'DiabetesPedigreeFunction', 'Age'],
                                                      'Outcome')

coreml_model.author = 'Nguyen Truong Thinh'
coreml_model.short_description = 'A logistic regression model trained on the Kaggle.com version of th Pima Indians ' \
                                 'diabetes dataset.'
# Features description
coreml_model.input_description['Pregnancies'] = 'Number of pregnancies.'
coreml_model.input_description['Glucose'] = 'Plasma glucose concentration after 2 hours in an oral glucose tolerance ' \
                                            'test.'
coreml_model.input_description['BloodPressure'] = 'Diastolic blood pressure.'
coreml_model.input_description['SkinThickness'] = 'Thickness of the triceps skin folds.'
coreml_model.input_description['BMI'] = 'Body mass index.'
coreml_model.input_description[
    'DiabetesPedigreeFunction'] = 'A function that determines the risk of diabetes based on family history.'
coreml_model.input_description['Age'] = 'The age of the subject.'
# Description of target variable
coreml_model.output_description['Outcome'] = 'A binary value, 1 indicates the patient has type-2 diabetes.'
coreml_model.save('diabetes_indian.mlpackage')
