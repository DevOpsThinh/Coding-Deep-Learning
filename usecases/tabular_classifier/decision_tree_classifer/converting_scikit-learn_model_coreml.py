# Author/ Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Create a Decision Tree classification model that can be used to convert into the Core ML Format
# via CoreML Tool .
# The model will be trained & converted on the popular UCI ML Iris flowers dataset.

import coremltools
from iris_flowers_decision_tree_model import model_trained

# Export the model to Core ML format
coreml_model = coremltools.converters.sklearn.convert(model_trained, ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], "target")
coreml_model.author = 'Nguyen Truong Thinh'
coreml_model.short_description = 'A Decision Tree model trained on the Iris flowers dataset.'
# Features descriptions
coreml_model.input_description['sepal length (cm)'] = 'Sepal length in cm.'
coreml_model.input_description['sepal width (cm)'] = 'Sepal width in cm.'
coreml_model.input_description['petal length (cm)'] = 'Petal length in cm.'
coreml_model.input_description['petal width (cm)'] = 'Petal width in cm.'
# Description of target variable
coreml_model.output_description['target'] = 'A categorical value value, 0 = Iris-Setosa, 1 = Iris-Versicolour, 3 = Iris-Virginica'
coreml_model.save("iris_flowers_dtm.mlpackage")



