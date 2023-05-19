# Author/ Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Use case: Create a Keras Inception v4 model that can be used to convert into the Core ML Format
#                 via CoreML Tool .
#                  The model will be trained & converted on the Kaggle Dogs vs Cats dataset via Keras functional api.

import coremltools as ct

from ml.deep_learning.keras_functional_api.inception_v4_network import model

# https://drive.google.com/file/d/1NqMs2js-uOyOLL9MB0nzX0AoHx-hy8ty/view?usp=share_link
weights_path = 'inceptionv4_dogscats_weights.h5'
model.load_weights(weights_path)

coreml_model = ct.convert(model, source="tensorflow", minimum_deployment_target=ct.target.iOS13)

coreml_model.author = 'Nguyen Truong Thinh'
coreml_model.short_description = 'An Inception v4 model trained on the Kaggle Dogs vs Cats dataset via Keras ' \
                                 'functional api.'
coreml_model.save("inception_v4_dogscats.mlmodel")

"""
Running TensorFlow Graph Passes: 100%|██████████| 6/6 [00:00<00:00,  8.02 passes/s]
Converting TF Frontend ==> MIL Ops: 100%|██████████| 1258/1258 [00:00<00:00, 1981.15 ops/s]
Running MIL frontend_tensorflow2 pipeline: 100%|██████████| 7/7 [00:00<00:00, 155.55 passes/s]
Running MIL default pipeline: 100%|██████████| 56/56 [00:04<00:00, 12.48 passes/s]
Running MIL backend_neuralnetwork pipeline: 100%|██████████| 8/8 [00:00<00:00, 256.00 passes/s]
Translating MIL ==> NeuralNetwork Ops: 100%|██████████| 1538/1538 [00:02<00:00, 647.43 ops/s]

Process finished with exit code 0
"""
