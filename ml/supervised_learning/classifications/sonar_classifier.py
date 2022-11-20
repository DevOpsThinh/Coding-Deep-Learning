# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
#
# Topic: Supervised Learning: Classifications in Action.
#            Minesweeper Challenge:
#               A binary classifier that discriminate between sonar signals
#               bounced off a metal cylinder and those bounced off a roughly cylindrical rock.

import connectionist_bench_sonar_mines_rocks as data
from base_classifier import base_train

# I can reach >79% accuracy, and you?
weights = base_train(data.X_train, data.Y_train,
                     data.X_test, data.Y_test,
                     iterations=100000, lr=0.01)
