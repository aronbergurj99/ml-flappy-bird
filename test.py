from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# X, y = make_regression(n_samples=200, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
# regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(100,10,2))
# regr.partial_fit([X_test[0]], [y_train[0]])

# print(X_test[0])

# print(regr.predict(X_test[0].reshape(-1,1)))
import numpy

import random
"""
'next_pipe_top_y', 'player_y', 'player_vel', 'next_pipe_dist_to_player', 'q_flap', 'q_noop'
"""

def make_dummydata():
    action = random.uniform(0,1)
    return [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)], [action, 1-action]



regr = MLPRegressor(random_state=1, hidden_layer_sizes=(100,10), activation="logistic", alpha=0.001)
train_x, train_y = make_dummydata()
regr.partial_fit([train_x], [train_y])

test_x, _ = make_dummydata()
test_x = numpy.array(test_x)
print(test_x)
print(test_x.reshape(1,-1))
print(regr.predict(test_x.reshape(1,-1)))