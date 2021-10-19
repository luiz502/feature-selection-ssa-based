import opfython.math.general as g
import opfython.stream.splitter as sp
import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.models.supervised import SupervisedOPF
import sklearn.datasets as sd
#from ssa import SSA
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.optimizers.evolutionary.ga import GA

import statistics as st
import opytimizer.math.random as r
import random as r
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
import featureselection_functions as fs
import numpy as np

def supervised_opf_feature_selection(opytimizer):
    
    global best_selected_features

    # Transforms the continum solution in boolean solution (feature array) by applying the transfer function
    features = fs.s1_transfer_function(opytimizer)

    # Remaking training and validation subgraphs with selected features
    X_train_selected = X_train[:, features]
    X_val_selected = X_val[:, features]

    # Creates a SupervisedOPF instance
    opf = SupervisedOPF(distance='log_squared_euclidean',
                        pre_computed_distance=None)

    # Fits training data into the classifier
    opf.fit(X_train_selected, Y_train)

    # Predicts new data from validate set
    preds = opf.predict(X_val_selected)

    # Calculates accuracy
    acc = g.opf_accuracy(Y_val, preds)

    # Error
    error = 1 - acc

    # If the error is lower than the best agent's error
    if error <= space.best_agent._fit:
        # Save the best selected features
        best_selected_features = features

    return error

# Loads digits dataset 
digits = sd.load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target

# Adding 1 to labels, i.e., OPF should have labels from 1+
Y += 1

# Splits data into training and test sets
X_train, X_test, Y_train, Y_test = sp.split(X, Y, percentage=0.5, random_state=1)

# Training set will be splited into training and validation sets
X_train, X_val, Y_train, Y_val = sp.split(X_train, Y_train, percentage = 0.2, random_state=1)


best_selected_features = []

# Number of agents
n_agents = 30

# Number of dataset features
n_variables = 64

# Maximum number of iterations
max_it = 10

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = ()
upper_bound = ()

# Transfer function's upper and lower bounds
for i in range(n_variables):
       lower_bound += (0,)
       upper_bound += (1,)

np.random.seed(r.randint(0, 1000))

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)

function = Function(supervised_opf_feature_selection)

optimizer = PSO()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task in order to find the best selected features
opt.start(n_iterations=max_it)

# Runs OPF in order to check the classifier's accuracy on the test set
opf = SupervisedOPF(distance='log_squared_euclidean',
                        pre_computed_distance=None)

# Remaking training and tests subgraphs with selected features
X_train_selected = X_train[:, best_selected_features]
X_test_selected = X_test[:, best_selected_features]

# Fits training data into the classifier
opf.fit(X_train_selected, Y_train)

# Predicts new data from test set 
preds = opf.predict(X_test_selected)

# Calculates accuracy
acc = g.opf_accuracy(Y_test, preds)

print(f"The accuracy on the test set is: {acc * 100}% ")
