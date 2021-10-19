import opfython.math.general as g
import opfython.stream.splitter as sp
import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.models.supervised import SupervisedOPF
import sklearn.datasets as sd
from ssa import SSA
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.optimizers.evolutionary.ga import GA

import statistics as st
import opytimizer.math.random as r
import random as r
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.boolean import BPSO
from opytimizer.spaces import SearchSpace
import featureselection as fs
import numpy as np


# Splits data into training and testing sets
X_train, X_val, Y_train, Y_val = s.split(
    X, Y, percentage=0.5, random_state=1)

# Loads digits dataset
#digits = load_digits()

#breast_cancer = load_breast_cancer()
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

acuracias = []
best_selected_features = []

def supervised_opf_feature_selection(opytimizer):
    
    global best_selected_features

    # Transforming the continum solution in boolean solution (feature vector)
    # Gathers features
    features = np.asarray(fs.s2_transfer_function(opytimizer)).astype(bool)

    # Remaking training and validation subgraphs with selected features
    X_train_selected = X_train[:, features]
    X_val_selected = X_val[:, features]

    # Creates a SupervisedOPF instance
    opf = SupervisedOPF(distance='log_squared_euclidean',
                        pre_computed_distance=None)


    # Fits training data into the classifier
    opf.fit(X_train_selected, Y_train)

    # Predicts new data
    preds = opf.predict(X_val_selected)

    # Calculates accuracy
    acc = g.opf_accuracy(Y_val, preds)

    if (1-acc) <= space.best_agent._fit:
        best_selected_features = features

    return 1 - acc

# Number of agents and decision variables
n_agents = 30
n_variables = 64

max_it = 100

# Lower and upper bounds (has to be the same size as n_variables
lower_bound = ()
upper_bound = ()

#We should use the transfer function's upper and lower bounds
#Sigmoid's upper and lower bound
for i in range(n_variables):
       lower_bound += (0,)
       upper_bound += (1,)

np.random.seed(125)

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)

function = Function(supervised_opf_feature_selection)

optimizer = GA()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=max_it)

acuracias.append(space.best_agent._fit)
'''
#Parameters
param1 = f"Algoritmo SSA\nParâmetros do Teste \n -Agentes: {n_agents}\n -Variáveis de decisão: {n_variables}\n -Iterações: {max_it}\n"
param2 = f"Database: Breast Cancer"

param = param1 + param2

print(param)

print(acuracias)
'''
print(f"O melhor conjunto de características é: {best_selected_features}")

opf = SupervisedOPF(distance='log_squared_euclidean',
                        pre_computed_distance=None)

X_train_selected = X_train[:, best_selected_features]
X_test_selected = X_test[:, best_selected_features]

# Fits training data into the classifier
opf.fit(X_train_selected, Y_train)

# Predicts new data 
preds = opf.predict(X_test_selected)

# Calculates accuracy
acc = g.opf_accuracy(Y_test, preds)

print(acc)

'''
average_acc = st.mean(acuracias)

print(f"\nMédia = {average_acc}")

std_acc = st.stdev(acuracias)

print(f"\nDesvio Padrão = {std_acc}")
'''
#print(parameters)