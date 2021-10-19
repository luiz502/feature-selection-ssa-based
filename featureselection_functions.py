import opytimizer.math.random as r
import math as m
import numpy as np

def s1_transfer_function(otimization_solution):
    """Transfer Function S1

    Args:
        otimization_solution (np.array): The best agent's position. 
    """

    features = []

    countSelectedFeatures = 0

    for i in range(otimization_solution.size):
        rand = r.generate_uniform_random_number()
        if rand < (1.0 / (1.0 + m.exp(-2 * otimization_solution[i]))):
            #If the feature is selected
            features.append(1)
            countSelectedFeatures += 1
        else:
            features.append(0)

    return features
        
def s2_transfer_function(otimization_solution):
    """ Transfer Function S2: The original Sigmoid fuction

    Args:
        otimization_solution (Agent): The optimization's best agent.
    """

    features = []

    numChosenFeatures = 0

    for i in range(otimization_solution.size):
        rand = r.generate_uniform_random_number()
        if rand < 1.0 / (1.0 + m.exp(-1 * otimization_solution[i])):
            features.append(1)
            numChosenFeatures += 1
        else:
            features.append(0)

    return features

def s3_transfer_function(otimization_solution):
    """ Transfer Function S2: The original Sigmoid fuction

    Args:
        otimization_solution (Agent): The optimization's best agent.
    """

    numChosenFeatures = 0

    for i in (otimization_solution.size()):
        rand = r.generate_uniform_random_number()
        if r < 1.0 / (1.0 + m.exp(-1 * otimization_solution.position[i] / 2)):
            otimization_solution.position[i] = 1
            numChoosenFeatures += 1
        else:
            otimization_solution.position[0]

    #return

def s4_transfer_function(otimization_solution):
    """ Transfer Function S2: The original Sigmoid fuction

    Args:
        otimization_solution (Agent): The optimization's best agent.
    """

    numChosenFeatures = 0

    for i in (otimization_solution.size()):
        rand = r.generate_uniform_random_number()
        if r < 1.0 / (1.0 + m.exp(-1 * otimization_solution.position[i] / 3)):
            otimization_solution.position[i] = 1
            numChoosenFeatures += 1
        else:
            otimization_solution.position[0]

    #return

np.random
