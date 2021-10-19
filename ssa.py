"""Squirrel Search Algorithm.
"""

import copy

import numpy as np
from numpy.lib.shape_base import dstack 
from tqdm import tqdm
from math import sqrt, pow

import opytimizer.math.random as r
import opytimizer.math.distribution as dis
#import opytimizer.utils.decorator as d
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class SSA(Optimizer):
    """A SSA class, inherited from Optimizer.

    This is the designed class to dife SSA-related 
    variables and methods.

    References:
        M. Jain, V. Singh, and A. Rani. A novel nature-inspired algorithm for
        optimization: Squirrel search algorithm. Swarm and Evolutionary Computation, 44:148–175, 2019.

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (str): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> SSA')
        
        # Overrides its parents class with the receiving params
        super(SSA, self).__init__()
        
        #Setting the author's recomended params

        #Gliding Constant
        self.gc = 1.9

        #Predator Presence Probability
        self.pdp = 0.1

        #Scale Factor
        self.sf = 18
        
        #Distance Gliding
        self.dg = (r.generate_integer_random_number(9, 20) / self.sf)

        #Builds the class
        self.build(params)

        logger.info('Class overrided.')
        
    #Getters and Setters methods
    @property
    def gc(self):
        """

        float: Gliding Constant
        
        """

        return self._gc
    
    @gc.setter
    def gc(self, gc):
        if not isinstance(gc, (float, int)):
            raise e.TypeError('`gc` should be a float or integer')
        if gc < 0:
            raise e.ValueError('`gc` should be >= 0')

        self._gc = gc

    @property
    def pdp(self):
        """

        float: Predator Presence Probability
        
        """

        return self._pdp
    
    @pdp.setter
    def pdp(self, pdp):
        if not isinstance(pdp, (float, int)):
            raise e.TypeError('`pdp` should be a float or integer')
        if pdp < 0:
            raise e.ValueError('`pdp` should be >= 0')
        if pdp > 1:
            raise e.ValueError('`pdp` should be <= 1')

        self._pdp = pdp

    @property
    def fitness_array(self):
        """

        np.array: Agent's fitness np.array

        """

        return self._fitness_array

    @fitness_array.setter
    def fitness_array(self, fitness_array):
        if not isinstance(fitness_array, np.ndarray):
            raise e.TypeError('`fitness_array` should be a numpy array')
        
        self._fitness_array = fitness_array

    @property
    def sf(self):
        """

        float: Scale Factor
        
        """

        return self._sf
    
    @sf.setter
    def sf(self, sf):
        if not isinstance(sf, (float, int)):
            raise e.TypeError('`sf` should be a float or integer')
        if sf < 0:
            raise e.ValueError('`sf` should be >= 0')
        
        self._sf = sf

    @property
    def dg(self):
        """

        float: Distance Gliding
        
        """

        return self._dg
    
    @dg.setter
    def dg(self, dg):
        if not isinstance(dg, (float, int)):
            raise e.TypeError('`dg` should be a float or integer')
        if dg < 0:
            raise e.ValueError('`dg` should be >= 0')

        self._dg = dg


    def _calculate_minimum_seasonal_constant(self, t, tmax):
        """ Method to calculate the minimal season constant

        Args:
            t (int): Number of the current iteration
            tmax (int): Maximum number of algorithm iterations
        """

        # Minimum Seasonal Constant's formula
        minimum_seasonal_constant = pow(10, -6)/pow((365), (t/(tmax/2.5)))

        return minimum_seasonal_constant

    def update(self, space, iteration, total_iterations):

        """Atualizes the flying squirrels's positions in the Search Space.
        
        Args:
            space (Space) : A Space object containing meta-information
            iteration (int) : The current iteration
            total_iterations (int) : The maximum number of iterations

        """
        # Gets all agents
        agent = space.agents

        # Calculates the minimum seasonal constant
        minimum_seasonal_constant = self._calculate_minimum_seasonal_constant(iteration, total_iterations)

        # Returns the indices of the ordered fitness_array 
        ascending_ordered_fitness = np.argsort(self.fitness_array, axis=-1)


        ''' Case 1: The flying squirrels that are in the acorn trees must move toward to hickory nut tree
        '''

        old_hickory_squirrel = agent[ascending_ordered_fitness[0]].position
        
        old_acorn_squirrel = agent[ascending_ordered_fitness[1]].position
        

        # According article, the first tree flying squirrels are on the acorn tree
        for i in range(1, 4):
            
            # Generates random number between [0, 1] for r1
            r1 = r.generate_uniform_random_number()

            acorn_squirrel = ascending_ordered_fitness[i]
            
            # Checks if the probability having predator is low
            if r1 >= self.pdp:
                # If it is, the flying squirrel moves toward to nut tree
                agent[acorn_squirrel].position = agent[acorn_squirrel].position + (self.dg * self.gc) * (old_hickory_squirrel - agent[acorn_squirrel].position)
                
            else:
                # If not is, the flying squirrel moves randomly
                for j in range(space.n_variables):
                    # New position's generation uses uniform distribution
                    agent[acorn_squirrel].position[j] = r.generate_uniform_random_number(agent[acorn_squirrel].lb[j], agent[acorn_squirrel].ub[j], space.n_dimensions)
        
        
        # According article, the nexts (n_agents - 3) can move toward to acorn nut trees or normal trees
        for i in range(4, space.n_agents):
           
            # Randomic number to decides if the i-th flying squirrel will move to acorn nut tree ou normal tree
            aux = r.generate_integer_random_number()

            normal_squirrel = ascending_ordered_fitness[i]
            
            if aux == 0:
                ''' Case 2: The flying squirrels that are on normal trees must move forward to acorn nut trees to fullfill your daily energy
                '''

                # Generates random number between [0, 1] for r2
                r2 = r.generate_uniform_random_number()
                
                # Checks if the probability having predator is low
                if r2 >= self.pdp:
                    # Move toward acorn nut tree
                    agent[normal_squirrel].position = agent[normal_squirrel].position + (self.dg * self.gc) * (old_acorn_squirrel - agent[normal_squirrel].position)
                    
                else:
                    # Moves randomly
                    for j in range(space.n_variables):
                        agent[normal_squirrel].position[j] = r.generate_uniform_random_number(agent[normal_squirrel].lb[j], agent[normal_squirrel].ub[j], space.n_dimensions)
                    
            else:
                '''
                Case 3: Some squirrels that are on normal tree and fullfilled your daily energies on acorn nut tree must move toward
                hickory nut tree so they storage hickory nuts for the winter.
                '''

                # Generates random number between [0, 1] for r3
                r3 = r.generate_uniform_random_number()
                
                # Checks if the probability having predator is low
                if r3 >= self.pdp:
                    # Moves toward hickory nut tree
                    agent[normal_squirrel].position = agent[normal_squirrel].position + (self.dg * self.gc) * (old_hickory_squirrel - agent[normal_squirrel].position)        
                
                else:
                    # Moves randomly
                    for j in range(space.n_variables):
                        agent[normal_squirrel].position[j] = r.generate_uniform_random_number(agent[normal_squirrel].lb[j], agent[normal_squirrel].ub[j], space.n_dimensions)
            
        #Season Constant
        hickory_squirrel = ascending_ordered_fitness[0]

        
        for i in range(0, 3):
            #Constante periódica para cada um dos 3 esquilos voadores nas árvores de bolota
            calc = 0.

            acorn_squirrel = ascending_ordered_fitness[i+1]

            for j in range(space.n_variables):
                calc += (agent[acorn_squirrel].position[j] - agent[hickory_squirrel].position[j]) ** 2
            
            seasonal_constant = sqrt(calc)
            
            if seasonal_constant < minimum_seasonal_constant:
                for j in range(space.n_variables):
                    agent[acorn_squirrel].position[j] = agent[acorn_squirrel].lb[j] + dis.generate_levy_distribution(1.5) * (agent[acorn_squirrel].ub[j] - agent[acorn_squirrel].lb[j])

    def compile(self, space):
        """Compiles additional information that is used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information
        """

        self.fitness_array = np.zeros((space.n_agents))

    def evaluate(self, space, function):
        """Evaluates the search space according to the objective function

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the object function.
            fitness_array(np.array): Agent's fitness

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the fitness value of current agent
            fit = function(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitess to the newer one
                agent.fit = fit
                
                # Save fitness of all agents on this 
                self.fitness_array[i] = agent.fit
                
            # Se o fitness do agente é melhor do que o fitness global
            if agent.fit < space.best_agent.fit:
                # Faz uma cópia do detalhada da melhor posição local do agente para o melhor agente
                space.best_agent.position = copy.deepcopy(agent.position)

                # Faz uma cópia detalhada do fitness do agente atual para o melhor agente
                space.best_agent.fit = copy.deepcopy(agent.fit)