import numpy as np


class GAfuncs:
    def __init__(self, matwidth, matheight):
        self.X = matwidth
        self.Y = matheight
    
    
    # returns a crossover between parents 1 and 2,
    def crossover(self, parent1, parent2):
        # mask for parent 1
        p1_mask = np.random.randint(2, size=(self.X, self.Y))
        # inverted mask for parent 2
        p2_mask = np.logical_xor(p1_mask, np.ones((self.X, self.Y)))
        
        # give the offspring the sum of the masked values.
        child = (p1_mask * parent1) + (p2_mask * parent2)
        return child
    
    
    # Mutates the matrix. Affected values are multiplied by
    # a factor sampled from a normal distribution with
    # standard deviation = std_dev.
    def mutate(self, matrix, std_dev=0.0001):
        return matrix * np.random.normal(1, std_dev,(self.X, self.Y))
