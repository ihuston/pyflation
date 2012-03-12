"""reheating.py - Cosmological models for reheating simulations

Provides classes for modelling cosmological reheating scenarios.
Especially important classes are:

* :class:`ReheatingBackground` - the class containing derivatives for first order calculation
* :class:`ReheatingFirstOrder` - the class containing derivatives for first order calculation
* :class:`FOReheatingTwoStage` - drives first order calculation 

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.

from pyflation import cosmomodels as c

class ReheatingModels(c.CosmologicalModel):
    '''
    Base class for background and first order reheating model classes.
    '''


    def __init__(self,):
        '''
        Constructor
        '''
        