''' pyflation.analysis.deltaprel - Module to calculate relative pressure
perturbations.

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''
from __future__ import division

def soundspeeds(Vphi, phidot, H):
    """Sound speeds of the background fields
    
    Arguments
    ---------
    Vphi: array_like
          First derivative of the potential with respect to the fields
          
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
       
    All the arguments should have the same number of dimensions. Vphi and phidot
    should be arrays of the same size, but H should have a dimension of size 1 
    corresponding to the "field" dimension of the other variables.
    """
    try:
        calphasq = 1 + 2*Vphi/(3*H**2*phidot)
    except ValueError:
        raise ValueError("""Arrays need to have the correct shape.
                            Vphi and phidot should have exactly the same shape,
                            and H should have a dimension of size 1 corresponding
                            to the "field" dimension of the others.""")
    return calphasq

def rhodots():
    """Derivative in e-fold time of the energy densities of the individual fields."""
    pass

def fullrhodot():
    """Combined derivative in e-fold time of the energy density of the field"""
    pass

def deltarhosmatrix():
    """Matrix of the first order perturbed energy densities of the field components."""
    pass

def deltaprel():
    """Perturbed relative pressure of the fields given as quantum mode functions."""
    pass

def deltaprelspectrum():
    """Power spectrum of the full perturbed relative pressure."""
    pass