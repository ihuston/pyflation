'''
comparison.py - Comparison of analytic and calculated solutions
Created on 12 Aug 2010

@author: Ian Huston
'''
from __future__ import division

import numpy as np

import analyticsolution
import calcedsolution
import fixtures

def compare_one_step(m, srcclass, nix, analytic_class=None, calced_class=None):
    """
    Compare the analytic and calculated solutions for equations from `srclass` using the 
    results from `m` at the timestep `nix`. 
    """
    fx = fixtures.fixture_from_model(m)
    
    if analytic_class is None:
        analytic_class = analyticsolution.NoPhaseBunchDaviesSolution
    if calced_class is None:
        calced_class = calcedsolution.NoPhaseBunchDaviesCalced
    
    asol = analytic_class(fx, srcclass)
    csol = calced_class(fx, srcclass)
    
    #Need to make analytic solution use 128 bit floats to avoid overruns
    asol.srceqns.k = np.float128(asol.srceqns.k)
    
    analytic_result = asol.full_source_from_model(m, nix)
    calced_result = csol.full_source_from_model(m, nix)
    
    difference = analytic_result - calced_result
    error = np.abs(difference)/np.abs(analytic_result)
    
    return difference, error, analytic_result, calced_result, asol.srceqns.k