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
import run_config

def compare_one_step(m, nix, srcclass=None, analytic_class=None, calced_class=None):
    """
    Compare the analytic and calculated solutions for equations from `srclass` using the 
    results from `m` at the timestep `nix`. 
    """
    fx = fixtures.fixture_from_model(m)
    
    if srcclass is None:
        srcclass = run_config.srcclass
    
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

def compare_J_terms(m, nix, srcclass=None, analytic_class=None, calced_class=None):
    """
    Compare the analytic and calculated results for each J_term using the results 
    from model `m` at the timestep `nix`
    """
    fx = fixtures.fixture_from_model(m)
    
    if srcclass is None:
        srcclass = run_config.srcclass
    
    if analytic_class is None:
        analytic_class = analyticsolution.NoPhaseBunchDaviesSolution
    if calced_class is None:
        calced_class = calcedsolution.NoPhaseBunchDaviesCalced
    
    asol = analytic_class(fx, srcclass)
    csol = calced_class(fx, srcclass)
    
    #Need to make analytic solution use 128 bit floats to avoid overruns
    asol.srceqns.k = np.float128(asol.srceqns.k)
    
    analytic_Cterms = asol.Cterms(m, nix)
    calced_Cterms = csol.Cterms(m, nix) 
    
    results = []
    
    for afunc, cfunc in zip(asol.jterms, csol.jterms):
        analytic_result = afunc(analytic_Cterms)
        calced_result = cfunc(calced_Cterms)
        results += (analytic_result, calced_result)
        
    return results