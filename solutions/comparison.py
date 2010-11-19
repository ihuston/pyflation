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

def compare_one_step(m, nix, srcclass=None, analytic_class=None, calced_class=None, fx=None):
    """
    Compare the analytic and calculated solutions for equations from `srclass` using the 
    results from `m` at the timestep `nix`. 
    """
    if fx is None:
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
    csol.srceqns.k = np.float128(csol.srceqns.k)
    csol.srceqns.fullk = np.float128(csol.srceqns.fullk)
    
    analytic_result = asol.full_source_from_model(m, nix)
    calced_result = csol.full_source_from_model(m, nix)
    
    difference = analytic_result - calced_result
    error = np.abs(difference)/np.abs(analytic_result)
    
    result = dict(difference=difference, error=error,
                  analytic_result=analytic_result, calced_result=calced_result,
                  k=asol.srceqns.k)
    return result

def compare_J_terms(m, nix, srcclass=None, analytic_class=None, calced_class=None, 
                    only_calced_Cterms=False, fx=None):
    """
    Compare the analytic and calculated results for each J_term using the results 
    from model `m` at the timestep `nix`
    """
    
    if fx is None:
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
    
        
    #Get background values
    bgvars = m.yresult[nix, 0:3, 0]
    a = m.ainit*np.exp(m.tresult[nix])
    #Get potentials
    potentials = m.potentials(np.array([bgvars[0]]), m.pot_params)
    
    #Set alpha and beta
    alpha = 1/(a*np.sqrt(2))
    beta = a*bgvars[2]
    
    dp1 = csol.get_dp1(csol.srceqns.fullk, alpha=alpha)
    dp1dot = csol.get_dp1dot(csol.srceqns.fullk, alpha=alpha, beta=beta)
    
    #Calculate dphi(q) and dphi(k-q)
    dp1_q = dp1[:csol.srceqns.k.shape[-1]]
    dp1dot_q = dp1dot[:csol.srceqns.k.shape[-1]]  
          
    theta_terms = csol.srceqns.getthetaterms(dp1, dp1dot)
    csol.srceqns.k = np.float128(csol.srceqns.k)
    csol.srceqns.fullk = np.float128(csol.srceqns.fullk)
    
    calced_Cterms = csol.calculate_Cterms(bgvars, a, potentials) 
    if only_calced_Cterms:
        analytic_Cterms = calced_Cterms
    else:
        analytic_Cterms = asol.calculate_Cterms(bgvars, a, potentials)
    
    results = {}
    
    for Jkey in csol.J_terms.iterkeys():
        afunc = asol.J_terms[Jkey]
        cfunc = csol.J_terms[Jkey]
        analytic_result = afunc(asol.srceqns.k, analytic_Cterms, alpha=alpha, beta=beta)
        calced_result = cfunc(theta_terms, dp1_q, dp1dot_q, calced_Cterms)
        diff = analytic_result - calced_result
        err = np.abs(diff)/np.abs(analytic_result)
        results[Jkey] = (diff, err, analytic_result, calced_result)
        
    return results