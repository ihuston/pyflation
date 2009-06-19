"""Test suite for sosource.py

Uses nosetest suite in numpy.testing
"""
from __future__ import division 

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.integrate import romb
import sosource

fixtures = [
    {"kmin": 0.001, "fullkmax": 1.026, "deltak": 0.001, "numsoks": 513, "nthetas": 129, "A": 1, "B": 1},
    {"kmin": 0.001, "fullkmax": 1.026, "deltak": 0.001, "numsoks": 513, "nthetas": 257, "A": 1, "B": 1},
    {"kmin": 0.001, "fullkmax": 1.026, "deltak": 0.001, "numsoks": 513, "nthetas": 513, "A": 1, "B": 1},
    {"kmin": 0.001, "fullkmax": 1.026, "deltak": 0.001, "numsoks": 513, "nthetas": 65, "A": 1, "B": 1},
    {"kmin": 1.000, "fullkmax": 2.025, "deltak": 0.001, "numsoks": 513, "nthetas": 129, "A": 1, "B": 1},
    {"kmin": 1.000, "fullkmax": 2.025, "deltak": 0.001, "numsoks": 513, "nthetas": 257, "A": 1, "B": 1} 
]

def check_klessq(fixture):
    """Do checking of klessq versus analytical result."""
    #Init vars
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"] + fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
    km = k[..., np.newaxis] #k matrix
    qm = q[np.newaxis, ...] #q matrix
    theta = np.linspace(0, np.pi, fixture["nthetas"])
    
    #Get calculated array
    klq = np.array([sosource.klessq(onek, q, theta) for onek in k])
    
    #Test cases
    assert_array_almost_equal(klq[:,:,-1], (km + qm))
    assert_array_almost_equal(klq[:,:,0], abs(km - qm))
    assert_array_almost_equal(klq[:,:, fixture["nthetas"]/2], np.sqrt(km**2 + qm**2))
    
def test_klessq():
    """klessq values check"""
    for fx in fixtures:
        yield check_klessq, fx

def preconvolution_analytic(fixture):
    """Return analytic solution for dp1=A/sqrt(k) and dp1dot=-A/sqrt(k) - A/B*sqrt(k)*1j"""
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
    A = fixture["A"]
    B = fixture["B"]
    km = k[..., np.newaxis]
    qm = q[np.newaxis, ...]
    
    aterm_analytic = (2*A/(3*km*qm)) * ((km+qm)**1.5-(abs(km-qm))**1.5)
    
    bterm_analytic = (-A/(km*qm)**2) * ((1/7)*((km+qm)**3.5-(abs(km-qm))**3.5)
        - (1/3)*(km**2 + qm**2) * ((km+qm)**1.5-(abs(km-qm))**1.5))
        
    cterm_analytic = - aterm_analytic - (2*A/(5*B*km*qm)) * ((km+qm)**2.5-(abs(km-qm))**2.5) * 1j
    
    analytic_terms = [aterm_analytic, bterm_analytic, cterm_analytic]
    return analytic_terms

def preconvolution_calced(fixture):
    """Return calculates solution for pre-convolution terms."""
     #Init vars
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
    A = fixture["A"]
    B = fixture["B"]    
    theta = np.linspace(0, np.pi, fixture["nthetas"])
    ie = k, q, theta
    
    dp1 = A/np.sqrt(fullk)
    dp1dot = -A/np.sqrt(fullk) -(A/B)*np.sqrt(fullk)*1j
    
    tterms = sosource.getthetaterms(ie, dp1, dp1dot)
    aterm = tterms[0,0] + tterms[0,1]*1j
    bterm = tterms[1,0] + tterms[1,1]*1j
    cterm = tterms[2,0] + tterms[2,1]*1j
    calced_terms = [aterm, bterm, cterm]
    return calced_terms

def preconvolution_envelope(fixture):
    """Check convolution using analytic solution. Not a test function."""
    calced_terms = preconvolution_calced(fixture)
    analytic_terms = preconvolution_analytic(fixture)    
    
    envs =[]
    for analytic_term, calced_term in zip(analytic_terms, calced_terms):
        envs.append(np.array([max(abs(analytic_term[:,i]-calced_term[:,i])/abs(analytic_term[:,i]))
                             for i in np.arange(len(analytic_term))]))
    return envs

def postconvolution_analytic(fixture):
    """Return analytic solution for post convolution terms."""
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = fullk[:fixture["numsoks"]]
    A = fixture["A"]
    B = fixture["B"]
    kmin = k[0]
    kmax = k[-1]
    
    a1 = (-1.5*np.pi + 3*np.log(2*np.sqrt(k)))*k**3
    a2 = (np.sqrt(kmax)*np.sqrt(kmax-k) * (-3*k**2 + 14*k*kmax - 8*kmax**2))
    a3 = (np.sqrt(kmax)*np.sqrt(kmax+k) * (3*k**2 + 14*k*kmax + 8*kmax**2))
    a4 = -(np.sqrt(kmin)*np.sqrt(k-kmin) * (3*k**2 - 14*k*kmin + 8*kmin**2))
    a5 = -(np.sqrt(kmin)*np.sqrt(k+kmin) * (3*k**2 + 14*k*kmin + 8*kmin**2))
    a6 = 3*k**3 * (np.arctan(np.sqrt(kmin)/np.sqrt(k-kmin)))
    a7 = 3*k**3 * (np.log(2*(np.sqrt(kmin) + np.sqrt(k+kmin))))
    a8 = -3*k**3 * (np.log(2*(np.sqrt(kmax) + np.sqrt(k+kmax))))
    a9 = -3*k**3 * (np.log(2*(np.sqrt(kmax) + np.sqrt(kmax-k))))
    
    aterm_analytic = (4*np.pi*A**2)/(3*24*k) * (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9)
    
    return aterm_analytic

def postconvolution_calced(fixture):
    """Return calculated solution for post convolution terms."""
    preconv = preconvolution_calced(fixture)
    preaterm = preconv[0]
    
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    q = fullk[np.newaxis, :fixture["numsoks"]]
    dp1 = fixture["A"]/np.sqrt(q)
    
    aterm = 2*np.pi * q**2 * dp1 * preaterm
    integrated = romb(aterm, fixture["deltak"])
    return integrated
    
def postconvolution_envelope(fixture):
    """Return envelope of errors in postconvolution calculated term versus analytic."""
    pass
