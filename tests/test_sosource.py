"""Test suite for sosource.py

Uses nosetest suite in numpy.testing
"""
import numpy as np
from numpy.testing import assert_array_almost_equal
import sosource

fixtures = [
    {"kmin": 0.001, "fullkmax": 1.026, "deltak": 0.001, "numsoks": 513, "nthetas": 129, "A": 1, "B": 1},
    {"kmin": 1.000, "fullkmax": 2.025, "deltak": 0.001, "numsoks": 513, "nthetas": 129, "A": 1, "B": 1}    
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

def get_convolution_envelope(fixture):
    """Check convolution using analytic solution. Not a test function."""
    #Init vars
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
    km = k[..., np.newaxis]
    qm = q[np.newaxis, ...]
    theta = np.linspace(0, np.pi, fixture["nthetas"])
    ie = k, q, theta
    
    dp1=A/sqrt(fullk)
    dp1dot=-A/sqrt(fullk) -(A/B)*sqrt(fullk)*1j
    
    #Get calculated array
    klq = np.array([sosource.klessq(onek, q, theta) for onek in k])
    tterms=sosource.getthetaterms(ie, dp1, dp1dot)
    
    ansoln= (2*A/(3*k2*q2)) * ((km+qm)**1.5-(abs(km-qm))**1.5)
    aterm=tterms[0,0]+tterms[0,1]*1j
    env=array([max(abs((ansoln[:,i]-aterm[:,i])/ansoln[:,i])) for i in arange(len(ansoln))])
    return env
