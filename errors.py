#!/usr/bin/env python
"""Error estimation for cosmomodels module
"""
from __future__ import division 

import numpy as np
from scipy.integrate import romb
from sosource import getthetaterms
import helpers
from harness import checkkend


kmins_default = [1e-61, 3e-61, 1e-60]
deltaks_default = [1e-61, 3e-61, 1e-60]
nthetas_default = [129, 257, 513]
numsoks_default = [257, 513, 1025]
As_default = [2.7e57]
Bs_default = [1e-62]

def generate_fixtures(kmins=kmins_default, deltaks=deltaks_default, numsoks=numsoks_default,
                      nthetas=nthetas_default, As=As_default, Bs=Bs_default):
    """Generator for fixtures created from cartesian products of input lists."""
    c = helpers.cartesian_product([kmins, deltaks, numsoks, nthetas, As, Bs])
    for now in c:
        fullkmax = checkkend(now[0], None, now[1], now[2])
        fx = {"kmin":now[0], "deltak":now[1], "numsoks":now[2], "fullkmax":fullkmax, "nthetas":now[3], "A":now[4], "B":now[5]}
        yield fx

class ErrorResult(object):
    """Class to hold tests and error results."""
    def __init__(self, fixture):
        """Initialize class"""
        self.fixture = fixture
        self.preconv = {}
        self.postconv = {}
        self.fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
        self.k = self.q = self.fullk[:fixture["numsoks"]]
    
    def do_pre_tests(self):
        """Do all the tests and store results."""
        self.preconv["analytic"] = preconvolution_analytic(self.fixture)
        self.preconv["calced"] = preconvolution_calced(self.fixture)
        
        #calculate errors
        errs = []
        for aterm, cterm in zip(self.preconv["analytic"], self.preconv["calced"]):
            errs.append(rel_error(aterm, cterm))
        self.preconv["rel_err"] = errs
    
    def do_post_tests(self):
        """Do post convolution tests."""
        self.postconv["analytic"]  = postconvolution_analytic(self.fixture)
        self.postconv["calced"] = postconvolution_calced(self.fixture)
        self.postconv["rel_err"] = rel_error(self.postconv["analytic"], self.postconv["calced"])
    
def rel_error(true_soln, est_soln):
    return np.abs(true_soln-est_soln)/np.abs(true_soln)

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
    
    tterms = getthetaterms(ie, dp1, dp1dot)
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

def run_postconvolution(fixturelist = fixtures):
    """Run post convolution tests over list of fixtures."""
    res_list = []
    for fx in fixturelist:
        try:
            e = ErrorResult(fx)
            e.do_post_tests()
            res_list.append(e)
        except Exception, ex:
            print "Error using fixture:\n", str(fx)
            print ex.message
    return res_list
