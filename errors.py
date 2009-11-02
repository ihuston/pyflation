#!/usr/bin/env python
"""Error estimation for cosmomodels module
"""
from __future__ import division 

import numpy as np
from scipy.integrate import romb
from sosource import getthetaterms
import helpers
from harness import checkkend
import logging

log = logging.getLogger(__name__)
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
        self.postconv["rel_err_a"] = rel_error(self.postconv["analytic"][0], self.postconv["calced"][0])
        self.postconv["rel_err_b"] = rel_error(self.postconv["analytic"][1], self.postconv["calced"][1])
    
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
    dterm = tterms[3,0] + tterms[3,1]*1j
    calced_terms = [aterm, bterm, cterm, dterm]
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
    
    b1a = ( np.log( (np.sqrt(kmax-k) + np.sqrt(kmax)) / ( np.sqrt(kmax+k) +np.sqrt(kmax) ) ) )
    b1b = ( np.log( (np.sqrt(k+kmin) +np.sqrt(kmin)) / (np.sqrt(k)) ) + 0.5*np.pi)
    b1c = ( -np.arctan(np.sqrt(kmin)/(np.sqrt(k-kmin))))
    b1 = 1323 * k**4 * (b1a + b1b + b1c)
    
    b2 = np.sqrt(kmax) * (-(1877*k**3 + 472*k*kmax**2)*(np.sqrt(k+kmax) + np.sqrt(kmax-k) ) 
                          +(626*k**2*kmax + 400*kmax**3)*(np.sqrt(kmax-k) - np.sqrt(k+kmax)))
    b3 = np.sqrt(kmin) * ((1877*k**3 + 472*k*kmin**2)*(np.sqrt(k+kmin) - np.sqrt(k-kmin) ) 
                          +(626*k**2*kmin + 400*kmin**3)*(np.sqrt(k+kmin) + np.sqrt(k-kmin)))
                          
    bterm_analytic = -(2*np.pi*A**2)/(1344*k**2) * (b1 + b2 + b3)
    
    return aterm_analytic, bterm_analytic, (b1,b2,b3)

def postconvolution_calced(fixture):
    """Return calculated solution for post convolution terms."""
    preconv = preconvolution_calced(fixture)
    preaterm = preconv[0]
    prebterm = preconv[1]
    
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    q = fullk[np.newaxis, :fixture["numsoks"]]
    dp1 = fixture["A"]/np.sqrt(q)
    
    aterm = 2*np.pi * q**2 * dp1 * preaterm
    integrated_a = romb(aterm, fixture["deltak"])
    
    bterm = 2*np.pi * q**2 * dp1 * prebterm
    integrated_b = romb(bterm, fixture["deltak"])
    
    return integrated_a, integrated_b
    
def postconvolution_envelope(fixture):
    """Return envelope of errors in postconvolution calculated term versus analytic."""
    pass

def run_postconvolution(fixturelist = generate_fixtures()):
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

def src_term_integrands(m, fixture, nix=0):
    """Return source term integrands for the given fixture."""
     #Init vars
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
       
    theta = np.linspace(0, np.pi, fixture["nthetas"])
    ie = k, q, theta
    bgvars = m.bgmodel.yresult[nix,0:3]
    a = m.ainit*np.exp(m.bgmodel.tresult[nix])
    A = (a*np.sqrt(2))**-1
    B = a*bgvars[2]
    dp1 = A/np.sqrt(fullk)
    dp1dot = -A/np.sqrt(fullk) -(A/B)*np.sqrt(fullk)*1j
    potentials = m.potentials(m.bgmodel.yresult[nix])
    
    log.info("Getting thetaterms...")
    tterms = getthetaterms(ie, dp1, dp1dot)
    log.info("Getting source term integrands...")
    src_terms = get_all_src_terms(bgvars, a, potentials, ie, dp1, dp1dot, tterms)
    return tterms, src_terms
    
def get_all_src_terms(bgvars, a, potentials, integrand_elements, dp1, dp1dot, theta_terms):
    """Return unintegrated slow roll source term.
    
    The source term before integration is calculated here using the slow roll
    approximation. This function follows the revised version of Eq (5.8) in 
    Malik 06 (astro-ph/0610864v5).
    
    Parameters
    ----------
    bgvars: tuple
            Tuple of background field values in the form `(phi, phidot, H)`
    
    a: float
       Scale factor at the current timestep, `a = ainit*exp(n)`
    
    potentials: tuple
                Tuple of potential values in the form `(U, dU, dU2, dU3)`
    
    integrand_elements: tuple 
         Contains integrand arrays in order (k, q, theta)
            
    dp1: array_like
         Array of known dp1 values
             
    dp1dot: array_like
            Array of dpdot1 values
             
    theta_terms: array_like
                 3-d array of integrated theta terms of shape (4, len(k), len(q))
             
    Returns
    -------
    src_integrand: array_like
        Array containing the unintegrated source terms for all k and q modes.
        
    References
    ----------
    Malik, K. 2006, JCAP03(2007)004, astro-ph/0610864v5
    """
    #Unpack variables
    phi, phidot, H = bgvars
    U, dU, dU2, dU3 = potentials
    k = integrand_elements[0][...,np.newaxis]
    q = integrand_elements[1][np.newaxis, ...]
    #Calculate dphi(q) and dphi(k-q)
    dp1_q = dp1[np.newaxis,:q.shape[-1]]
    dp1dot_q = dp1dot[np.newaxis,q.shape[-1]]
    atmp, btmp, ctmp, dtmp = theta_terms
    aterm = atmp[0] + atmp[1]*1j
    bterm = btmp[0] + btmp[1]*1j
    cterm = ctmp[0] + ctmp[1]*1j
    dterm = dtmp[0] + dtmp[1]*1j
    
    #Calculate unintegrated source term
    #First major term:
    if dU3!=0.0:
        src_integrand = (1/(2*np.pi)**2)*((1/H**2) * dU3 * q**2 * dp1_q * aterm)
    else:
        src_integrand = np.zeros_like(aterm)
    #Second major term:
    src_integrand2 = (1/(2*np.pi)**2) * (phidot/((a*H)**2)) * ((3*dU2*(a*q)**2 + 3.5*q**4 + 2*(k**2)*(q**2))*aterm 
                      - (4.5 + (q/k)**2)* k * (q**3) * bterm) * dp1_q
    #Third major term:
    src_integrand3 = (1/(2*np.pi)**2) * (phidot * ((-1.5*q**2)*cterm + (2 - (q/k)**2)*k*q*dterm) * dp1dot_q)
    #Multiply by prefactor
    src_terms = src_integrand, src_integrand2, src_integrand3
    
    return src_terms