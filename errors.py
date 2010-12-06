#!/usr/bin/env python
"""Error estimation for cosmomodels module
"""
from __future__ import division 

import numpy as np
from scipy.integrate import romb
from scipy.special import erf
from scipy import log as clog
from scipy import sqrt
from sourceterm.sosource import getthetaterms
import helpers
from run_config import getkend
import logging

log = logging.getLogger(__name__)
kmins_default = [1e-61, 3e-61, 1e-60]
deltaks_default = [1e-61, 3e-61, 1e-60]
nthetas_default = [129, 257, 513]
numsoks_default = [257, 513, 1025]
As_default = [2.7e57]
Bs_default = [1e-62]
etas_default = [-2.7559960682873626e+68]

def generate_fixtures(kmins=kmins_default, deltaks=deltaks_default, numsoks=numsoks_default,
                      nthetas=nthetas_default, As=As_default, Bs=Bs_default, etas=etas_default):
    """Generator for fixtures created from cartesian products of input lists."""
    c = helpers.cartesian_product([kmins, deltaks, numsoks, nthetas, As, Bs, etas])
    for now in c:
        fullkmax = getkend(now[0], now[1], now[2], kend=None)
        fx = {"kmin":now[0], "deltak":now[1], "numsoks":now[2], "fullkmax":fullkmax, "nthetas":now[3], "A":now[4], "B":now[5], "eta":now[6]}
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
        #calculate errors
        errs = []
        for aterm, cterm in zip(self.postconv["analytic"], self.postconv["calced"]):
            errs.append(rel_error(aterm, cterm))
        self.postconv["rel_err"] = errs

    
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
    
    aterm_analytic = (2*A/(3*km*qm)) * ((km+qm)**1.5-(np.abs(km-qm))**1.5)
    
    bterm_analytic = (-A/(km*qm)**2) * ((1/7)*((km+qm)**3.5-(np.abs(km-qm))**3.5)
        - (1/3)*(km**2 + qm**2) * ((km+qm)**1.5-(np.abs(km-qm))**1.5))
        
    cterm_analytic = - aterm_analytic - (2*A/(5*B*km*qm)) * ((km+qm)**2.5-(np.abs(km-qm))**2.5) * 1j
    
    dterm_analytic =  - bterm_analytic + (A/(B*(qm)**2)) * ( (km**2.5)/9.0*(((1+qm/km)**4.5-(np.abs(1-qm/km))**4.5) )
                                                              -0.2*km**0.5*(km**2 + qm**2) *((1+qm/km)**2.5-(np.abs(1-qm/km))**2.5)) * 1j
    1/0
    analytic_terms = [aterm_analytic, bterm_analytic, cterm_analytic, dterm_analytic]
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
    
    dp1 = A/sqrt(fullk)
    dp1dot = -A/sqrt(fullk) -(A/B)*sqrt(fullk) * 1j
    
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
    
    aterm = aterm_analytic(A, k, kmin, kmax)
    bterm = bterm_analytic(A, k, kmin, kmax)
    cterm = cterm_analytic(A, B, k, kmin, kmax, aterm)
    dterm = dterm_analytic(A, B, k, kmin, kmax, bterm)
    
    return aterm, bterm, cterm, dterm
    
def aterm_analytic(A, k, kmin, kmax):
    a1 = (-1.5*np.pi + 3*np.log(2*sqrt(k)))*k**3
    a2 = (sqrt(kmax)*sqrt(kmax-k) * (-3*k**2 + 14*k*kmax - 8*kmax**2))
    a3 = (sqrt(kmax)*sqrt(kmax+k) * (3*k**2 + 14*k*kmax + 8*kmax**2))
    a4 = -(sqrt(kmin)*sqrt(k-kmin) * (3*k**2 - 14*k*kmin + 8*kmin**2))
    a5 = -(sqrt(kmin)*sqrt(k+kmin) * (3*k**2 + 14*k*kmin + 8*kmin**2))
    a6 = 3*k**3 * (np.arctan(sqrt(kmin)/sqrt(k-kmin)))
    a7 = 3*k**3 * (np.log(2*(sqrt(kmin) + sqrt(k+kmin))))
    a8 = -3*k**3 * (np.log(2*(sqrt(kmax) + sqrt(k+kmax))))
    a9 = -3*k**3 * (np.log(2*(sqrt(kmax) + sqrt(kmax-k))))
    
    aterm_analytic = (4*np.pi*A**2)/(3*24*k) * (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9)
    return aterm_analytic

def bterm_analytic(A, k, kmin, kmax):
    b1a = ( np.log( (sqrt(kmax-k) + sqrt(kmax)) / ( sqrt(kmax+k) +sqrt(kmax) ) ) )
    b1b = ( np.log( (sqrt(k+kmin) +sqrt(kmin)) / (sqrt(k)) ) + 0.5*np.pi)
    b1c = ( -np.arctan(sqrt(kmin)/(sqrt(k-kmin))))
    b1 = 252 * k**2 * (b1a + b1b + b1c)
    
    b2 = sqrt(kmax) * (-(260*k - 32*kmax**2/k)*(sqrt(k+kmax) + sqrt(kmax-k) ) 
                          +(-88*kmax + 64*kmax**3/k**2)*(sqrt(kmax-k) - sqrt(k+kmax)))
    b3 = sqrt(kmin) * ((260*k - 32*kmin**2/k)*(sqrt(k+kmin) - sqrt(k-kmin) ) 
                          +(-88*kmin + 64*kmin**3/k**2)*(sqrt(k+kmin) + sqrt(k-kmin)))
                          
    bterm_analytic = -(2*np.pi*A**2)/(1344) * (b1 + b2 + b3)
    return bterm_analytic
    
def bterm_analytic2(A, k, kmin, kmax):
    b1a = np.log( (sqrt(k))/(sqrt(k+kmin) + sqrt(kmin)) )
    b1b = ( np.log( (sqrt(k+kmax) + sqrt(kmax))/(sqrt(kmax-k) + sqrt(kmax)) )  -0.5*np.pi )
    b1c = np.arctan( (sqrt(kmin))/(sqrt(k-kmin)) )

    b1 = -252*k**4 * (b1a + b1b + b1c)
    b2 = sqrt(kmax)*( (-260*k**3 + 32*k*kmax**2) * (sqrt(k+kmax) + sqrt(kmax-k))
                        +(88*k**2*kmax - 64*kmax**3) * (sqrt(k+kmax) - sqrt(kmax-k)) )
    b3 = sqrt(kmin)*( (260*k**3 - 32*k*kmin**2) * (sqrt(k+kmin) - sqrt(k-kmin))
                        +(-88*k**2*kmin + 64*kmin**3) * (sqrt(k+kmin) + sqrt(k-kmin)) )
        
    bterm_analytic = -(2*np.pi*A**2)/(1344*k**2) * (b1 + b2 + b3)
    return bterm_analytic, (b1,b2,b3)
    
def cterm_analytic(A, B, k, kmin, kmax, aterm):
    
    c1a = ( np.log( (sqrt(kmax-k) + sqrt(kmax))/(sqrt(kmax+k) + sqrt(kmax)) ) )
    c1b = ( np.log( (sqrt(k+kmin) + sqrt(kmin))/(sqrt(k)) )  -0.5*np.pi) 
    c1c = ( + np.arctan( (sqrt(kmin))/(sqrt(k-kmin)) ))
    
    c1 = 15*k**3 * (c1a + c1b + c1c)
    c2 = sqrt(kmax)  * ( (15*k**2 + 136*kmax**2) * (sqrt(k+kmax) + sqrt(kmax-k))
                           +(118*k*kmax +48*kmax**3/k) * (sqrt(k+kmax) - sqrt(kmax-k)))
    c3 = -sqrt(kmin) * ( (15*k**2 + 136*kmin**2) * (sqrt(k+kmin) + sqrt(k-kmin))
                           +(118*k*kmin +48*kmin**3/k) * (sqrt(k+kmin) - sqrt(k-kmin)) ) 
    
    chalf = -(4*np.pi*A**2)/(192*5*B) * 1j * (c1 + c2 + c3)
    cterm = -aterm + chalf
    return cterm

def dterm_analytic(A, B, k, kmin, kmax, bterm):
    d1a = ( np.log( (sqrt(k+kmax) + sqrt(kmax))/(sqrt(k+kmin) +sqrt(kmin)) ) )
    d1b = ( np.log( (sqrt(kmax-k) + sqrt(kmax))/(sqrt(k)) ) -0.5*np.pi )
    d1c = ( + np.arctan( (sqrt(kmin))/(sqrt(k-kmin)) ) )    
    
    d1 = -135*k**3 * (d1a + d1b + d1c)
    d2 = +sqrt(kmax) * ( (-185*k**2 + 168*kmax**2 -32*kmax**4/k**2) * (sqrt(k+kmax) - sqrt(kmax-k))
                           +(70*k*kmax +16*kmax**3/k) * (sqrt(k+kmax) + sqrt(kmax-k)) )
    d3 = -sqrt(kmin) * ( (-185*k**2 + 168*kmin**2 -32*kmin**4/k**2) * (sqrt(k+kmin) - sqrt(k-kmin))
                           +(70*k*kmin +16*kmin**3/k) * (sqrt(k+kmin) + sqrt(k-kmin)) )
    
    dhalf = (np.pi*A**2)/(B*900) * 1j * (d1 + d2 + d3)
    dterm = -bterm + dhalf
    return dterm

def postconvolution_calced(fixture):
    """Return calculated solution for post convolution terms."""
    preconv = preconvolution_calced(fixture)
    preaterm, prebterm, precterm, predterm = preconv
    
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    q = fullk[np.newaxis, :fixture["numsoks"]]
    dp1 = fixture["A"]/sqrt(q)
    
    aterm = 2*np.pi * q**2 * dp1 * preaterm
    integrated_a = romb(aterm, fixture["deltak"])
    
    bterm = 2*np.pi * q**2 * dp1 * prebterm
    integrated_b = romb(bterm, fixture["deltak"])
    
    cterm = 2*np.pi * q**2 * dp1 * precterm
    integrated_c = romb(cterm, fixture["deltak"])
    
    dterm = 2*np.pi * q**2 * dp1 * predterm
    integrated_d = romb(dterm, fixture["deltak"])
    
    return integrated_a, integrated_b, integrated_c, integrated_d
    
def postconvolution_generic(fixture, atodterms):
    """Return calculated solution for post convolution terms."""
    
    preaterm, prebterm, precterm, predterm = atodterms
    
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    q = fullk[np.newaxis, :fixture["numsoks"]]
    dp1 = fixture["A"]/sqrt(q)
    
    aterm = 2*np.pi * q**2 * dp1 * preaterm
    integrated_a = romb(aterm, fixture["deltak"])
    
    bterm = 2*np.pi * q**2 * dp1 * prebterm
    integrated_b = romb(bterm, fixture["deltak"])
    
    cterm = 2*np.pi * q**2 * dp1 * precterm
    integrated_c = romb(cterm, fixture["deltak"])
    
    dterm = 2*np.pi * q**2 * dp1 * predterm
    integrated_d = romb(dterm, fixture["deltak"])
    
    return integrated_a, integrated_b, integrated_c, integrated_d
    
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
    A = (a*sqrt(2))**-1
    B = a*bgvars[2]
    dp1 = A/sqrt(fullk)
    dp1dot = -A/sqrt(fullk) -(A/B)*sqrt(fullk)*1j
    potentials = m.potentials(m.bgmodel.yresult[nix], m.pot_params)
    
    log.info("Getting thetaterms...")
    tterms = getthetaterms(ie, dp1, dp1dot)
    log.info("Getting source term integrands...")
    src_terms = get_all_src_terms(bgvars, a, potentials, ie, dp1, dp1dot, tterms)
    return tterms, src_terms
   
def get_both_src_integrands(m, fixture, nix=0):
    """Return source term integrands for the given fixture."""
    #Init vars
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
       
    theta = np.linspace(0, np.pi, fixture["nthetas"])
    ie = k, q, theta
    bgvars = m.bgmodel.yresult[nix,0:3]
    a = m.ainit*np.exp(m.bgmodel.tresult[nix])
    A = (a*sqrt(2))**-1
    B = a*bgvars[2]
    dp1 = A/sqrt(fullk)
    dp1dot = -A/sqrt(fullk) -(A/B)*sqrt(fullk)*1j
    potentials = m.potentials(m.bgmodel.yresult[nix], m.pot_params)
    
    log.info("Getting thetaterms...")
    calced_tterms = preconvolution_calced(fixture)
    analytic_tterms = preconvolution_analytic(fixture) 
    log.info("Getting source term integrands...")
    calced_src_terms = get_all_src_terms(bgvars, a, potentials, ie, dp1, dp1dot, calced_tterms)
    analytic_src_terms = get_all_src_terms(bgvars, a, potentials, ie, dp1, dp1dot, analytic_tterms)
    return analytic_src_terms, calced_src_terms

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
    aterm, bterm, cterm, dterm = theta_terms
    
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

def do_convolution(src_terms, dk):
    src_integrand = src_terms[0] + src_terms[1] + src_terms[2]
    return romb(src_integrand, dx=dk)

def preconv_calced_withphase(fixture):
    """Return calculates solution for pre-convolution terms."""
    #Init vars
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
    A = fixture["A"]
    B = fixture["B"]    
    theta = np.linspace(0, np.pi, fixture["nthetas"])
    ie = k, q, theta
    eta = fixture["eta"]
    
    dp1 = A/sqrt(fullk) * np.exp(-fullk*eta*1j)
    dp1dot = -A/sqrt(fullk) -(A/B)*sqrt(fullk) * 1j
    
    tterms = getthetaterms(ie, dp1, dp1dot)
    aterm = tterms[0,0] + tterms[0,1]*1j
    bterm = tterms[1,0] + tterms[1,1]*1j
    cterm = tterms[2,0] + tterms[2,1]*1j
    dterm = tterms[3,0] + tterms[3,1]*1j
    calced_terms = [aterm, bterm, cterm, dterm]
    return calced_terms

def preconv_analytic_withphase(fixture):
    """Return analytic solution for preconvolution terms with a phase."""
    fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
    k = q = fullk[:fixture["numsoks"]]
    A = fixture["A"]
    B = fixture["B"]
    eta = fixture["eta"]
    km = k[..., np.newaxis]
    qm = q[np.newaxis, ...]
    
    erfi = lambda z: -1*1j*erf(1j*z)
    
    aterm_analytic = (A/(km*qm)) * ( ( np.exp(0.75*clog(-1)) * sqrt(np.pi) / (2*np.exp(1.5*clog(eta))) ) * ( erfi( np.exp(0.75*clog(-1)) *sqrt(eta) *sqrt(km+qm)  ) 
                                                                                     - erfi( np.exp(0.75*clog(-1)) *sqrt(eta) *sqrt(km-qm)  ) ) 
                                     + (1j/(eta)) * (sqrt(km+qm)*np.exp(-1*1j*eta*(km+qm))  - sqrt(km-qm)*np.exp(-1*1j*eta*(km-qm))) )
    
    return aterm_analytic
    
    