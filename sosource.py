"""Second order helper functions to set up source term
    $Id: sosource.py,v 1.3 2008/11/14 14:04:05 ith Exp $
    """
import numpy as N
from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import tables
  
def getmodelfromfile(filename):
    """Return model having read model data from a file"""
    pass
    
def getsourceterm(m):
    """Return source term (slow-roll for now), once first order system has been executed."""
    #Make sure first order is complete
    m.checkfirstordercomplete()
    
    #Get first order results (from file or variables)
    phi, phidot, H, dphi1real, dphi1dotreal, dphi1imag, dphi1dotimag = m.yresult[:,i,:] for i in range(7)
    dphi1 = dphi1real + dphi1imag*1j
    dphi1dot = dphi1dotreal + dphi1dotimag*1j
    U, dU, dU2, dU3 = m.potentials()
    
    #Initialize variables to store result
    s2shape = (len(m.k),len(m.k)
    source = []
    #Main loop over each time step
    for nix, n in enumerate(m.fotresult):
        #Single time step
        a = m.ainit*exp(n)
        
        #Initialize result variable for k modes
        s2 = N.empty(s2shape)
        for kix, k in enumerate(m.k):
            #Single k mode
            
            #Result variable for source
            s1 = N.empty_like(m.k)
            for qix, q in enumerate(m.k):
                #Single q mode
                #First major term:
                term1 = 1/(2*N.pi**2) * (potentialterms) * q**2 *dphi1dot[t,N.abs(qix-kix+1)]
                s1[qix] = term1 + term2 + term3
            #add sourceterm for each q
            s2[kix] = s1
        #save results for each q
        source.append(s2)