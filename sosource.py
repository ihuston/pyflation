"""Second order helper functions to set up source term
    $Id: sosource.py,v 1.7 2008/11/24 14:52:03 ith Exp $
    """

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import tables
import numpy as N
from scipy import integrate
import helpers
  
def getmodelfromfile(filename):
    """Return model having read model data from a file"""
    pass      
    
def getsourceintegrand(m):
    """Return source term (slow-roll for now), once first order system has been executed."""
    #Make sure first order is complete
    m.checkfirstordercomplete()
    
    #Get first order ICs:
    nanfiller = [m.getfoystart(m.tresult[tix], [tix]) for tix in xrange(len(m.tresult))]
    nanfiller = N.array(nanfiller)
    
    #switch nans for ICs in m.yresult
    myr = m.yresult.copy()
    are_nan = N.isnan(myr)
    myr[are_nan] = nanfiller[are_nan]
    
    #Get first order results (from file or variables)
    phi, phidot, H, dphi1real, dphi1dotreal, dphi1imag, dphi1dotimag = [myr[:,i,:] for i in range(7)]
    dphi1 = dphi1real + dphi1imag*1j
    dphi1dot = dphi1dotreal + dphi1dotimag*1j
    pottuple = m.potentials(N.rollaxis(myr, 1))
    #Get potentials in right shape
    pt = []
    for p in pottuple:
        if N.shape(p) != N.shape(pottuple[0]):
            pt.append(p*N.ones_like(pottuple[0]))
        else:
            pt.append(p)
    U, dU, dU2, dU3 = pt
    
    #Initialize variables to store result
    s2shape = (len(m.k),len(m.k))
    source = []
    #Main loop over each time step
    for nix, n in enumerate(m.tresult):
        #Single time step
        a = m.ainit*N.exp(n)
        
        #Initialize result variable for k modes
        s2 = N.empty(s2shape)
        for kix, k in enumerate(m.k):
            #Single k mode
            
            #Result variable for source
            s1 = N.empty_like(m.k)
            for qix, q in enumerate(m.k):
                #Single q mode
                #Check abs(qix-kix)-1 is not negative
                dphi1ix = N.abs(qix-kix) -1
                if dphi1ix < 0:
                    dp1diff = dp1dotdiff = 0
                else:
                    dp1diff = dphi1[nix, dphi1ix]
                    dp1dotdiff = dphi1dot[nix, dphi1ix]
                
                #First major term:
                term1 = (1/(2*N.pi**2) * (1/H[nix,kix]**2) * (dU3[nix,kix] + 3*phidot[nix,kix]*dU2[nix,kix]) 
                            * q**2*dp1diff*dphi1[nix,qix])
                #Second major term:
                term2 = (1/(2*N.pi**2) * ((1/(a*H[nix,kix]) + 0.5)*q**2 - 2*(q**4/k**2)) * dp1dotdiff * dphi1dot[nix,qix])
                #Third major term:
                term3 = (1/(2*N.pi**2) * 1/(a*H[nix,kix])**2 * (2*(q**6/k**2) + 2.5*q**4 + 2*(k*q)**2) * phidot[nix,kix] 
                            * dp1diff * dphi1[nix,qix])
                s1[qix] = term1 + term2 + term3
            #add sourceterm for each q
            s2[kix] = s1
        #save results for each q
        source.append(s2)
    source = N.array(source)
    return source
            
def getsource(m):
    """Return integrated source function for model m using romberg integration."""
    if not helpers.ispower2(len(m.k)-1):
        raise AttributeError("Need to have 2**n + 1 different k values for integration.")
    msource = integrate.romb(getsourceintegrand(m))
    return msource

def savetofile(filename, m, sourceterm=None):
    """Save the source term to the hdf5 file with filename."""
    if not filename or not m:
        raise TypeError("Need to specify both filename and model variable.")
    if sourceterm is None:
        print "No source term given, calculating now..."
        sourceterm = getsource(m)
                
    try:
        rf = tables.openFile(filename, "a", "Source term result")
        try:
            if not "results" in rf.root:
                rf.createGroup(rf.root, "results", "Results")
            if not "sourceterm" in rf.root.results:
                stab = rf.createTable(rf.root.results, "sourceterm", sourcetermdict(m, sourceterm))
            else: 
                stab = rf.root.results.sourceterm
            #Fill roww with data
            tabrow = stab.row
            for kix, k in enumerate(m.k):
                tabrow["k"] = k
                tabrow["source"] = sourceterm[:,kix]
                tabrow.append()
            stab.flush()
        finally:
            rf.close()
    except IOError:
        raise

def sourcetermdict(m, sourceterm):
    """Return dictionary with source term table configuration for HDF5 file"""
    if not m or sourceterm is None:
        raise TypeError("Need to specify both model and sourceterm.")
    sdict = {
    "k" : tables.Float64Col(),
    "source" : tables.Float64Col(sourceterm[:,0].shape)}
    
    return sdict