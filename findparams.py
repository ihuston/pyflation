#Run multiple cm2 models
import cosmomodels as c
import numpy as N
from pdb import set_trace

def massvspectrum(masses=None, nefolds=5, pivotk=N.array([1.3125e-58], ystart=None)):
    """Run tests for different masses, return mass used and spectrum after horizon exit"""
    if masses is None:
        masses = N.logspace(N.log10(5e-6), N.log10(7e-6))
    if ystart is None:
    results = None        
    for ms in masses:
        sim = c.FOCanonicalTwoStage(solver="rkdriver_withks", ystart=ystart, k=pivotk, mass=ms, tend=83, quiet=True)
        
        print "Running model with mass " , ms
        sim.run(saveresults=False)
        Pr = sim.findspectrum()
        tres = sim.findHorizoncrossings()[:,0] + nefolds/sim.tstep_wanted
        Pres = Pr[tres.astype(int)].diagonal()[0]
                    
        #set_trace()
        if results is not None:
            results = N.vstack((results, N.array([ms, Pres])))
        else:
            results = N.array([ms, Pres])
        del sim
    
    return results

def lambdavspectrum(ls=None, nefolds=5):
    """Run tests for different masses, return mass used and spectrum after horizon exit"""
    if ls is None:
        ls = N.logspace(N.log10(1e-10), N.log10(1e-8))
    
    results = None        
    for l in ls:
        sim = c.CanonicalTwoStage(k=N.array([1.3125e-58]), potential_func=c.cmpotentials.lambdaphi4, 
                                pot_params={"lambda":l}, tend=83, quiet=True,
                                ystart=N.array([25.0,-1.0,0,0,0,0,0]))
        
        print "Running model with lambda " , l
        sim.run(saveresults=False)
        Pr = sim.findPr()
        tres = sim.findHorizoncrossings()[:,0] + nefolds/sim.tstep_wanted
        Pres = Pr[tres.astype(int)].diagonal()[0]
                    
        #set_trace()
        if results is not None:
            results = N.vstack((results, N.array([l, Pres])))
        else:
            results = N.array([l, Pres])
        del sim
    
    return results

def cubevspectrum(ls=None, nefolds=5):
    """Run tests for different masses, return mass used and spectrum after horizon exit"""
    if ls is None:
        ls = N.logspace(N.log10(1e-10), N.log10(1e-1))
    
    results = None        
    for l in ls:
        sim = c.FOCanonicalTwoStage(k=N.array([1.3125e-58]), potential_func="phicubed", 
                                pot_params={"l":l}, tend=90, quiet=True, solver="rkdriver_withks",
                                ystart=N.array([23.0,-1.0,0,0,0,0,0]))
        
        print "Running cube model with l " , l
        sim.run(saveresults=False)
        Pr = sim.Pr
        tres = sim.findHorizoncrossings()[:,0] + nefolds/sim.tstep_wanted
        Pres = Pr[tres.astype(int)].diagonal()[0]
                    
        #set_trace()
        if results is not None:
            results = N.vstack((results, N.array([l, Pres])))
        else:
            results = N.array([l, Pres])
        del sim
    
    return results

if __name__ == "__main__":
    massvspectrum()