# -*- coding: utf-8 -*-
#Run multiple cm2 models
from __future__ import division
import cosmomodels as c
import numpy as N
from copy import deepcopy
WMAP5PIVOT = N.array([5.25e-60])

fixtures = {
    "mass": {"vars": "mass", "values":N.linspace(6.3265e-6, 6.3427e-6),
             "pivotk":WMAP5PIVOT, "pot": "msqphisq",
             "ystart":N.array([18.0, # \phi_0
                        -0.1, # \dot{\phi_0}
                         0.0, # H - leave as 0.0 to let program determine
                         1.0, # Re\delta\phi_1
                         0.0, # Re\dot{\delta\phi_1}
                         1.0, # Im\delta\phi_1
                         0.0  # Im\dot{\delta\phi_1}
                         ])},
        #
    "lambda": {"vars":"lambda", "values": N.linspace(1e-10, 1e-8), 
               "pivotk":WMAP5PIVOT, "pot": "lambdaphi4",
               "ystart": N.array([25.0,
                                  -1.0,
                                  0.0,
                                  1.0,
                                  0,
                                  1.0,
                                  0])},
     #
     "linde": {"vars":"lambda", "values": N.linspace(1.54e-13, 1.57e-13), 
               "pivotk":WMAP5PIVOT, "pot": "linde",
               "ystart": N.array([25.0,
                                   0.0,
                                   0.0,
                                   1.0,
                                   0,
                                   1.0,
                                   0])},
    #
    "linde2": {"vars":"mass", "values": N.linspace(4.9e-8, 1e-6),
                "pivotk":WMAP5PIVOT, "pot": "linde",
                "ystart": N.array([25.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                    0,
                                    1.0,
                                    0])}
}

def param_vs_spectrum(fixture, nefolds=5):
    """Run tests for a particular parameter, return mass used and spectrum after horizon exit"""
    results = None
    fx = deepcopy(fixture)
    fxystart = fx["ystart"]
    for ps in fx["values"]:
        sim = c.FOCanonicalTwoStage(solver="rkdriver_withks", ystart=fxystart.copy(),
                                    k=fx["pivotk"], potential_func=fx["pot"],
                                    pot_params={fx["vars"]: ps},
                                    tend=83, quiet=True)
        
        
        sim.run(saveresults=False)
        scaledPr = sim.k**3/(2*N.pi**2)*sim.Pr
        tres = sim.findHorizoncrossings()[:,0] + nefolds/sim.tstep_wanted
        Pres = scaledPr[tres.astype(int)].diagonal()[0]
        print "Running model with %s=%s gives scaledPr=%s"%(fx["vars"], str(ps), str(Pres))
        if results is not None:
            results = N.vstack((results, N.array([ps, Pres])))
        else:
            results = N.array([ps, Pres])
        del sim, tres, Pres
    
    return results
    
def param_vs_spectrum_3d(fixture, nefolds=5):
    """Run tests for a particular parameter, return mass used and spectrum after horizon exit"""
    results = None
    fx = deepcopy(fixture)
    fxystart = fx["ystart"]
    for ps in fx["values"]:
        sim = c.FOCanonicalTwoStage(solver="rkdriver_withks", ystart=fxystart.copy(),
                                    k=fx["pivotk"], potential_func=fx["pot"],
                                    pot_params={fx["vars"]: ps},
                                    tend=83, quiet=True)
        
        
        sim.run(saveresults=False)
        scaledPr = sim.k**3/(2*N.pi**2)*sim.Pr
        tres = sim.findHorizoncrossings()[:,0] + nefolds/sim.tstep_wanted
        Pres = scaledPr[tres.astype(int)].diagonal()[0]
        print "Running model with %s=%s gives scaledPr=%s"%(fx["vars"], str(ps), str(Pres))
        if results is not None:
            results = N.vstack((results, N.array([ps, Pres])))
        else:
                results = N.array([ps, Pres])
        del sim, tres, Pres
                
    return results


if __name__ == "__main__":
    print param_vs_spectrum(fixtures["mass"].copy())