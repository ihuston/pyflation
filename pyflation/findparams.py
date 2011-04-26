# -*- coding: utf-8 -*-
""" findparams.py - Find values of parameters in potentials using WMAP normalization

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

This module provides helper functions to find the correct parameter values 
necessary in the potential functions used elsewhere in the pyflation package.

The fixtures dictionary contains standard setups for the different potentials,
including a range of values for the parameters. For example in fixtures["msqphisq"]
initial conditions for the quadratic model are given and a range of mass 
values in specified under the "values" key. 

When used with the param_vs_spectrum function, the first order calculation 
will be run with each of these mass values and the power spectrum of scalar
curvature perturbations are computed. 
The results of the param_vs_spectrum function are returned as an array of 
the (mass, powerspectrum) values.

The interpolate_parameter function takes the results from param_vs_spectrum and
calculates the correct value of the parameter to match the power spectrum to the
WMAP 7 year value given by WMAP7PR, using spline interpolation.

The run_and_interpolate function is a convenience function which takes a fixture
and both runs the param_vs_spectrum and interpolate_parameter functions to find
the matching value of the parameter.

Please note that depending on the number of parameter values specified it may
take a long time to run the first order calculation the specified number of times. 
 
"""
from __future__ import division

import numpy as np
from copy import deepcopy
import sys
import logging
from scipy import interpolate

#Set up logging with basic configuration
logging.basicConfig()

try:
    #Local modules from pyflation package
    from pyflation import helpers, configuration
    from pyflation import cosmomodels as c
    _debug = configuration._debug
except ImportError,e:
    if __name__ == "__main__":
        msg = """Pyflation module needs to be available. 
Either run this script from the base directory as bin/firstorder.py or add directory enclosing pyflation package to PYTHONPATH."""
        print msg, e
        sys.exit(1)
    else:
        raise
    
WMAP7PIVOT = np.array([5.25e-60])
WMAP7PR = 2.457e-9

fixtures = {
    "msqphisq": {"vars": ["mass"], "values":[np.linspace(6.3265e-6, 6.34269e-6)],
             "pivotk":WMAP7PIVOT, "pot": "msqphisq",
             "ystart":np.array([18.0, # \phi_0
                        -0.1, # \dot{\phi_0}
                         0.0, # H - leave as 0.0 to let program determine
                         1.0, # Re\delta\phi_1
                         0.0, # Re\dot{\delta\phi_1}
                         1.0, # Im\delta\phi_1
                         0.0  # Im\dot{\delta\phi_1}
                         ])},
        #
    "lambdaphi4": {"vars":["lambda"], "values": [np.linspace(1e-10, 1e-8)], 
               "pivotk":WMAP7PIVOT, "pot": "lambdaphi4",
               "ystart": np.array([25.0,
                                  -1.0,
                                  0.0,
                                  1.0,
                                  0,
                                  1.0,
                                  0])},
    #
    "phi2over3": {"vars":["sigma"], "values": [np.linspace(1e-10, 1e-8)], 
               "pivotk":WMAP7PIVOT, "pot": "phi2over3",
               "ystart": np.array([10.0,
                                  0.0,
                                  0.0,
                                  1.0,
                                  0,
                                  1.0,
                                  0])},
     #
     "linde": {"vars":["mass","lambda"], "values": [np.linspace(4.9e-8,6e-8), np.linspace(1.54e-13, 1.57e-13)], 
               "pivotk":WMAP7PIVOT, "pot": "linde",
               "ystart": np.array([25.0,
                                   0.0,
                                   0.0,
                                   1.0,
                                   0,
                                   1.0,
                                   0])},
                                   
     "hybrid2and4": {"vars":["mass","lambda"], "values": [np.linspace(4.9e-8,6e-8), np.linspace(1.54e-13, 1.57e-13)], 
               "pivotk":WMAP7PIVOT, "pot": "hybrid2and4",
               "ystart": np.array([25.0,
                                   0.0,
                                   0.0,
                                   1.0,
                                   0,
                                   1.0,
                                   0])},
    #
     "bump_potential": {"vars": ["mass"], "values":[np.linspace(6.3e-6, 6.4e-6)],
             "pivotk":WMAP7PIVOT, "pot": "bump_potential",
             "ystart":np.array([18.0, # \phi_0
                        -0.1, # \dot{\phi_0}
                         0.0, # H - leave as 0.0 to let program determine
                         1.0, # Re\delta\phi_1
                         0.0, # Re\dot{\delta\phi_1}
                         1.0, # Im\delta\phi_1
                         0.0  # Im\dot{\delta\phi_1}
                         ])},
     "step_potential": {"vars": ["mass"], "values":[np.linspace(6.3e-6, 6.4e-6)],
             "pivotk":WMAP7PIVOT, "pot": "step_potential",
             "ystart":np.array([18.0, # \phi_0
                        -0.1, # \dot{\phi_0}
                         0.0, # H - leave as 0.0 to let program determine
                         1.0, # Re\delta\phi_1
                         0.0, # Re\dot{\delta\phi_1}
                         1.0, # Im\delta\phi_1
                         0.0  # Im\dot{\delta\phi_1}
                         ])},
}

def param_vs_spectrum(fixture, nefolds=5):
    """Run tests for a particular parameter, return mass used and spectrum after horizon exit"""
    #Variable to store results
    results = None
    logging.info("Starting parameter run with potential %s." % fixture["pot"])
    fx = deepcopy(fixture)
    fxystart = fx["ystart"]
    values_matrix = helpers.cartesian(fx["values"])
    for ps in values_matrix:
        sim = c.FOCanonicalTwoStage(solver="rkdriver_tsix", ystart=fxystart.copy(),
                                    k=fx["pivotk"], potential_func=fx["pot"],
                                    pot_params= dict(zip(fx["vars"], ps)),
                                    tend=83, quiet=True)
        
        try:
            sim.run(saveresults=False)
            scaledPr = sim.k**3/(2*np.pi**2)*sim.Pr
            tres = sim.findallkcrossings(sim.tresult, sim.yresult[:,2], factor=1)[:,0] + nefolds/sim.tstep_wanted
            Pres = scaledPr[tres.astype(int)].diagonal()[0]
            logging.debug("Running model with %s gives scaledPr=%s"%(str(dict(zip(fx["vars"], ps))), str(Pres)))
            if results is not None:
                results = np.vstack((results, np.hstack([ps, Pres])))
            else:
                results = np.hstack([ps, Pres])
            del sim, tres, Pres
        except c.ModelError:
            logging.exception("Error with %s." % str(dict(zip(fx["vars"], ps))))
            del sim
            
    return results

def interpolate_parameter(results, k=None):
    """Interpolate between results to return parameter value at k."""
    if not k:
        k = np.atleast_1d(WMAP7PR)
    else:
        k = np.atleast_1d(k)
    
    #Interpolate results to find more accurate endpoint
    tck = interpolate.splrep(results[:,1], results[:,0])
    y2 = interpolate.splev(k, tck)
    return y2

def run_and_interpolate(fixture, nefolds=5, k=None):
    """ Run the fixture over the parameter values and interpolate to 
    find the closest value which matches WMAP7 power spectrum.
    """
    results = param_vs_spectrum(fixture, nefolds)
    param_value = interpolate_parameter(results, k)
    return param_value
    
    
def param_vs_spectrum_force_tend(fixture, nefolds=5, tend=200):
    """Run tests for a particular parameter, return mass used and spectrum after horizon exit.
    This function forces the end of inflation at a particular time for use with 
    models where inflation doesn't end naturally."""
    #Variable to store results
    results = None
    fx = deepcopy(fixture)
    fxystart = fx["ystart"]
    values_matrix = helpers.cartesian(fx["values"])
    for ps in values_matrix:
        sim = c.FOCanonicalTwoStage(solver="rkdriver_tsix", ystart=fxystart.copy(),
                                    k=fx["pivotk"], potential_func=fx["pot"],
                                    pot_params= dict(zip(fx["vars"], ps)),
                                    tend=tend, quiet=True)
        
        try:
            try:
                sim.runbg()
            except c.ModelError:
                pass
            sim.fotend = np.float64(tend)
            sim.fotendindex = np.int(sim.fotend/sim.tstep_wanted)
            sim.setfoics()
            sim.runfo()
            scaledPr = sim.k**3/(2*np.pi**2)*sim.Pr
            tres = sim.findallkcrossings(sim.tresult, sim.yresult[:,2], factor=1)[:,0] + nefolds/sim.tstep_wanted
            Pres = scaledPr[tres.astype(int)].diagonal()[0]
            print "Running model with %s gives scaledPr=%s"%(str(dict(zip(fx["vars"], ps))), str(Pres))
            if results is not None:
                results = np.vstack((results, np.hstack([ps, Pres])))
            else:
                results = np.hstack([ps, Pres])
            del sim, tres, Pres
        except c.ModelError:
            print "Error with %s." % str(dict(zip(fx["vars"], ps)))
            del sim
            
    
    return results


if __name__ == "__main__":
    print run_and_interpolate(fixtures["mass"].copy())