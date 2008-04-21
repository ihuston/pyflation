"""Cosmological Model simulations by Ian Huston
    $Id: cosmomodels.py,v 1.1 2008/04/21 15:21:12 ith Exp $
    
    Provides generic class CosmologicalModel that can be used as a base for explicit models."""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import pylab as P
import sys

class CosmologicalModel:
    """Generic class for cosmological model simulations.
    Has no derivs function to pass to ode solver, but does have
    plotting function and initial conditions check."""
    
    solverlist = ["odeint", "rkdriver_dumb"]
    
    def __init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, eps=1.0e-6, dxsav=0.0, solver="odeint"):
        """Initialize model variables, some with default values. Default solver is odeint."""
        if not ystart:
            self.ystart = 0.0
        else:
            self.ystart = ystart
        
        if tstart < tend: 
            self.tstart, self.tend = tstart, tend
        elif tstart==tend:
            raise ValueError, "End time is the same as start time!"
        else:
            raise ValueError, "Ending time is before starting time!"
        
        if tstep_wanted >= tstep_min:
            self.tstep_wanted, self.tstep_min = tstep_wanted, tstep_min
        else:
            raise ValueError, "Desired time step is smaller than specified minimum!"
        
        if eps < 1:
            self.eps = eps
        else:
            raise ValueError, "Not enough accuracy! Change eps < 1."
        
        if dxsav >=0.0:
            self.dxsav = dxsav
        else:
            raise ValueError, "Data saving step must be 0 or positive!"
        
        if solver in self.solverlist:
            self.solver = solver
        else:
            raise ValueError, "Solver not recognized!"
        
        self.tresult = None
        self.yresult = None
        self.modelrun = False
        
        self.plottitle = "A generic Cosmological Model"
        self.tname = "Time"
        self.ynames = ["First dependent variable"]
        
    def derivs(self, t, yarray):
        pass
    
    def run(self):
        if self.solver == "odeint":
            try:
                self.tresult, self.yresult, self.nok, self.nbad = odeint(self.ystart, self.tstart,
                    self.tend, self.tstep_wanted, self.tstep_min, self.derivs, self.eps, self.dxsav)
            except IndexError, e:
                raise ModelError(e.message, self.tresult, self.yresult)
            self.modelrun = True
            return
        
        if self.solver == "rkdriver_dumb":
            #Loosely estimate number of steps based on requested step size
            nstep = ceil((self.tend - self.tstart)/self.tstep_wanted)
            try:
                self.tresult, self.yresult = rkdriver_dumb(self.ystart, self.tstart, self.tend, nstep, self.derivs)
            except Error, e:
                raise ModelError(e.message, self.tresult, self.yresult)
            self.modelrun = True
            return
        
        #Don't know how to handle any other solvers!
        raise ModelError("Unknown solver!")
        
        
    
    def plotresults(self, saveplot = False):
        if not self.modelrun:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        
        P.plot(self.tresult, self.yresult)
        P.xlabel(self.tname)
        P.ylabel(self.ynames[0])
        P.title(self.plottitle)
        P.show()

class TestModel(CosmologicalModel):
    """Test class defining a very simple function"""
    def __init__(self, ystart=1.0, tstart=0.0, tend=1.0, tstep_wanted=0.01, tstep_min=0.001):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min)
        
        self.plottitle = r"TestModel: $\frac{dy}{dt} = t$"
        self.tname = "Time"
        self.ynames = ["Simple y"]
    
    def derivs(self, t, y):
        return t
    

class ModelError(StandardError):
    """Generic error for model simulating. Attributes include current results stack."""
    
    def __init__(self, expression, tresult=None, yresult=None):
        self.expression = expression
        self.tresult = tresult
        self.yresults = yresult
