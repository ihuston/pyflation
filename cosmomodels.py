"""Cosmological Model simulations by Ian Huston
    $Id: cosmomodels.py,v 1.10 2008/04/23 19:03:54 ith Exp $
    
    Provides generic class CosmologicalModel that can be used as a base for explicit models."""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import pylab as P
import rk4
import sys
import os.path
import datetime
import pickle

class ModelError(StandardError):
    """Generic error for model simulating. Attributes include current results stack."""
    
    def __init__(self, expression, tresult=None, yresult=None):
        self.expression = expression
        self.tresult = tresult
        self.yresult = yresult

class CosmologicalModel:
    """Generic class for cosmological model simulations.
    Has no derivs function to pass to ode solver, but does have
    plotting function and initial conditions check."""
    
    solverlist = ["odeint", "rkdriver_dumb"]
    
    def __init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, eps=1.0e-6, dxsav=0.0, solver="odeint"):
        """Initialize model variables, some with default values. Default solver is odeint."""
        
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
        
        self.tresult = None #Will hold last time result
        self.yresult = None #Will hold array of last y results
        self.runcount = 0 #How many times has the model been run?
        self.resultlist = [] #List of all completed results.
        
        
        self.plottitle = "A generic Cosmological Model"
        self.tname = "Time"
        self.ynames = ["First dependent variable"]
        
    def derivs(self, t, yarray):
        """Return an array of derivatives of the dependent variables yarray at timestep t"""
        pass
    
    def run(self):
        """Execute a simulation run using the parameters already provided."""
        
        if self.solver not in self.solverlist:
            raise ModelError("Unknown solver!")
                
        if self.solver == "odeint":
            try:
                self.tresult, self.yresult, self.nok, self.nbad = rk4.odeint(self.ystart, self.tstart,
                    self.tend, self.tstep_wanted, self.tstep_min, self.derivs, self.eps, self.dxsav)
                #Commented out next line to work with array of k values
                #self.yresult = N.hsplit(self.yresult, self.yresult.shape[1])
            except rk4.SimRunError:
                raise
            except StandardError, e:
                #raise ModelError("Error running odeint", self.tresult, self.yresult)
                raise
        
        if self.solver == "rkdriver_dumb":
            #Loosely estimate number of steps based on requested step size
            nstep = N.ceil((self.tend - self.tstart)/self.tstep_wanted)
            try:
                self.tresult, self.yresult = rk4.rkdriver_dumb(self.ystart, self.tstart, self.tend, nstep, self.derivs)
            except StandardError, e:
                #raise ModelError("Error running rkdriver_dumb", self.tresult, self.yresult)
                raise
        
        #Aggregrate results and calling parameters into results list
        callingparams = (self.ystart, self.tstart, self.tend, self.tstep_wanted, self.tstep_min, 
                            self.eps, self.dxsav, self.solver, datetime.datetime.now() )
        
        self.resultlist.append([callingparams, self.tresult, self.yresult])
        
        self.runcount += 1
        
        return
        
    def argstring(self):
        a = r"; Arguments: ystart="+ str(self.ystart) + r", tstart=" + str(self.tstart) 
        a += r", tend=" + str(self.tend) + r", mass=" + str(self.mass)
        return a
    
    def plotresults(self, saveplot = False):
        """Plot results of simulation run on a graph."""
        
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        
        P.plot(self.tresult, self.yresult)
        P.xlabel(self.tname)
        P.ylabel(self.ynames[0])
        P.title(self.plottitle + self.argstring())
        P.show()
        return
    
    def saveallresults(self, filename=None):
        """Tries to save file as a pickled object in directory results."""
        
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if not filename:
            filename = "./results/run" + now
            
        if os.path.isdir(os.path.dirname(filename)):
            if os.path.isfile(filename):
               raise IOError("File already exists!")
        else:
            raise IOError("Directory 'results' does not exist")
        
        try:
            resultsfile = open(filename, "w")
            try:
                pickle.dump(self.resultlist, resultsfile)
            finally:
                resultsfile.close()
        except IOError:
            raise
    
    def loadresults(self, filename):
        """Loads results from a file and appends them to current results list."""
        
        if not os.path.isfile(filename):
            raise IOError("File does not exist!")
        
        try:
            resultsfile = open(filename, "r")
            try:
                newresults = pickle.load(resultsfile)
                self.resultlist.extend(newresults)
            finally:
                resultsfile.close()
        except IOError:
                raise    

class TestModel(CosmologicalModel):
    """Test class defining a very simple function"""
    def __init__(self, ystart=1.0, tstart=0.0, tend=1.0, tstep_wanted=0.01, tstep_min=0.001):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min)
        
        self.plottitle = r"TestModel: $\frac{dy}{dt} = t$"
        self.tname = "Time"
        self.ynames = ["Simple y"]
    
    def derivs(self, t, y):
        """Very simple set of ODEs"""
        return N.cos(t)

class BasicBgModel(CosmologicalModel):
    """Basic model with background equations
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - a : Scale Factor
    """
    
    def __init__(self, ystart=N.array([0.1,0.1,0.1]), tstart=0.0, tend=120.0, tstep_wanted=0.02, tstep_min=0.0001):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min)
        
        #Mass of inflaton in Planck masses
        self.mass = 1.0
        
        self.plottitle = "Basic Cosmological Model"
        self.tname = "Conformal time"
        self.ynames = [r"Inflaton $\phi$", "", r"Scale factor $a$"]
        
    
    def derivs(self, t, y):
        """Basic background equations of motion.
            dydx[0] = dy[0]/d\eta etc"""
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        
        #factor in eom [1/3 a^2 U_0]^{1/2}
        Ufactor = N.sqrt((1.0/3.0)*(y[2]**2)*U)
        
        #Set derivatives
        dydx = N.zeros(3)
        
        #d\phi_0/d\eta = y_1
        dydx[0] = y[1] 
        
        #dy_1/d\eta = -2
        dydx[1] = -2*Ufactor*y[1] - (y[2]**2)*dUdphi
        
        #da/d\eta = [1/3 a^2 U_0]^{1/2}*a
        dydx[2] = Ufactor*y[2]
        
        return dydx
    
    def plotresults(self, saveplot = False):
        """Plot results of simulation run on a graph."""
        
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        
        P.plot(self.tresult, self.yresult[0], self.tresult, self.yresult[2])
        P.xlabel(self.tname)
        P.ylabel("")
        P.legend((self.ynames[0], self.ynames[2]))
        P.title(self.plottitle + self.argstring())
        P.show()
        return

class FirstOrderModel(CosmologicalModel):
    """First order model with background equations
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - a : Scale Factor
    """
    
    def __init__(self, ystart=(N.array([[0.1],[0.1],[0.1],[0.1],[0.1]])*N.ones((5,10))), tstart=0.0, tend=120.0, tstep_wanted=0.02, tstep_min=0.0001):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min)
        
        #Mass of inflaton in Planck masses
        self.mass = 1.0
        
        self.plottitle = "First Order Model"
        self.tname = "Conformal time"
        self.ynames = [r"Inflaton $\phi$", "", r"Scale factor $a$"]
        
        
    
    def derivs(self, t, y):
        """First order equations of motion.
            dydx[0] = dy[0]/d\eta etc"""
        
        #Let k roam for a start
        k = 10**(N.arange(10.0)-8)
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        #2nd deriv
        d2Udphi2 = mass2
        
        #Things we only want to calculate once
        #a^2
        asq = y[4]**2
        
        #factor in eom \mathcal{H} = [1/3 a^2 U_0]^{1/2}
        H = N.sqrt((1.0/3.0)*(asq)*U + 0.5*(y[1]**2))
        
        #Set derivatives
        dydx = N.zeros((5,10))
        
        #d\phi_0/d\eta = y_1
        dydx[0] = y[1] 
        
        #d^2phi/d\eta^2 = -2Hphi^prime -a^2U_,phi
        dydx[1] = -2*H*y[1] - asq*dUdphi
        
        #dy_2/d\eta = \delta\phi_1^prime
        dydx[2] = y[3]
        
        #delta\phi^prime^prime
        dydx[3] = -2*H*y[3] - (k**2)*y[2] - asq*d2Udphi2*y[2]
        
        #da/d\eta = [1/3 a^2 U_0]^{1/2}*a
        dydx[4] = H*y[2]
        
        return dydx
    
    def plotresults(self, saveplot = False):
        """Plot results of simulation run on a graph."""
        
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        
        P.plot(self.tresult, self.yresult[0], self.tresult, self.yresult[2])
        P.xlabel(self.tname)
        P.ylabel("")
        P.legend((self.ynames[0], self.ynames[2]))
        P.title(self.plottitle + self.argstring())
        P.show()
       