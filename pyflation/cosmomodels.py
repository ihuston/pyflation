"""Cosmological Model simulations by Ian Huston
    $Id: cosmomodels.py,v 1.233 2010/01/18 16:50:57 ith Exp $
    
    Provides generic class CosmologicalModel that can be used as a base for explicit models."""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0

#system modules
import numpy as np
import os.path
import datetime
from scipy import interpolate
import tables 
import logging

#local modules from pyflation
from configuration import _debug
import cmpotentials
import rk4


#Start logging
root_log_name = logging.getLogger().name
module_logger = logging.getLogger(root_log_name + "." + __name__)

#WMAP pivot scale and Power spectrum
WMAP_PIVOT = 5.25e-60 #WMAP pivot scale in Mpl
WMAP_PR = 2.457e-09 #Power spectrum calculated at the WMAP_PIVOT scale. Real WMAP result quoted as 2.07e-9


class ModelError(StandardError):
    """Generic error for model simulating. Attributes include current results stack."""
    pass

class CosmologicalModel(object):
    """Generic class for cosmological model simulations.
    Has no derivs function to pass to ode solver, but does have
    plotting function and initial conditions check.
    
    Results can be saved in a pickled file as a list of tuples of the following
    structure:
       resultset = (lastparams, tresult, yresult)
       
       lastparams is formatted as in the function callingparams(self) below
    """
    solverlist = ["rkdriver_withks", "rkdriver_new", "rkdriver_tsix"]
    ynames = ["First dependent variable"]
    tname = "Time"
    plottitle = "A generic Cosmological Model"
    
    def __init__(self, ystart=None, simtstart=0.0, tstart=0.0, tstartindex=None, tend=83.0, tstep_wanted=0.01, tstep_min=0.001, eps=1.0e-10,
                 dxsav=0.0, solver="rkdriver_withks", potential_func=None, pot_params=None, **kwargs):
        """Initialize model variables, some with default values. Default solver is odeint."""
        #Start logging
        self._log = logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))
        
        self.ystart = ystart
        self.k = getattr(self, "k", None) #so we can test whether k is set
        
        if tstartindex is None:
            self.tstartindex = np.array([0])
        else:
            self.tstartindex = tstartindex
        
        if np.all(tstart < tend): 
            self.tstart, self.tend = tstart, tend
        elif np.all(tstart==tend):
            raise ValueError, "End time is the same as start time!"
        else:
            raise ValueError, "Ending time is before starting time!"
        
        if np.all(simtstart < tend): 
            self.simtstart = simtstart
        elif simtstart == tend:
            raise ValueError("End time is same a simulation start time!")
        else:
            raise ValueError("End time is before simulation start time!")
        
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
        
        #Change potentials to be right function
        if potential_func is None:
            potential_func = "msqphisq"
        self.potentials = cmpotentials.__getattribute__(potential_func)
        self.potential_func = potential_func
        
        #Set potential parameters to default to empty dictionary.
        if pot_params is None:
            pot_params = {}
        self.pot_params = pot_params
                
        #Set self.pot_params to argument
        if not isinstance(pot_params, dict) and pot_params is not None:
            raise ModelError("Need to provide pot_params as a dictionary of parameters.")
        else:
            self.pot_params = pot_params
        
        self.tresult = None #Will hold last time result
        self.yresult = None #Will hold array of last y results
        self.resultlist = [] #List of all completed results.
        
    def derivs(self, yarray, t):
        """Return an array of derivatives of the dependent variables yarray at timestep t"""
        pass
    
    def potentials(self, y, pot_params=None):
        """Return a 4-tuple of potential, 1st, 2nd and 3rd derivs given y."""
        pass
    
    def findH(self,potential,y):
        """Return value of comoving Hubble variable given potential and y."""
        pass
    
    def run(self, saveresults=True):
        """Execute a simulation run using the parameters already provided."""
        if self.solver not in self.solverlist:
            raise ModelError("Unknown solver!")
        #Test whether k exists and if so change init conditions
               
        if self.solver in ["rkdriver_withks", "rkdriver_new"]:
            #set_trace()
            #Loosely estimate number of steps based on requested step size
            if _debug:
                self._log.debug("Starting simulation with %s.", self.solver)
            solver = rk4.__getattribute__(self.solver)
            try:
                self.tresult, self.yresult = solver(vstart=self.ystart, 
                                                    simtstart=self.simtstart, 
                                                    ts=self.tstart, 
                                                    te=self.tend, 
                                                    allks=self.k, 
                                                    h=self.tstep_wanted, 
                                                    derivs=self.derivs)
            except StandardError:
                self._log.exception("Error running %s!", self.solver)
                raise
            
        if self.solver in ["rkdriver_tsix"]:
            #set_trace()
            #Loosely estimate number of steps based on requested step size
            if not hasattr(self, "tstartindex"):
                raise ModelError("Need to specify initial starting indices!")
            if _debug:
                self._log.debug("Starting simulation with %s.", self.solver)
            solver = rk4.__getattribute__(self.solver)
            try:
                self.tresult, self.yresult = solver(ystart=self.ystart, 
                                                    simtstart=self.simtstart, 
                                                    tsix=self.tstartindex, 
                                                    tend=self.tend, 
                                                    allks=self.k, 
                                                    h=self.tstep_wanted, 
                                                    derivs=self.derivs)
            except StandardError:
                self._log.exception("Error running %s!", self.solver)
                raise
            
        
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        self.resultlist.append([self.lastparams, self.tresult, self.yresult])        
        if saveresults:
            try:
                fname = self.saveallresults()
                self._log.info("Results saved in " + fname)
            except IOError, er:
                self._log.error("Error trying to save results! Results NOT saved.\n" + er)            
        return
    
    def callingparams(self):
        """Returns list of parameters to save with results."""      
        #Form dictionary of inputs
        params = {"tstart":self.tstart,
                  "tend":self.tend,
                  "tstep_wanted":self.tstep_wanted,
                  "tstep_min":self.tstep_min,
                  "eps":self.eps,
                  "dxsav":self.dxsav,
                  "solver":self.solver,
                  "classname":self.__class__.__name__,
                  "datetime":datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                  }
        return params
    
    def gethf5paramsdict(self):
        """Describes the fields required to save the calling parameters."""
        
        params = {
        "solver" : tables.StringCol(50),
        "classname" : tables.StringCol(255),
        "tstart" : tables.Float64Col(np.shape(self.tstart)),
        "simtstart" : tables.Float64Col(),
        "tend" : tables.Float64Col(),
        "tstep_wanted" : tables.Float64Col(),
        "tstep_min" : tables.Float64Col(),
        "eps" : tables.Float64Col(),
        "dxsav" : tables.Float64Col(),
        "datetime" : tables.Float64Col()
        }
        return params
    
    def gethf5yresultdict(self):
        """Return dict describing fields for yresult table of hf5 file."""
        yresdict = {
        "yresult": tables.Float64Col(self.yresult[:,:,0].shape)}
        if self.k is not None:
            yresdict["k"] = tables.Float64Col()
            yresdict["foystart"] = tables.Float64Col(self.foystart[:,0].shape)
            yresdict["fotstart"] = tables.Float64Col()
        return yresdict        
           
    def saveallresults(self, filename=None, filetype="hf5", **kwargs):
        """Tries to save file as a pickled object in directory 'results'."""
        
        now = self.lastparams["datetime"]
        if not filename:
            filename = os.path.join(os.getcwd(), "run" + now + "." + filetype)
            self._log.info("Filename set to " + filename)
            
        if os.path.isdir(os.path.dirname(filename)):
            if os.path.isfile(filename):
                if _debug:
                    self._log.debug("File already exists! Using append data mode.")
                filemode = "a"
            else:
                if _debug:
                    self._log.debug("File does not exist, using write mode.")
                filemode = "w" #Writing to new file
        else:
            raise IOError("Directory 'results' does not exist")
        
        if filetype is "hf5":
            try:
                self.saveresultsinhdf5(filename, filemode, **kwargs)
            except IOError:
                raise
        else:
            raise NotImplementedError("Saving results in format %s is not implemented." % filetype)
        self.lastsavedfile = filename
        return filename
        
    def saveresultsinhdf5(self, filename, filemode, hdf5complevel=2, hdf5complib="blosc"):
        """Save simulation results in a HDF5 format file with filename.
            filename - full path and name of file (should end in hf5 for consistency.
            filemode - ["w"|"a"]: "w" specifies write to a new file, overwriting existing one
                        "a" specifies append to current file or create if does not exist.
            hdf5complevel - Compression level to use with PyTables, default 2.
            hdf5complib - Compression library to use with PyTables, default "blosc".
        """
        #Check whether we should store ks and set group name accordingly
        if self.k is None:
            grpname = "bgresults"
        else:
            grpname = "results" 
        try:
            rf = tables.openFile(filename, filemode)
            try:
                if filemode is "w":
                    #Add compression
                    # Select which compression library to use in configuration
                    filters = tables.Filters(complevel=hdf5complevel, complib=hdf5complib)
                    
                    #Create groups required
                    resgroup = rf.createGroup(rf.root, grpname, "Results of simulation")
                    tresarr = rf.createArray(resgroup, "tresult", self.tresult)
                    paramstab = rf.createTable(resgroup, "parameters", self.gethf5paramsdict(), filters=filters)
                    #Add in potential parameters pot_params as a table
                    potparamsshape = {"name": tables.StringCol(255),
                                      "value": tables.Float64Col()}
                    potparamstab = rf.createTable(resgroup, "pot_params", potparamsshape , filters=filters)
                    
                    #Need to check if results are k dependent
                    if grpname is "results":
                        if hasattr(self, "bgmodel"):
                            #Store bg results:
                            bggrp = rf.createGroup(rf.root, "bgresults", "Background results")
                            bgtrarr = rf.createArray(bggrp, "tresult", self.bgmodel.tresult)
                            bgyarr = rf.createArray(bggrp, "yresult", self.bgmodel.yresult)
                        #Save results
                        yresarr = rf.createEArray(resgroup, "yresult", tables.Float64Atom(), self.yresult[:,:,0:0].shape, filters=filters, expectedrows=8194)# chunkshape=(10,7,10))
                        karr = rf.createEArray(resgroup, "k", tables.Float64Atom(), (0,), filters=filters)
                        if hasattr(self, "foystart"):
                            foystarr = rf.createEArray(resgroup, "foystart", tables.Float64Atom(), self.foystart[:,0:0].shape, filters=filters)
                            fotstarr = rf.createEArray(resgroup, "fotstart", tables.Float64Atom(), (0,), filters=filters)
                            fotsxarr = rf.createEArray(resgroup, "fotstartindex", tables.Float64Atom(), (0,), filters=filters)
                    else:
                        #Only save bg results
                        yresarr = rf.createArray(resgroup, "yresult", self.yresult)
                elif filemode is "a":
                    try:
                        resgroup = rf.getNode(rf.root, grpname)
                        #paramstab = resgroup.parameters
                        yresarr = resgroup.yresult
                        tres = resgroup.tresult[:]
                        if grpname is "results":
                            #Don't need to append bg results, only fo results
                            if hasattr(self, "foystart"):
                                foystarr = resgroup.foystart
                                fotstarr = resgroup.fotstart
                                fotsxarr = resgroup.fotstartindex
                            karr = resgroup.k
                    except tables.NoSuchNodeError:
                        raise IOError("File is not in correct format! Correct results tables do not exist!")
                    if np.shape(tres) != np.shape(self.tresult):
                        raise IOError("Results file has different size of tresult!")
                else:
                    raise IOError("Can only write or append to files!")
                #Now save data
                #Save parameters
                paramstabrow = paramstab.row
                params = self.callingparams()
                for key in params:
                    paramstabrow[key] = params[key]
                paramstabrow.append() #Add to table
                paramstab.flush()
                
                #Save potential parameters
                potparamsrow = potparamstab.row
                for key in self.pot_params:
                    potparamsrow["name"] = key
                    potparamsrow["value"] = self.pot_params[key]
                    potparamsrow.append()
                potparamstab.flush()
                 
                #Save first order results
                if grpname is "results":
                    yresarr.append(self.yresult)
                    karr.append(self.k)
                    if hasattr(self, "foystart"):
                        foystarr.append(self.foystart)
                        fotstarr.append(self.fotstart)
                        fotsxarr.append(self.fotstartindex)
                rf.flush()
                #Log success
                if _debug:
                    self._log.debug("Successfully wrote results to file " + filename)
            finally:
                rf.close()
        except IOError:
            raise
            
class TestModel(CosmologicalModel):
    """Test class defining a very simple function"""
    #Names of variables
    ynames = [r"Simple $y$", r"$\dot{y}$"]
    plottitle = r"TestModel: $\frac{d^2y}{dt^2} = y$"
    tname = "Time"
            
    def __init__(self, ystart=np.array([1.0,1.0]), tstart=0.0, tend=1.0, tstep_wanted=0.01, tstep_min=0.001):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min)
        

    
    def derivs(self, y, t, **kwargs):
        """Very simple set of ODEs"""
        dydx = np.zeros(2)
        
        dydx[0] = y[1]
        dydx[1] = y[0]
        return dydx

class BasicBgModel(CosmologicalModel):
    """Basic model with background equations
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - a : Scale Factor
    """
    #Graph variables
    plottitle = "Basic Cosmological Model"
    tname = "Conformal time"
    ynames = [r"Inflaton $\phi$", "", r"Scale factor $a$"]    
    
    def __init__(self, ystart=np.array([0.1,0.1,0.1]), tstart=0.0, tend=120.0, 
                    tstep_wanted=0.02, tstep_min=0.0001, solver="rkdriver_withks"):
        
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        #Mass of inflaton in Planck masses
        self.mass = 1.0
        
    def potentials(self, y, pot_params=None):
        """Return value of potential at y, along with first and second derivs."""
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        #2nd deriv
        d2Udphi2 = mass2
        #3rd deriv
        d3Udphi3 = 0
        return U, dUdphi, d2Udphi2, d3Udphi3
    
    def derivs(self, y, t, **kwargs):
        """Basic background equations of motion.
            dydx[0] = dy[0]/d\eta etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)[0:3]
        
        #factor in eom [1/3 a^2 U_0]^{1/2}
        Ufactor = np.sqrt((1.0/3.0)*(y[2]**2)*U)
        
        #Set derivatives
        dydx = np.zeros(3)
        
        #d\phi_0/d\eta = y_1
        dydx[0] = y[1] 
        
        #dy_1/d\eta = -2
        dydx[1] = -2*Ufactor*y[1] - (y[2]**2)*dUdphi
        
        #da/d\eta = [1/3 a^2 U_0]^{1/2}*a
        dydx[2] = Ufactor*y[2]
        
        return dydx
    
class PhiModels(CosmologicalModel):
    """Parent class for models implementing the scheme in Malik 06[astro-ph/0610864]"""
    #Graph titles
    plottitle = r"Malik Models in $n$"
    tname = r"E-folds $n$"
    ynames = [r"$\phi$", r"$\dot{\phi}_0$", r"$H$"]
    
    def __init__(self, *args, **kwargs):
        """Call superclass init method."""
        super(PhiModels, self).__init__(*args, **kwargs)
        
    def findH(self, U, y):
        """Return value of Hubble variable, H at y for given potential."""
        phidot = y[1]
        
        #Expression for H
        H = np.sqrt(U/(3.0-0.5*(phidot**2)))
        return H
    
    def potentials(self, y, pot_params=None):
        """Return value of potential at y, along with first and second derivs."""
        pass
    
    def findinflend(self):
        """Find the efold time where inflation ends,
            i.e. the hubble flow parameter epsilon >1.
            Returns tuple of endefold and endindex (in tresult)."""
        
        self.epsilon = self.getepsilon()
        if not any(self.epsilon>1):
            raise ModelError("Inflation did not end during specified number of efoldings. Increase tend and try again!")
        endindex = np.where(self.epsilon>=1)[0][0]
        
        #Interpolate results to find more accurate endpoint
        tck = interpolate.splrep(self.tresult[:endindex], self.epsilon[:endindex])
        t2 = np.linspace(self.tresult[endindex-1], self.tresult[endindex], 100)
        y2 = interpolate.splev(t2, tck)
        endindex2 = np.where(y2>1)[0][0]
        #Return efold of more accurate endpoint
        endefold = t2[endindex2]
        
        return endefold, endindex
    
    def getepsilon(self):
        """Return an array of epsilon = -\dot{H}/H values for each timestep."""
        #Find Hdot
        if len(self.yresult.shape) == 3:
            Hdot = np.array(map(self.derivs, self.yresult, self.tresult))[:,2,0]
            epsilon = - Hdot/self.yresult[:,2,0]
        else:
            Hdot = np.array(map(self.derivs, self.yresult, self.tresult))[:,2]
            epsilon = - Hdot/self.yresult[:,2]
        return epsilon

    
class CanonicalBackground(PhiModels):
    """Basic model with background equations in terms of n
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\n : First deriv of \phi
       y[2] - H: Hubble parameter
    """
    #Titles
    plottitle = r"Background Malik model in $n$"
    tname = r"E-folds $n$"
    ynames = [r"$\phi$", r"$\dot{\phi}_0$", r"$H$"]
        
    def __init__(self,  *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalBackground, self).__init__(*args, **kwargs)
        
        #Set initial H value if None
        if np.all(self.ystart[2] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[2] = self.findH(U, self.ystart)
    
    def derivs(self, y, t, **kwargs):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y, self.pot_params)[0:3]       
        
        #Set derivatives
        dydx = np.zeros_like(y)
        
        #d\phi_0/dn = y_1
        dydx[0] = y[1] 
        
        #dphi^prime/dn
        dydx[1] = -(U*y[1] + dUdphi)/(y[2]**2)
        
        #dH/dn
        dydx[2] = -0.5*(y[1]**2)*y[2]

        return dydx

class CanonicalFirstOrder(PhiModels):
    """First order model using efold as time variable.
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - H : Hubble parameter
       y[3] - \delta\varphi_1 : First order perturbation [Real Part]
       y[4] - \delta\varphi_1^\prime : Derivative of first order perturbation [Real Part]
       y[5] - \delta\varphi_1 : First order perturbation [Imag Part]
       y[6] - \delta\varphi_1^\prime : Derivative of first order perturbation [Imag Part]
       """
       
    #Text for graphs
    plottitle = "Complex First Order Malik Model in Efold time"
    tname = r"$n$"
    ynames = [r"$\varphi_0$",
                    r"$\dot{\varphi_0}$",
                    r"$H$",
                    r"Real $\delta\varphi_1$",
                    r"Real $\dot{\delta\varphi_1}$",
                    r"Imag $\delta\varphi_1$",
                    r"Imag $\dot{\delta\varphi_1}$"]
        
    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalFirstOrder, self).__init__(*args, **kwargs)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start if not given
        if k is None:
            self.k = 10**(np.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = np.array([15.0,-0.1,0.0,1.0,0.0,1.0,0.0])   
        
        #Set initial H value if None
        if np.all(self.ystart[2] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[2] = self.findH(U, self.ystart)
                        
    def derivs(self, y, t, **kwargs):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        #If k not given select all
        if "k" not in kwargs or kwargs["k"] is None:
            k = self.k
        else:
            k = kwargs["k"]
            
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y, self.pot_params)[0:3]        
        
        #Set derivatives taking care of k type
        if type(k) is np.ndarray or type(k) is list: 
            dydx = np.zeros((7,len(k)))
        else:
            dydx = np.zeros(7)
            
        
        #d\phi_0/dn = y_1
        dydx[0] = y[1] 
        
        #dphi^prime/dn
        dydx[1] = -(U*y[1] + dUdphi)/(y[2]**2)
        
        #dH/dn
        dydx[2] = -0.5*(y[1]**2)*y[2]
        
        #d\deltaphi_1/dn = y[4]
        dydx[3] = y[4]
        
        #Get a
        a = self.ainit*np.exp(t)
        
        #d\deltaphi_1^prime/dn  #
        dydx[4] = (-(3 + dydx[2]/y[2])*y[4] - ((k/(a*y[2]))**2)*y[3]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[3]/(y[2]**2)))
                
        #Complex parts
        dydx[5] = y[6]
        
        #
        dydx[6] = (-(3 + dydx[2]/y[2])*y[6]  - ((k/(a*y[2]))**2)*y[5]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[5]/(y[2]**2)))
        
        return dydx
        

class CanonicalSecondOrder(PhiModels):
    """Second order model using efold as time variable.
       y[0] - \delta\varphi_2 : Second order perturbation [Real Part]
       y[1] - \delta\varphi_2^\prime : Derivative of second order perturbation [Real Part]
       y[2] - \delta\varphi_2 : Second order perturbation [Imag Part]
       y[3] - \delta\varphi_2^\prime : Derivative of second order perturbation [Imag Part]
       """
    #Text for graphs
    plottitle = "Complex Second Order Malik Model with source term in Efold time"
    tname = r"$n$"
    ynames = [r"Real $\delta\varphi_2$",
                    r"Real $\dot{\delta\varphi_2}$",
                    r"Imag $\delta\varphi_2$",
                    r"Imag $\dot{\delta\varphi_2}$"]
                    
    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalSecondOrder, self).__init__(*args, **kwargs)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start if not given
        if k is None:
            self.k = 10**(np.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = np.array([0.0,0.0,0.0,0.0])   
                    
    def derivs(self, y, t, **kwargs):
        """Equation of motion for second order perturbations including source term"""
        if _debug:
            self._log.debug("args: %s", str(kwargs))
        #If k not given select all
        if "k" not in kwargs or kwargs["k"] is None:
            k = self.k
            nokix = True
            kix = np.arange(len(k))
        else:
            k = kwargs["k"]
            kix = kwargs["kix"]
        
        if kix is None:
            raise ModelError("Need to specify kix in order to calculate 2nd order perturbation!")
        
        fotix = np.int(np.around((t - self.second_stage.simtstart)/self.second_stage.tstep_wanted))
        
        #debug logging
        if _debug:
            self._log.debug("t=%f, fo.tresult[tix]=%f, fotix=%f", t, self.second_stage.tresult[fotix], fotix)
        #Get first order results for this time step
        if nokix:
            fovars = self.second_stage.yresult[fotix].copy()
            src = self.source[fotix]
        else:
            fovars = self.second_stage.yresult[fotix].copy()[:,kix]
            src = self.source[fotix][kix]
        phi, phidot, H = fovars[0:3]
        epsilon = self.second_stage.bgepsilon[fotix]
        #Get source terms
        
        srcreal, srcimag = src.real, src.imag
        #get potential from function
        U, dU, d2U, d3U = self.potentials(fovars, self.pot_params)[0:4]        
        
        #Set derivatives taking care of k type
        if type(k) is np.ndarray or type(k) is list: 
            dydx = np.zeros((4,len(k)))
        else:
            dydx = np.zeros(4)
            
        #Get a
        a = self.ainit*np.exp(t)
        #Real parts
        #d\deltaphi_2/dn = y[1]
        dydx[0] = y[1]
        
        #d\deltaphi_2^prime/dn  #
        dydx[1] = (-(3 - epsilon)*y[1] - ((k/(a*H))**2)*y[0]
                    -(d2U/H**2 - 3*(phidot**2))*y[0] - srcreal)
                
        #Complex \deltaphi_2
        dydx[2] = y[3]
        
        #Complex derivative
        dydx[3] = (-(3 - epsilon)*y[3] - ((k/(a*H))**2)*y[2]
                    -(d2U/H**2 - 3*(phidot**2))*y[2] - srcimag)
        
        return dydx
        
class CanonicalHomogeneousSecondOrder(PhiModels):
    """Second order homogeneous model using efold as time variable.
       y[0] - \delta\varphi_2 : Second order perturbation [Real Part]
       y[1] - \delta\varphi_2^\prime : Derivative of second order perturbation [Real Part]
       y[2] - \delta\varphi_2 : Second order perturbation [Imag Part]
       y[3] - \delta\varphi_2^\prime : Derivative of second order perturbation [Imag Part]
       """
    #Text for graphs
    plottitle = "Complex Homogeneous Second Order Model with source term in Efold time"
    tname = r"$n$"
    ynames = [r"Real $\delta\varphi_2$",
                    r"Real $\dot{\delta\varphi_2}$",
                    r"Imag $\delta\varphi_2$",
                    r"Imag $\dot{\delta\varphi_2}$"]
                    
    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalHomogeneousSecondOrder, self).__init__(*args, **kwargs)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start if not given
        if k is None:
            self.k = 10**(np.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = np.array([0.0,0.0,0.0,0.0])   
                    
    def derivs(self, y, t, **kwargs):
        """Equation of motion for second order perturbations including source term"""
        if _debug:
            self._log.debug("args: %s", str(kwargs))
        #If k not given select all
        if "k" not in kwargs or kwargs["k"] is None:
            k = self.k
            kix = np.arange(len(k))
        else:
            k = kwargs["k"]
            kix = kwargs["kix"]
        
        if kix is None:
            raise ModelError("Need to specify kix in order to calculate 2nd order perturbation!")
        #Need t index to use first order data
        if kwargs["tix"] is None:
            raise ModelError("Need to specify tix in order to calculate 2nd order perturbation!")
        else:
            tix = kwargs["tix"]
        #debug logging
        if _debug:
            self._log.debug("tix=%f, t=%f, fo.tresult[tix]=%f", tix, t, self.second_stage.tresult[tix])
        #Get first order results for this time step
        fovars = self.second_stage.yresult[tix].copy()[:,kix]
        phi, phidot, H = fovars[0:3]
        epsilon = self.second_stage.bgepsilon[tix]
        #Get source terms
#        src = self.source[tix][kix]
#        srcreal, srcimag = src.real, src.imag
        #get potential from function
        U, dU, d2U, d3U = self.potentials(fovars, self.pot_params)[0:4]        
        
        #Set derivatives taking care of k type
        if type(k) is np.ndarray or type(k) is list: 
            dydx = np.zeros((4,len(k)))
        else:
            dydx = np.zeros(4)
            
        #Get a
        a = self.ainit*np.exp(t)
        #Real parts
        #d\deltaphi_2/dn = y[1]
        dydx[0] = y[1]
        
        #d\deltaphi_2^prime/dn  #
        dydx[1] = (-(3 - epsilon)*y[1] - ((k/(a*H))**2)*y[0]
                    -(d2U/H**2 - 3*(phidot**2))*y[0] )
                
        #Complex \deltaphi_2
        dydx[2] = y[3]
        
        #Complex derivative
        dydx[3] = (-(3 - epsilon)*y[3] - ((k/(a*H))**2)*y[2]
                    -(d2U/H**2 - 3*(phidot**2))*y[2] )
        
        return dydx
        
class CanonicalRampedSecondOrder(PhiModels):
    """Second order model using efold as time variable.
       y[0] - \delta\varphi_2 : Second order perturbation [Real Part]
       y[1] - \delta\varphi_2^\prime : Derivative of second order perturbation [Real Part]
       y[2] - \delta\varphi_2 : Second order perturbation [Imag Part]
       y[3] - \delta\varphi_2^\prime : Derivative of second order perturbation [Imag Part]
       """
    #Text for graphs
    plottitle = "Complex Second Order Malik Model with source term in Efold time"
    tname = r"$n$"
    ynames = [r"Real $\delta\varphi_2$",
                    r"Real $\dot{\delta\varphi_2}$",
                    r"Imag $\delta\varphi_2$",
                    r"Imag $\dot{\delta\varphi_2}$"]
                    
    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalRampedSecondOrder, self).__init__(*args, **kwargs)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start if not given
        if k is None:
            self.k = 10**(np.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = np.array([0.0,0.0,0.0,0.0])   
            
        #Ramp arguments in form
        # Ramp = (tanh(a*(t - t_ic - b) + c))/d if abs(t-t_ic+b) < e
        if "rampargs" not in kwargs:
            self.rampargs = {"a": 15.0,
                        "b": 0.3,
                        "c": 1,
                        "d": 2, 
                        "e": 1}
        else:
            self.rampargs = kwargs["rampargs"]
                    
    def derivs(self, y, t, **kwargs):
        """Equation of motion for second order perturbations including source term"""
        if _debug:
            self._log.debug("args: %s", str(kwargs))
        #If k not given select all
        if "k" not in kwargs or kwargs["k"] is None:
            k = self.k
            nokix = True
            kix = np.arange(len(k))
        else:
            k = kwargs["k"]
            kix = kwargs["kix"]
        
        if kix is None:
            raise ModelError("Need to specify kix in order to calculate 2nd order perturbation!")
        
        fotix = np.int(np.around((t - self.second_stage.simtstart)/self.second_stage.tstep_wanted))
        
        #debug logging
        if _debug:
            self._log.debug("t=%f, fo.tresult[tix]=%f, fotix=%f", t, self.second_stage.tresult[fotix], fotix)
        #Get first order results for this time step
        if nokix:
            fovars = self.second_stage.yresult[fotix].copy()
            src = self.source[fotix]
        else:
            fovars = self.second_stage.yresult[fotix].copy()[:,kix]
            src = self.source[fotix][kix]
        phi, phidot, H = fovars[0:3]
        epsilon = self.second_stage.bgepsilon[fotix]
        
        #Get source terms and multiply by ramp
        if nokix:
            tanharg = t - self.tstart - self.rampargs["b"]
        else:
            tanharg =  t-self.tstart[kix] - self.rampargs["b"]
                
        #When absolute value of tanharg is less than e then multiply source by ramp for those values.
        if np.any(abs(tanharg)<self.rampargs["e"]):
            #Calculate the ramp
            ramp = (np.tanh(self.rampargs["a"]*tanharg) + self.rampargs["c"])/self.rampargs["d"]
            #Get the second order timestep value
            sotix = t / self.tstep_wanted
            #Compare with tstartindex values. Set the ramp to zero for any that are equal
            ramp[self.tstartindex==sotix] = 0
            #Scale the source term by the ramp value.
            if _debug:
                self._log.debug("Ramp value set as %s.", ramp[abs(tanharg<self.rampargs["e"])])
            src[abs(tanharg)<self.rampargs["e"]] = ramp*src
        
        #Split source into real and imaginary parts.
        srcreal, srcimag = src.real, src.imag
        #get potential from function
        U, dU, d2U, d3U = self.potentials(fovars, self.pot_params)[0:4]        
        
        #Set derivatives taking care of k type
        if type(k) is np.ndarray or type(k) is list: 
            dydx = np.zeros((4,len(k)))
        else:
            dydx = np.zeros(4)
            
        #Get a
        a = self.ainit*np.exp(t)
        #Real parts
        #d\deltaphi_2/dn = y[1]
        dydx[0] = y[1]
        
        #d\deltaphi_2^prime/dn  #
        dydx[1] = (-(3 - epsilon)*y[1] - ((k/(a*H))**2)*y[0]
                    -(d2U/H**2 - 3*(phidot**2))*y[0] - srcreal)
                
        #Complex \deltaphi_2
        dydx[2] = y[3]
        
        #Complex derivative
        dydx[3] = (-(3 - epsilon)*y[3] - ((k/(a*H))**2)*y[2]
                    -(d2U/H**2 - 3*(phidot**2))*y[2] - srcimag)
        
        return dydx
        
class MultiStageModel(CosmologicalModel):
    """Parent of all multi (2 or 3) stage models. Contains methods to determine ns, k crossing and outlines
    methods to find Pr that are implemented in children."""
    
    def __init__(self, *args, **kwargs):
        """Initialize super class instance."""
        super(MultiStageModel, self).__init__(*args, **kwargs)
        #Set constant factor for 1st order initial conditions
        if "cq" in kwargs:
            self.cq = kwargs["cq"]
        else:
            self.cq = 50 #Default value as in Salopek et al.
        
        
    def finda_end(self, Hend, Hreh=None):
        """Given the Hubble parameter at the end of inflation and at the end of reheating
            calculate the scale factor at the end of inflation."""
        if Hreh is None:
            Hreh = Hend #Instantaneous reheating
        a_0 = 1 # Normalize today
        a_end = a_0*np.exp(-72.3)*((Hreh/(Hend**4.0))**(1.0/6.0))
        #a_end = a_0*np.exp(-71.49)*((Hreh/(Hend**4.0))**(1.0/6.0))
        return a_end
        
    def findkcrossing(self, k, t, H, factor=None):
        """Given k, time variable and Hubble parameter, find when mode k crosses the horizon."""
        #threshold
        err = 1.0e-26
        if factor is None:
            factor = self.cq #time before horizon crossing
        #get aHs
        if len(H.shape) > 1:
            #Use only one dimensional H
            H = H[:,0]
        aH = self.ainit*np.exp(t)*H
        try:
            kcrindex = np.where(np.sign(k - (factor*aH))<0)[0][0]
        except IndexError, ex:
            raise ModelError("k mode " + str(k) + " crosses horizon after end of inflation!")
        kcrefold = t[kcrindex]
        return kcrindex, kcrefold
    
    def findallkcrossings(self, t, H):
        """Iterate over findkcrossing to get full list"""
        return np.array([self.findkcrossing(onek, t, H) for onek in self.k])
    
    def findHorizoncrossings(self, factor=1):
        """FInd horizon crossing for all ks"""
        return np.array([self.findkcrossing(onek, self.tresult, oneH, factor) for onek, oneH in zip(self.k, np.rollaxis(self.yresult[:,2,:], -1,0))])
    
    def getfoystart(self):
        """Return model dependent setting of ystart""" 
        pass

    def getdeltaphi(self):
        """Return \delta\phi_1 no matter what variable is used for simulation. Implemented model-by-model."""
        pass
    
    def findPr(self):
        """Return the spectrum of curvature perturbations P_R for each k.Implemented model-by-model."""
        pass
    
    def findPgrav(self):
        """Return the spectrum of tensor perturbations P_grav for each k. Implemented model-by-model."""
        pass
    
    def findPphi(self):
        """Return the spectrum of scalar perturbations P_phi for each k."""
        pass
     
    def findns(self, k=None, nefolds=3):
        """Return the value of n_s at the specified k mode."""
        
        #If k is not defined, get value at all self.k
        if k is None:
            k = self.k
        else:
            if k<self.k.min() and k>self.k.max():
                self._log.warn("Warning: Extrapolating to k value outside those used in spline!")
        
        ts = self.findHorizoncrossings(factor=1)[:,0] + nefolds/self.tstep_wanted #About nefolds after horizon exit
        xp = np.log(self.Pr[ts.astype(int)].diagonal())
        lnk = np.log(k)
        
        #Need to sort into ascending k
        sortix = lnk.argsort()
                
        #Use cubic splines to find deriv
        tck = interpolate.splrep(lnk[sortix], xp[sortix])
        ders = interpolate.splev(lnk[sortix], tck, der=1)
        
        ns = 1 + ders
        #Unorder the ks again
        nsunsort = np.zeros(len(ns))
        nsunsort[sortix] = ns
        
        return nsunsort
    
    def callingparams(self):
        """Returns list of parameters to save with results."""
        #Test whether k has been set
        try:
            self.k
        except (NameError, AttributeError):
            self.k=None
        #Form dictionary of inputs
        params = {"tstart":self.tstart,
                  "ainit":self.ainit,
                  "potential_func":self.potentials.__name__,
                  "tend":self.tend,
                  "tstep_wanted":self.tstep_wanted,
                  "tstep_min":self.tstep_min,
                  "eps":self.eps,
                  "dxsav":self.dxsav,
                  "solver":self.solver,
                  "classname":self.__class__.__name__,
                  "datetime":datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                  }
        return params
    
    def gethf5paramsdict(self):
        """Describes the fields required to save the calling parameters."""
        params = {
        "solver" : tables.StringCol(50),
        "classname" : tables.StringCol(255),
        "tstart" : tables.Float64Col(np.shape(self.tstart)),
        "simtstart" : tables.Float64Col(),
        "ainit" : tables.Float64Col(),
        "potential_func" : tables.StringCol(255),
        "tend" : tables.Float64Col(),
        "tstep_wanted" : tables.Float64Col(),
        "tstep_min" : tables.Float64Col(),
        "eps" : tables.Float64Col(),
        "dxsav" : tables.Float64Col(),
        "datetime" : tables.Float64Col()
        }
        return params
    
    
class CanonicalMultiStage(MultiStageModel):
    """Implementation of generic two stage model with standard initial conditions for phi.
    """
                    
    def __init__(self, *args, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
        #Call superclass
        super(CanonicalMultiStage, self).__init__(*args, **kwargs)
    
    @property
    def Pphi(self, recompute=False):
        """Return the spectrum of scalar perturbations P_phi for each k.
        
        This is the unscaled version $P_{\phi}$ which is related to the scaled version by
        $\mathcal{P}_{\phi} = k^3/(2pi^2) P_{\phi}$. Note that result is stored as the
        instance variable self.Pphi. 
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should value be recomputed even if already stored? Default is False.
        
        Returns
        -------
        Pphi: array_like
              Array of Pphi values for all timesteps and k modes
        """
        #Basic caching of result
        if not hasattr(self, "_Pphi") or recompute:        
            deltaphi = self.deltaphi
            self._Pphi = deltaphi*deltaphi.conj()
        return self._Pphi
    
    @property            
    def Pr(self, recompute=False):
        """Return the spectrum of curvature perturbations $P_R$ for each k.
        
        This is the unscaled version $P_R$ which is related to the scaled version by
        $\mathcal{P}_R = k^3/(2pi^2) P_R$. Note that result is stored as the instance variable
        self.Pr. 
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should value be recomputed even if already stored? Default is False.
                   
        Returns
        -------
        Pr: array_like
            Array of Pr values for all timesteps and k modes
        """
        #Basic caching of result
        if not hasattr(self, "_Pr") or recompute:        
            Pphi = self.Pphi
            phidot = self.yresult[:,1,:] #bg phidot
            self._Pr = Pphi/(phidot**2) #change if bg evol is different
        return self._Pr
    
    def findPgrav(self, recompute=False):
        """Return the spectrum of tensor perturbations $P_grav$ for each k.
        
        Note that result is stored as the instance variable self.Pgrav. 
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should value be recomputed even if already stored? Default is False.
                   
        Returns
        -------
        Pgrav: array_like
               Array of Pgrav values for all timesteps and k modes
        """
        #Basic caching of result
        if not hasattr(self, "Pgrav") or recompute:        
            Pphi = self.findPphi()
            self.Pgrav = 2*Pphi
        return self.Pgrav
    
    def getzeta(self):
        """Return the curvature perturbation on uniform-density hypersurfaces zeta."""
        #Get needed variables
        phidot = self.yresult[:,1,:]
        a = self.ainit*np.exp(self.tresult)
        H = self.yresult[:,2,:]
        dUdphi = self.firstordermodel.potentials(self.yresult[:,0,:][np.newaxis,:], self.pot_params)[1]
        deltaphi = self.yresult[:,3,:] + self.yresult[:,5,:]*1j
        deltaphidot = self.yresult[:,4,:] + self.yresult[:,6,:]*1j
        
        deltarho = H**2*(phidot*deltaphidot - phidot**3*deltaphidot) + dUdphi*deltaphi
        drhodt = (H**3)*(phidot**2)*(-1/a[:,np.newaxis]**2 - 2) -H*phidot*dUdphi
        
        zeta = -H*deltarho/drhodt
        return zeta, deltarho, drhodt
        
    def findzetasq(self):
        """Return the spectrum of zeta."""
        pass
    
                                        
class TwoStageModel(MultiStageModel):
    """Uses a background and firstorder class to run a full (first-order) simulation.
        Main additional functionality is in determining initial conditions.
        Variables finally stored are as in first order class.
    """                
    def __init__(self, ystart=None, tstart=0.0, tstartindex=None, tend=83.0, tstep_wanted=0.01, tstep_min=0.0001, 
                 k=None, ainit=None, solver="rkdriver_withks", bgclass=None, foclass=None, 
                 potential_func=None, pot_params=None, simtstart=0, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
      
        #Initial conditions for each of the variables.
        if ystart is None:
            #Initial conditions for all variables
            self.ystart = np.array([18.0, # \phi_0
                                   -0.1, # \dot{\phi_0}
                                    0.0, # H - leave as 0.0 to let program determine
                                    1.0, # Re\delta\phi_1
                                    0.0, # Re\dot{\delta\phi_1}
                                    1.0, # Im\delta\phi_1
                                    0.0  # Im\dot{\delta\phi_1}
                                    ])
        else:
            self.ystart = ystart
        if not tstartindex:
            self.tstartindex = np.array([0])
        else:
            self.tstartindex = tstartindex
        #Call superclass
        newkwargs = dict(ystart=self.ystart, 
                         tstart=tstart,
                         tstartindex=self.tstartindex, 
                         tend=tend, 
                         tstep_wanted=tstep_wanted, 
                         tstep_min=tstep_min, 
                         solver=solver, 
                         potential_func=potential_func, 
                         pot_params=pot_params, 
                         **kwargs)
        
        super(TwoStageModel, self).__init__(**newkwargs)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
                
        #Let k roam if we don't know correct ks
        if k is None:
            self.k = 10**(np.arange(7.0)-62)
        else:
            self.k = k
        self.simtstart = simtstart
            
        if bgclass is None:
            self.bgclass = CanonicalBackground
        else:
            self.bgclass = bgclass
        if foclass is None:
            self.foclass = CanonicalFirstOrder
        else:
            self.foclass = foclass
        
        #Setup model variables    
        self.bgmodel = self.firstordermodel = None
                    
    def setfoics(self):
        """After a bg run has completed, set the initial conditions for the 
            first order run."""
        #debug
        #set_trace()
        
        #Find initial conditions for 1st order model
        #Find a_end using instantaneous reheating
        #Need to change to find using splines
        Hend = self.bgmodel.yresult[self.fotendindex,2]
        self.a_end = self.finda_end(Hend)
        self.ainit = self.a_end*np.exp(-self.bgmodel.tresult[self.fotendindex])
        
        
        #Find epsilon from bg model
        try:
            self.bgepsilon
        except AttributeError:            
            self.bgepsilon = self.bgmodel.getepsilon()
        #Set etainit, initial eta at n=0
        self.etainit = -1/(self.ainit*self.bgmodel.yresult[0,2]*(1-self.bgepsilon[0]))
        
        #find k crossing indices
        kcrossings = self.findallkcrossings(self.bgmodel.tresult[:self.fotendindex], 
                            self.bgmodel.yresult[:self.fotendindex,2])
        kcrossefolds = kcrossings[:,1]
                
        #If mode crosses horizon before t=0 then we will not be able to propagate it
        if any(kcrossefolds==0):
            raise ModelError("Some k modes crossed horizon before simulation began and cannot be initialized!")
        
        #Find new start time from earliest kcrossing
        self.fotstart, self.fotstartindex = kcrossefolds, kcrossings[:,0].astype(np.int)
        self.foystart = self.getfoystart()
        return  
        
    def runbg(self):
        """Run bg model after setting initial conditions."""

        #Check ystart is in right form (1-d array of three values)
        if len(self.ystart.shape) == 1:
            ys = self.ystart[0:3]
        elif len(self.ystart.shape) == 2:
            ys = self.ystart[0:3,0]
        #Choose tstartindex to be simply the first timestep.
        tstartindex = np.array([0])
        
        kwargs = dict(ystart=ys, 
                      tstart=self.tstart,
                      tstartindex=tstartindex, 
                      tend=self.tend,
                      tstep_wanted=self.tstep_wanted, 
                      tstep_min=self.tstep_min, 
                      solver=self.solver,
                      potential_func=self.potential_func, 
                      pot_params=self.pot_params)
         
        self.bgmodel = self.bgclass(**kwargs)
        #Start background run
        self._log.info("Running background model...")
        try:
            self.bgmodel.run(saveresults=False)
        except ModelError, er:
            self._log.exception("Error in background run, aborting!")
        #Find end of inflation
        self.fotend, self.fotendindex = self.bgmodel.findinflend()
        self._log.info("Background run complete, inflation ended " + str(self.fotend) + " efoldings after start.")
        return
        
    def runfo(self):
        """Run first order model after setting initial conditions."""

        #Initialize first order model
        kwargs = dict(ystart=self.foystart, 
                      tstart=self.fotstart,
                      simtstart=self.simtstart, 
                      tstartindex = self.fotstartindex, 
                      tend=self.fotend, 
                      tstep_wanted=self.tstep_wanted, 
                      tstep_min=self.tstep_min, 
                      solver=self.solver,
                      k=self.k, 
                      ainit=self.ainit, 
                      potential_func=self.potential_func, 
                      pot_params=self.pot_params)
        
        self.firstordermodel = self.foclass(**kwargs)
        #Set names as in ComplexModel
        self.tname, self.ynames = self.firstordermodel.tname, self.firstordermodel.ynames
        #Start first order run
        self._log.info("Beginning first order run...")
        try:
            self.firstordermodel.run(saveresults=False)
        except ModelError, er:
            raise ModelError("Error in first order run, aborting! Message: " + er.message)
        
        #Set results to current object
        self.tresult, self.yresult = self.firstordermodel.tresult, self.firstordermodel.yresult
        return
    
    def run(self, saveresults=True):
        """Run the full model.
        
        The background model is first run to establish the end time of inflation and the start
        times for the k modes. Then the initial conditions are set for the first order variables.
        Finally the first order model is run and the results are saved if required.
        
        Parameters
        ----------
        saveresults: boolean, optional
                     Should results be saved at the end of the run. Default is False.
                     
        Returns
        -------
        filename: string
                  name of the results file if any
        """
        #Run bg model
        self.runbg()
        
        #Set initial conditions for first order model
        self.setfoics()
        
        #Run first order model
        self.runfo()
        
        #Save results in resultlist and file
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        
        self.resultlist.append([self.lastparams, self.tresult, self.yresult])        
        
        if saveresults:
            try:
                self._log.info("Results saved in " + self.saveallresults())
            except IOError, er:
                self._log.exception("Error trying to save results! Results NOT saved.")        
        return
    

class FOCanonicalTwoStage(CanonicalMultiStage, TwoStageModel):
    """Implementation of First Order Canonical two stage model with standard initial conditions for phi.
    """
    #Text for graphs
    plottitle = "FOCanonicalTwoStage Model in Efold Time"
    tname = r"$n$" 
    ynames = [r"$\varphi_0$",
                    r"$\dot{\varphi_0}$",
                    r"$H$",
                    r"Real $\delta\varphi_1$",
                    r"Real $\dot{\delta\varphi_1}$",
                    r"Imag $\delta\varphi_1$",
                    r"Imag $\dot{\delta\varphi_1}$"]
                                                  
    def __init__(self, *args, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
        #Call superclass
        super(FOCanonicalTwoStage, self).__init__(*args, **kwargs)
        
    def getfoystart(self, ts=None, tsix=None):
        """Model dependent setting of ystart"""
        if _debug:
            self._log.debug("Executing getfoystart to get initial conditions.")
        #Set variables in standard case:
        if ts is None or tsix is None:
            ts, tsix = self.fotstart, self.fotstartindex
            
        #Reset starting conditions at new time
        foystart = np.zeros((len(self.ystart), len(self.k)))
        #set_trace()
        #Get values of needed variables at crossing time.
        astar = self.ainit*np.exp(ts)
        
        #Truncate bgmodel yresult down if there is an extra dimension
        if len(self.bgmodel.yresult.shape) > 2:
            bgyresult = self.bgmodel.yresult[..., 0]
        else:
            bgyresult = self.bgmodel.yresult
            
        Hstar = bgyresult[tsix,2]
        Hzero = bgyresult[0,2]
        
        epsstar = self.bgepsilon[tsix]
        etastar = -1/(astar*Hstar*(1-epsstar))
        try:
            etadiff = etastar - self.etainit
        except AttributeError:
            etadiff = etastar + 1/(self.ainit*Hzero*(1-self.bgepsilon[0]))
        keta = self.k*etadiff
        
        #Set bg init conditions based on previous bg evolution
        try:
            foystart[0:3] = bgyresult[tsix,:].transpose()
        except ValueError:
            foystart[0:3] = bgyresult[tsix,:][:, np.newaxis]
        
        #Find 1/asqrt(2k)
        arootk = 1/(astar*(np.sqrt(2*self.k)))
        #Find cos and sin(-keta)
        csketa = np.cos(-keta)
        snketa = np.sin(-keta)
        
        #Set Re\delta\phi_1 initial condition
        foystart[3,:] = csketa*arootk
        #set Re\dot\delta\phi_1 ic
        foystart[4,:] = -arootk*(csketa - (self.k/(astar*Hstar))*snketa)
        #Set Im\delta\phi_1
        foystart[5,:] = snketa*arootk
        #Set Im\dot\delta\phi_1
        foystart[6,:] = -arootk*((self.k/(astar*Hstar))*csketa + snketa)
        
        return foystart
    
    def getdeltaphi(self):
        return self.deltaphi
    
    @property
    def deltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi: array_like
                  Array of $\delta\phi$ values for all timesteps and k modes.
        """
        
        if not hasattr(self, "deltaphi") or recompute:
            self._dp = self.yresult[:,3,:] + self.yresult[:,5,:]*1j
        return self._dp
        
class FONewCanonicalTwoStage(FOCanonicalTwoStage):
    """Implementation of First Order Canonical two stage model with standard initial conditions for phi.
    """
                    
    def __init__(self, *args, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
        #Call superclass
        super(FONewCanonicalTwoStage, self).__init__(*args, **kwargs)
        
    def getfoystart(self, ts=None, tsix=None):
        """Model dependent setting of ystart"""
        if _debug:
            self._log.debug("Executing getfoystart to get initial conditions.")
        #Set variables in standard case:
        if ts is None or tsix is None:
            ts, tsix = self.fotstart, self.fotstartindex
            
        #Reset starting conditions at new time
        foystart = np.zeros((len(self.ystart), len(self.k)))
        #set_trace()
        #Get values of needed variables at crossing time.
        astar = self.ainit*np.exp(ts)
        
        #Truncate bgmodel yresult down if there is an extra dimension
        if len(self.bgmodel.yresult.shape) > 2:
            bgyresult = self.bgmodel.yresult[..., 0]
        else:
            bgyresult = self.bgmodel.yresult
            
        Hstar = bgyresult[tsix,2]
        epsstar = self.bgepsilon[tsix]
        etastar = -1/(astar*Hstar*(1-epsstar))
        try:
            etadiff = etastar - self.etainit
        except AttributeError:
            etadiff = etastar + 1/(self.ainit*bgyresult[0,2]*(1-self.bgepsilon[0]))
        keta = self.k*etadiff
        
        #Set bg init conditions based on previous bg evolution
        try:
            foystart[0:3] = bgyresult[tsix,:].transpose()
        except ValueError:
            foystart[0:3] = bgyresult[tsix,:][:, np.newaxis]
        
        #Find 1/asqrt(2k)
        arootk = np.sqrt(self.k**3/(2*np.pi**2))/(astar*(np.sqrt(2*self.k)))
        #Find cos and sin(-keta)
        csketa = np.cos(-keta)
        snketa = np.sin(-keta)
        
        #Set Re\delta\phi_1 initial condition
        foystart[3,:] = csketa*arootk
        #set Re\dot\delta\phi_1 ic
        foystart[4,:] = -arootk*(csketa - (self.k/(astar*Hstar))*snketa)
        #Set Im\delta\phi_1
        foystart[5,:] = snketa*arootk
        #Set Im\dot\delta\phi_1
        foystart[6,:] = -arootk*((self.k/(astar*Hstar))*csketa + snketa)
        
        return foystart
    
    def getdeltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi: array_like
                  Array of $\delta\phi$ values for all timesteps and k modes.
        """
        
        if not hasattr(self, "deltaphi") or recompute:
            self.deltaphi = (2*np.pi**2)/(self.k**3) * (self.yresult[:,3,:] + self.yresult[:,5,:]*1j) #complex deltaphi
        return self.deltaphi
                
    
def make_wrapper_model(modelfile, *args, **kwargs):
    """Return a wrapper class that provides the given model class from a file."""
    #Check file exists
    if not os.path.isfile(modelfile):
        raise IOError("File does not exist!")
    try:
        rf = tables.openFile(modelfile, "r")
        try:
            try:
                params = rf.root.results.parameters
                modelclassname = params[0]["classname"]
            except tables.NoSuchNodeError:
                raise ModelError("File does not contain correct model data structure!")
        finally:
            rf.close()
    except IOError:
        raise
    try:
        modelclass = globals()[modelclassname]
    except AttributeError:
        raise ModelError("Model class does not exist!")
                
    class ModelWrapper(modelclass):
        """Wraps first order model using HDF5 file of results."""
        
        def __init__(self, filename, *args, **kwargs):
            """Get results from file and instantiate variables.
            Opens file with handle saved as self._rf. File is closed in __del__"""
            #Call super class __init__ method
            super(ModelWrapper, self).__init__(*args, **kwargs)
            
            #Check file exists
            if not os.path.isfile(filename):
                raise IOError("File does not exist!")
            try:
                if _debug:
                    self._log.debug("Opening file " + filename + " to read results.")
                try:
                    self._rf = tables.openFile(filename, "r")
                    self.yresult = self._rf.root.results.yresult
                    self.tresult = self._rf.root.results.tresult
                    self.fotstart = self._rf.root.results.fotstart
                    if "fotstartindex" in self._rf.root.results:
                        #for backwards compatability only set if it exists
                        self.fotstartindex = self._rf.root.results.fotstartindex
                    self.foystart = self._rf.root.results.foystart
                    self.k = self._rf.root.results.k[:]
                    params = self._rf.root.results.parameters
                except tables.NoSuchNodeError:
                    raise ModelError("File does not contain correct model data structure!")
                try:
                    self.source = self._rf.root.results.sourceterm
                except tables.NoSuchNodeError:
                    if _debug:
                        self._log.debug("First order file does not have a source term.")
                    self.source = None
                # Put potential parameters into right variable
                try:
                    potparamstab = self._rf.root.results.pot_params
                    for row in potparamstab:
                        key = row["name"]
                        val = row["value"]
                        self.pot_params[key] = val
                except tables.NoSuchNodeError:
                    if _debug:
                        self._log.debug("No pot_params table in file.")                
                
                #Put params in right slots
                for ix, val in enumerate(params[0]):
                    self.__setattr__(params.colnames[ix], val)
                #set correct potential function (only works with cmpotentials currently)
                self.potentials = cmpotentials.__getattribute__(self.potential_func)
            except IOError:
                raise
            
            #Fix bgmodel to actual instance
            if self.ystart is not None:
                #Check ystart is in right form (1-d array of three values)
                if len(self.ystart.shape) == 1:
                    ys = self.ystart[0:3]
                elif len(self.ystart.shape) == 2:
                    ys = self.ystart[0:3,0]
            else:
                ys = self.foystart[0:3,0]
            self.bgmodel = self.bgclass(ystart=ys, tstart=self.tstart, tend=self.tend, 
                            tstep_wanted=self.tstep_wanted, tstep_min=self.tstep_min, solver=self.solver,
                            potential_func=self.potential_func, pot_params=self.pot_params)
            #Put in data
            try:
                if _debug:
                    self._log.debug("Trying to get background results...")
                self.bgmodel.tresult = self._rf.root.bgresults.tresult[:]
                self.bgmodel.yresult = self._rf.root.bgresults.yresult
            except tables.NoSuchNodeError:
                raise ModelError("File does not contain background results!")
            #Get epsilon
            if _debug:
                self._log.debug("Calculating self.bgepsilon...")
            self.bgepsilon = self.bgmodel.getepsilon()
            #Success
            self._log.info("Successfully imported data from file into model instance.")
        
        def __del__(self):
            """Close file when object destroyed."""
            try:
                if _debug:
                    self._log.debug("Trying to close file...")
                self._rf.close()
            except IOError:
                raise
    return ModelWrapper(modelfile, *args, **kwargs)

class ThirdStageModel(MultiStageModel):
    """Runs third stage calculation (typically second order perturbations) using
    a two stage model instance which could be wrapped from a file."""

    def __init__(self, second_stage, soclass=None, ystart=None, **soclassargs):
        """Initialize variables and check that tsmodel exists and is correct form."""
        
        #Test whether tsmodel is of correct type
        if not isinstance(second_stage, TwoStageModel):
            raise ModelError("Need to provide a TwoStageModel instance to get first order results from!")
        else:
            self.second_stage = second_stage
            #Set properties to be those of second stage model
            self.k = np.copy(self.second_stage.k)
            self.simtstart = self.second_stage.tresult[0]
            self.fotstart = np.copy(self.second_stage.fotstart)
            self.fotstartindex = np.copy(self.second_stage.fotstartindex)
            self.ainit = self.second_stage.ainit
            self.potentials = self.second_stage.potentials
            self.potential_func = self.second_stage.potential_func
            self.pot_params = self.second_stage.pot_params
        
        if ystart is None:
            ystart = np.zeros((4, len(self.k)))
            
        #Need to make sure that the tstartindex terms are changed over to new timestep.
        fotstep = self.second_stage.tstep_wanted
        sotstep = fotstep*2
        sotstartindex = np.around(self.fotstartindex*(fotstep/sotstep) + sotstep/2).astype(np.int)
        
        kwargs = dict(ystart=ystart,
                      tstart=self.second_stage.tresult[0],
                      tstartindex=sotstartindex,
                      simtstart=self.simtstart,
                      tend=self.second_stage.tresult[-1],
                      tstep_wanted=sotstep,
                      tstep_min=self.second_stage.tstep_min*2,
                      solver="rkdriver_new",
                      potential_func=self.second_stage.potential_func,
                      pot_params=self.second_stage.pot_params
                      )
        #Update sokwargs with any arguments from soclassargs
        if soclassargs is not None:
            kwargs.update(soclassargs)
            
        #Call superclass
        super(ThirdStageModel, self).__init__(**kwargs)
        
        if soclass is None:
            self.soclass = CanonicalSecondOrder
        else:
            self.soclass = soclass
        self.somodel = None
    
    def setup_soclass(self):
        """Initialize the second order class that will be used to run simulation."""
        sokwargs = {
        "ystart": self.ystart,
        "tstart": self.fotstart,
        "tstartindex": self.tstartindex,
        "simtstart": self.simtstart,
        "tend": self.tend,
        "tstep_wanted": self.tstep_wanted,
        "tstep_min": self.tstep_min,
        "solver": self.solver,
        "k": self.k,
        "ainit": self.ainit,
        "potential_func": self.potential_func,
        "pot_params": self.pot_params,
        "cq": self.cq}
        
        
        self.somodel = self.soclass(**sokwargs)
        self.tname, self.ynames = self.somodel.tname, self.somodel.ynames
        #Set second stage and source terms for somodel
        self.somodel.source = self.source
        self.somodel.second_stage = self.second_stage
        return
    
    def runso(self):
        """Run second order model."""
        
        #Initialize second order class
        self.setup_soclass()
        #Start second order run
        self._log.info("Beginning second order run...")
        try:
            self.somodel.run(saveresults=False)
            pass
        except ModelError:
            self._log.exception("Error in second order run, aborting!")
            raise
        
        self.tresult, self.yresult = self.somodel.tresult, self.somodel.yresult
        return
    
    def run(self, saveresults=True):
        """Run simulation and save results."""
        #Run second order model
        self.runso()
        
        #Save results in resultlist and file
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        
        self.resultlist.append([self.lastparams, self.tresult, self.yresult])        
        
        if saveresults:
            try:
                self._log.info("Results saved in " + self.saveallresults())
            except IOError, er:
                self._log.exception("Error trying to save results! Results NOT saved.")        
        return
    
    
            
class SOCanonicalThreeStage(CanonicalMultiStage, ThirdStageModel):
    """Concrete implementation of ThirdStageCanonical to include second order calculation including
    source term from a first order model."""
    
    #Text for graphs
    plottitle = "Complex Second Order Malik Model with source term in Efold time"
    tname = r"$n$"
    ynames = [r"Real $\delta\varphi_2$",
                    r"Real $\dot{\delta\varphi_2}$",
                    r"Imag $\delta\varphi_2$",
                    r"Imag $\dot{\delta\varphi_2}$"]
                    
    def __init__(self, *args, **kwargs):
        """Initialize variables and call super class __init__ method."""
        super(SOCanonicalThreeStage, self).__init__(*args, **kwargs)
        #try to set source term
        if _debug:
            self._log.debug("Trying to set source term for second order model...")
        self.source = self.second_stage.source[:]
        if self.source is None:
            raise ModelError("First order model does not have a source term!")
        
    def getdeltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi: array_like
                  Array of $\delta\phi$ values for all timesteps and k modes.
        """
        
        if not hasattr(self, "deltaphi") or recompute:
            dp1 = self.second_stage.yresult[:,3,:] + self.second_stage.yresult[:,5,:]*1j
            dp2 = self.yresult[:,0,:] + self.yresult[:,2,:]*1j
            self.deltaphi = dp1 + 0.5*dp2
        return self.deltaphi
        
class CombinedCanonicalFromFile(CanonicalMultiStage):
    """Model class for combined first and second order data, assumed to be used with a file wrapper."""
    
    #Text for graphs
    plottitle = "Combined First and Second Order Canonical Model in Efold time"
    tname = r"$n$"
    ynames = [r"$\varphi_0$",
                r"$\dot{\varphi_0}$",
                r"$H$",
                r"Real $\delta\varphi_1$",
                r"Real $\dot{\delta\varphi_1}$",
                r"Imag $\delta\varphi_1$",
                r"Imag $\dot{\delta\varphi_1}$",
                r"Real $\delta\varphi_2$",
                r"Real $\dot{\delta\varphi_2}$",
                r"Imag $\delta\varphi_2$",
                r"Imag $\dot{\delta\varphi_2}$"]
    
    def __init__(self, *args, **kwargs):
        """Initialize vars and call super class."""
        super(CombinedCanonicalFromFile, self).__init__(*args, **kwargs)
        if "bgclass" not in kwargs or kwargs["bgclass"] is None:
            self.bgclass = CanonicalBackground
        else:
            self.bgclass = bgclass
        if "foclass" not in kwargs or kwargs["foclass"] is None:
            self.foclass = CanonicalFirstOrder
        else:
            self.foclass = foclass
    
    @property
    def deltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi: array_like
                  Array of $\delta\phi$ values for all timesteps and k modes.
        """
        if not hasattr(self, "deltaphi") or recompute:
            dp1 = self.dp1
            dp2 = self.dp2
            self._dp = dp1 + 0.5*dp2
        return self._dp
        
    @property
    def dp1(self, recompute=False):
        """Return (and save) the first order perturbation."""
        
        if not hasattr(self, "_dp1") or recompute:
            dp1 = self.yresult[:,3,:] + self.yresult[:,5,:]*1j
            self._dp1 = dp1
        return self._dp1
    
    @property
    def dp2(self, recompute=False):
        """Return (and save) the first order perturbation."""
        
        if not hasattr(self, "_dp2") or recompute:
            dp2 = self.yresult[:,7,:] + self.yresult[:,9,:]*1j
            self._dp2 = dp2
        return self._dp2
    

class OneZeroIcsTwoStage(TwoStageModel):
    """Implementation of First Order Canonical two stage model with standard initial conditions for phi.
    """
                    
    def __init__(self, *args, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
        #Call superclass
        super(OneZeroIcsTwoStage, self).__init__(*args, **kwargs)
        
    def getfoystart(self, ts=None, tsix=None):
        """Model dependent setting of ystart"""
        if _debug:
            self._log.debug("Executing getfoystart to get initial conditions.")
        #Set variables in standard case:
        if ts is None or tsix is None:
            ts, tsix = self.fotstart, self.fotstartindex
            
        #Reset starting conditions at new time
        foystart = np.zeros((len(self.ystart), len(self.k)))
        
        #Truncate bgmodel yresult down if there is an extra dimension
        if len(self.bgmodel.yresult.shape) > 2:
            bgyresult = self.bgmodel.yresult[..., 0]
        else:
            bgyresult = self.bgmodel.yresult
        
        #Set bg init conditions based on previous bg evolution
        try:
            foystart[0:3] = bgyresult[tsix,:].transpose()
        except ValueError:
            foystart[0:3] = bgyresult[tsix,:][:, np.newaxis]
        
        #Set Re\delta\phi_1 initial condition
        foystart[3,:] = 1.0
        #set Re\dot\delta\phi_1 ic
        foystart[4,:] = 0.0
        #Set Im\delta\phi_1
        foystart[5,:] = 1.0
        #Set Im\dot\delta\phi_1
        foystart[6,:] = 0.0
        
        return foystart
    
    def getdeltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi: array_like
                  Array of $\delta\phi$ values for all timesteps and k modes.
        """
        
        if not hasattr(self, "deltaphi") or recompute:
            self.deltaphi = self.yresult[:,3,:] + self.yresult[:,5,:]*1j #complex deltaphi
        return self.deltaphi 
        
class NonPhysicalNoImagTwoStage(TwoStageModel):
    """Implementation of First Order Canonical two stage model with standard initial conditions for phi.
    """
                    
    def __init__(self, *args, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
        #Call superclass
        super(NonPhysicalNoImagTwoStage, self).__init__(*args, **kwargs)
        
    def getfoystart(self, ts=None, tsix=None):
        """Model dependent setting of ystart"""
        if _debug:
            self._log.debug("Executing getfoystart to get initial conditions.")
        #Set variables in standard case:
        if ts is None or tsix is None:
            ts, tsix = self.fotstart, self.fotstartindex
            
        #Reset starting conditions at new time
        foystart = np.zeros((len(self.ystart), len(self.k)))
        
        #Truncate bgmodel yresult down if there is an extra dimension
        if len(self.bgmodel.yresult.shape) > 2:
            bgyresult = self.bgmodel.yresult[..., 0]
        else:
            bgyresult = self.bgmodel.yresult
        
        #Set bg init conditions based on previous bg evolution
        try:
            foystart[0:3] = bgyresult[tsix,:].transpose()
        except ValueError:
            foystart[0:3] = bgyresult[tsix,:][:, np.newaxis]
        
        #Set Re\delta\phi_1 initial condition
        foystart[3,:] = 1.0
        #set Re\dot\delta\phi_1 ic
        foystart[4,:] = 0.0
        #Set Im\delta\phi_1
        foystart[5,:] = 0.0
        #Set Im\dot\delta\phi_1
        foystart[6,:] = 0.0
        
        return foystart
    
    def getdeltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi: array_like
                  Array of $\delta\phi$ values for all timesteps and k modes.
        """
        
        if not hasattr(self, "deltaphi") or recompute:
            self.deltaphi = self.yresult[:,3,:] + self.yresult[:,5,:]*1j #complex deltaphi
        return self.deltaphi 