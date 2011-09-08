"""cosmomodels.py - Cosmological Model simulations

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

Provides classes for modelling cosmological inflationary scenarios.
Especially important classes are:

FOCanonicalTwoStage - drives first order calculation 
SOCanonicalThreeStage - drives second order calculation

CanonicalFirstOrder - the class containing derivatives for first order calculation
CanonicalRampedSecondOrder - the class containing derivatives and ramped
                             source term for second order calculation.

The make_wrapper_model function takes a filename and returns a model instance
corresponding to the one stored in the file. This allows easier access to the
results than through a direct inspection of the HDF5 results file.


"""

from __future__ import division

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

#Profiling decorator when not using profiling.
if not "profile" in __builtins__:
    def profile(f):
        return f

class ModelError(StandardError):
    """Generic error for model simulating. Attributes include current results stack."""
    pass

class CosmologicalModel(object):
    """Generic class for cosmological model simulations.
    Contains run() method which chooses a solver and runs the simulation.
    
    All cosmological model classes are subclassed from this one.
    
    Initialization arguments
    ------------------------
    ystart - array_like, initial values for y variables
    simtstart - float, initial overall time for simulation to start
    tstart - array, individual start times for each k mode
    tstartindex - array, individual start time indices for each k mode
    tend - float, overall end time for simulation
    tstep_wanted - float, size of time step to use in evolution
    solver - string, the name of the rk4 driver function to use
    potential_func - string, the name of the potential function in cmpotentials
    pot_params - dict, any modifications to the default parameters in the potential
    
    """
    solverlist = ["rkdriver_tsix"]
    
    def __init__(self, ystart=None, simtstart=0.0, tstart=0.0, tstartindex=None, 
                 tend=83.0, tstep_wanted=0.01, solver="rkdriver_tsix", 
                 potential_func=None, pot_params=None, nfields=1, **kwargs):
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
        
        self.tstep_wanted = tstep_wanted
        
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
            
        #Set the number of fields using keyword argument, defaults to 1.
        if nfields < 1:
            raise ValueError("Cannot have zero or negative number of fields.")
        else:
            self.nfields = nfields        
             
        self.tresult = None #Will hold last time result
        self.yresult = None #Will hold array of last y results
        
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
                  "solver":self.solver,
                  "classname":self.__class__.__name__,
                  "datetime":datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                  "nfields":self.nfields
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
        "datetime" : tables.Float64Col(),
        "nfields" : tables.IntCol()
        }
        return params   
           
    def saveallresults(self, filename=None, filetype="hf5", yresultshape=None, **kwargs):
        """Saves results already calculated into a file."""
        
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
        
        #Check whether we should store ks and set group name accordingly
        if self.k is None:
            grpname = "bgresults"
            if yresultshape is None:
                yresultshape = (0, self.nfields*2+1, 1)
        else:
            grpname = "results"
            if yresultshape is None:
                yresultshape = (0, self.nfields*4+1, len(self.k))
                
        if filetype is "hf5":
            try:
                if filemode == "w":
                    rf = self.createhdf5structure(filename, grpname, yresultshape, **kwargs)
                elif filemode == "a":
                    rf = tables.openFile(filename, filemode)
                self.saveresultsinhdf5(rf, grpname)
            except IOError:
                raise
        else:
            raise NotImplementedError("Saving results in format %s is not implemented." % filetype)
        return filename
    
    def createhdf5structure(self, filename, grpname="results", yresultshape=None, hdf5complevel=2, hdf5complib="blosc"):
        """Create a new hdf5 file with the structure capable of holding results."""
                    
        try:
            rf = tables.openFile(filename, "w")
            # Select which compression library to use in configuration
            filters = tables.Filters(complevel=hdf5complevel, complib=hdf5complib)
            
            #Create groups required
            resgroup = rf.createGroup(rf.root, grpname, "Results of simulation")
            tresarr = rf.createEArray(resgroup, "tresult", 
                                      tables.Float64Atom(), 
                                      (0,), #Shape of a single atom 
                                      filters=filters, 
                                      expectedrows=8194)
            paramstab = rf.createTable(resgroup, "parameters", 
                                       self.gethf5paramsdict(), 
                                       filters=filters)
            #Add in potential parameters pot_params as a table
            potparamsshape = {"name": tables.StringCol(255),
                              "value": tables.Float64Col()}
            potparamstab = rf.createTable(resgroup, "pot_params", 
                                          potparamsshape, filters=filters)
            
            #Need to check if results are k dependent
            if grpname is "results":
                if hasattr(self, "bgmodel"):
                    #Store bg results:
                    bggrp = rf.createGroup(rf.root, "bgresults", "Background results")
                    bgtrarr = rf.createArray(bggrp, "tresult", self.bgmodel.tresult)
                    bgyarr = rf.createArray(bggrp, "yresult", self.bgmodel.yresult)
                #Save results
                yresarr = rf.createEArray(resgroup, "yresult", tables.ComplexAtom(itemsize=16), yresultshape, filters=filters, expectedrows=8194)
                karr = rf.createArray(resgroup, "k", self.k)
                if hasattr(self, "foystart"):
                    foystarr = rf.createArray(resgroup, "foystart", self.foystart)
                    fotstarr = rf.createArray(resgroup, "fotstart", self.fotstart)
                    fotsxarr = rf.createArray(resgroup, "fotstartindex", self.fotstartindex)
            else:
                #Only make bg results array
                yresarr = rf.createEArray(resgroup, "yresult", tables.Float64Atom(), yresultshape, filters=filters, expectedrows=8300)
        except IOError:
            raise
        
        return rf
        
    def saveresultsinhdf5(self, rf, grpname="results"):
        """Save simulation results in a HDF5 format file with filename.
            filename - full path and name of file (should end in hf5 for consistency.
            filemode - ["w"|"a"]: "w" specifies write to a new file, overwriting existing one
                        "a" specifies append to current file or create if does not exist.
            hdf5complevel - Compression level to use with PyTables, default 2.
            hdf5complib - Compression library to use with PyTables, default "blosc".
        """
        try:
            #Get tables and array handles
            resgrp = rf.getNode(rf.root, grpname)
            
            #Now save data
            #Save parameters
            paramstab = resgrp.parameters
            paramstabrow = paramstab.row
            params = self.callingparams()
            for key in params:
                paramstabrow[key] = params[key]
            paramstabrow.append() #Add to table
            paramstab.flush()
            
            #Save potential parameters
            potparamstab = resgrp.pot_params
            potparamsrow = potparamstab.row
            for key in self.pot_params:
                potparamsrow["name"] = key
                potparamsrow["value"] = self.pot_params[key]
                potparamsrow.append()
            potparamstab.flush()
             
            #Get yresult array handle
            yresarr = resgrp.yresult
            yresarr.append(self.yresult)
            
            #Flush saved results to file
            rf.flush()
            #Close file
            rf.close()
            #Log success
            if _debug:
                self._log.debug("Successfully wrote results to file " + rf.filename)
        except IOError:
            raise
            
class TestModel(CosmologicalModel):
    """Test class defining a very simple function"""
    #Names of variables
    ynames = [r"Simple $y$", r"$\dot{y}$"]
    plottitle = r"TestModel: $\frac{d^2y}{dt^2} = y$"
    tname = "Time"
            
    def __init__(self, ystart=np.array([1.0,1.0]), tstart=0.0, tend=1.0, tstep_wanted=0.01):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted)
        

    
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
                    tstep_wanted=0.02, solver="rkdriver_tsix"):
        
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, solver=solver)
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
        phidot = y[self.phidots_ix]
        
        #Expression for H
        H = np.sqrt(U/(3.0-0.5*(np.sum(phidot**2))))
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
            phidots = self.yresult[:,self.phidots_ix,0]
        else:
            phidots = self.yresult[:,self.phidots_ix]
        #Make sure to do sum across only phidot axis (1 in this case)
        epsilon = 0.5*np.sum(phidots**2, axis=1)
        return epsilon

    
class CanonicalBackground(PhiModels):
    """Basic model with background equations in terms of n
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\n : First deriv of \phi
       y[2] - H: Hubble parameter
    """
        
    def __init__(self,  *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalBackground, self).__init__(*args, **kwargs)
        
        #Set field indices. These can be used to select only certain parts of
        #the y variable, e.g. y[self.bg_ix] is the array of background values.
        self.H_ix = self.nfields*2
        self.bg_ix = slice(0,self.nfields*2+1)
        self.phis_ix = slice(0,self.nfields*2,2)
        self.phidots_ix = slice(1,self.nfields*2,2)
        
        #Set initial H value if None
        if np.all(self.ystart[self.H_ix] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[self.H_ix] = self.findH(U, self.ystart)
    
    def derivs(self, y, t, **kwargs):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
                
        #get potential from function
        U, dUdphi = self.potentials(y, self.pot_params)[0:2]       
        
        #Set derivatives
        dydx = np.zeros_like(y)
        
        #d\phi_0/dn = y_1
        dydx[self.phis_ix] = y[self.phidots_ix] 
        
        #dphi^prime/dn
        dydx[self.phidots_ix] = -(U*y[self.phidots_ix] + dUdphi[...,np.newaxis])/(y[self.H_ix]**2)
        
        #dH/dn
        dydx[self.H_ix] = -0.5*(np.sum(y[self.phidots_ix]**2, axis=0))*y[self.H_ix]

        return dydx

class CanonicalFirstOrder(PhiModels):
    """First order model using efold as time variable with multiple fields.
    
    nfields holds the number of fields and the yresult variable is then laid
    out as follows:
    
    yresult[0:nfields*2] : background fields and derivatives
    yresult[nfields*2] : Hubble variable H
    yresult[nfields*2 + 1:] : perturbation fields and derivatives
       """
            
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
        
        #Set field indices. These can be used to select only certain parts of
        #the y variable, e.g. y[self.bg_ix] is the array of background values.
        self.H_ix = self.nfields*2
        self.bg_ix = slice(0,self.nfields*2+1)
        self.phis_ix = slice(0,self.nfields*2,2)
        self.phidots_ix = slice(1,self.nfields*2,2)
        self.pert_ix = slice(self.nfields*2+1, None)
        self.dps_ix = slice(self.nfields*2+1, None, 2)
        self.dpdots_ix = slice(self.nfields*2+2, None, 2)
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart= np.array([18.0,-0.1]*self.nfields + [0.0] + [1.0,0.0]*self.nfields)
        
        #Set initial H value if None
        if np.all(self.ystart[self.H_ix] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[self.H_ix] = self.findH(U, self.ystart)
                       
    @profile                        
    def derivs(self, y, t, **kwargs):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        #If k not given select all
        if "k" not in kwargs or kwargs["k"] is None:
            k = self.k
        else:
            k = kwargs["k"]
            
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y[self.bg_ix,0], self.pot_params)[0:3]        
        
        #Set derivatives taking care of k type
        if type(k) is np.ndarray or type(k) is list: 
            dydx = np.zeros((2*self.nfields**2 + 2*self.nfields + 1,len(k)), dtype=y.dtype)
        else:
            dydx = np.zeros(2*self.nfields**2 + 2*self.nfields + 1, dtype=y.dtype)
            
        #d\phi_0/dn = y_1
        dydx[self.phis_ix] = y[self.phidots_ix] 
        
        #dphi^prime/dn
        dydx[self.phidots_ix] = -(U*y[self.phidots_ix] + dUdphi[...,np.newaxis])/(y[self.H_ix]**2)
        
        #dH/dn Do sum over fields not ks so use axis=0
        dydx[self.H_ix] = -0.5*(np.sum(y[self.phidots_ix]**2, axis=0))*y[self.H_ix]
        
        #Get a
        a = self.ainit*np.exp(t)
        H = y[self.H_ix]
        
        dydx[self.dps_ix] = y[self.dpdots_ix]
        
        #Sum term for perturbation
        phidot = y[self.phidots_ix]
        term1 = (d2Udphi2[...,np.newaxis] 
                + phidot[:,np.newaxis,:]*dUdphi[np.newaxis,:,np.newaxis] 
                + dUdphi[:,np.newaxis,np.newaxis] * phidot[np.newaxis,...]
                + phidot[:,np.newaxis,:]*phidot[np.newaxis,...]*U )
        term2 = y[self.dps_ix].reshape((self.nfields, self.nfields, len(k)))
        
        
        termsum = np.sum(term1[:,:,np.newaxis,:]*term2[np.newaxis,...], axis=-2)
        termsum = termsum.reshape((self.nfields**2,len(k)))
        
        #d\deltaphi_1^prime/dn  
        # Do sum over second field index so axis=-1
        dydx[self.dpdots_ix] = -(U * y[self.dpdots_ix]/H**2 + (k/(a*H))**2 * y[self.dps_ix]
                                + termsum/H**2)
        return dydx
        

class CanonicalSecondOrder(PhiModels):
    """Second order model using efold as time variable.
       y[0] - \delta\varphi_2 : Second order perturbation [Real Part]
       y[1] - \delta\varphi_2^\prime : Derivative of second order perturbation [Real Part]
       y[2] - \delta\varphi_2 : Second order perturbation [Imag Part]
       y[3] - \delta\varphi_2^\prime : Derivative of second order perturbation [Imag Part]
       """
                        
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
        #if _debug:
        #    self._log.debug("args: %s", str(kwargs))
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
            src = self.source[fotix].copy()
        else:
            fovars = self.second_stage.yresult[fotix].copy()[:,kix]
            src = self.source[fotix][kix].copy()
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
            needramp = abs(tanharg)<self.rampargs["e"]
            if _debug:
                self._log.debug("Limits of indices which need ramp are %s.", np.where(needramp)[0][[0,-1]])
            src[needramp] = ramp[needramp]*src[needramp]
        
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
        
class MultiStageDriver(CosmologicalModel):
    """Parent of all multi (2 or 3) stage models. Contains methods to determine ns, k crossing and outlines
    methods to find Pr that are implemented in children."""
    
    def __init__(self, *args, **kwargs):
        """Initialize super class instance."""
        super(MultiStageDriver, self).__init__(*args, **kwargs)
        #Set constant factor for 1st order initial conditions
        if "cq" in kwargs:
            self.cq = kwargs["cq"]
        else:
            self.cq = 50 #Default value as in Salopek et al.
        
    
    def find_efolds_after_inflation(self, Hend, Hreh=None):
        """Calculate the number of efolds after inflation given the reheating
        temperature and assuming standard calculation of radiation and matter phases.
        
        Parameters
        ----------
        Hend : scalar, value of Hubble parameter at end of inflation
        Hreh : scalar (default=Hend), value of Hubble parameter at end of reheating
        
        Returns
        -------
        N : scalar, number of efolds after the end of inflation until today.
            N = ln (a_today/a_end) where a_end is scale factor at end of inflation.
            
        Further Information
        -------------------
        See Huston, arXiv: 1006.5321, 
        Liddle and Lyth, Cambridge University Press 2000, or 
        Peiris and Easther, JCAP 0807 (2008) 024, arXiv:0805.2154, 
        for more details on calculation of post-inflation expansion. 
        """
        if Hreh is None:
            Hreh = Hend #Instantaneous reheating
        N_after = 72.3 + 2.0/3.0*np.log(Hend) - 1.0/6.0*np.log(Hreh)
        return N_after
        
    def finda_end(self, Hend, Hreh=None, a_0=1):
        """Given the Hubble parameter at the end of inflation and at the end of reheating
            calculate the scale factor at the end of inflation.
            
        This function assumes by default that the scale factor = 1 today and should be used with 
        caution. A more correct approach is to call find_efolds_after_inflation directly
        and to use the result as required. 
        
        Parameters
        ----------
        Hend : scalar, value of Hubble parameter at end of inflation
        Hreh : scalar (default=Hend), value of Hubble parameter at end of reheating
        a_0 : scalar (default=1), value of scale factor today
        
        Returns
        -------
        a_end : scalar, scale factor at the end of inflation
        
        """
        N_after = self.find_efolds_after_inflation(Hend, Hreh)
        a_end = a_0*np.exp(-N_after)
        return a_end
    
    def finda_0(self, Hend, Hreh=None, a_end=None):
        """Given the Hubble parameter at the end of inflation and at the end of reheating,
        and the scale factor at the end of inflation, calculate the scale factor today.
        
        Parameters
        ----------
        Hend : scalar, value of Hubble parameter at end of inflation
        Hreh : scalar (default=Hend), value of Hubble parameter at end of reheating
        a_end : scalar (default calculated from tresult), value of scale factor 
                at the end of inflation
        
        Returns
        -------
        a_0 : scalar, scale factor today
        
        """
        if a_end is None:
            try:
                a_end = self.ainit*np.exp(self.tresult[-1])
            except TypeError:
                raise ModelError("Simulation has not been run yet.")
            
        N_after = self.find_efolds_after_inflation(Hend, Hreh)
        a_0 = a_end*np.exp(N_after)
        return a_0 
        
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
    
    @property
    def deltaphi(self, recompute=False):
        """Return the value of deltaphi for this model, recomputing if necessary."""
        pass
    
    @property
    def Pphi(self, recompute=False):
        """Return the spectrum of scalar perturbations P_phi for each field and k.
        
        This is the unscaled version $P_{\phi}$ which is related to the scaled version by
        $\mathcal{P}_{\phi} = k^3/(2pi^2) P_{\phi}$. Note that result is stored as the
        instance variable self.Pphi. 
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should value be recomputed even if already stored? Default is False.
        
        Returns
        -------
        Pphi: array_like, dtype: float64
              3-d array of Pphi values for all timesteps, fields and k modes
        """
        #Basic caching of result
        if not hasattr(self, "_Pphi") or recompute:        
            deltaphi = self.deltaphi
            self._Pphi = np.float64(deltaphi*deltaphi.conj())
        return self._Pphi
    
    @property
    def calPphi(self):
        """Return the spectrum of scalar perturbations for each field and k mode.
        
        This is the scaled power spectrum $\mathcal{P}_{\phi_I}$ for each field
        and is given by
        
        $\mathcal{P}_{\delta\varphi_I}(k) = k^3/(2\pi^2) |\delta\varphi_I(k)|^2.$
        
        Returns
        -------
        calPphi: array_like, dtype: float64
                 3-d array of calPphi values for all timesteps, fields and k modes
              
        """
        return 1/(2*np.pi**2) * self.k**3 * self.Pphi
    
    @property
    def Pr(self, recompute=False):
        pass
    
    @property
    def Pgrav(self, recompute=False):
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
        if not hasattr(self, "_Pgrav") or recompute:        
            self._Pgrav = 2*self.Pphi
        return self._Pgrav
            
    def getfoystart(self):
        """Return model dependent setting of ystart""" 
        pass

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
                  "solver":self.solver,
                  "classname":self.__class__.__name__,
                  "datetime":datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                  "nfields":self.nfields
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
        "datetime" : tables.Float64Col(),
        "nfields" : tables.IntCol()
        }
        return params
    
class FOCanonicalTwoStage(MultiStageDriver):
    """Uses a background and firstorder class to run a full (first-order) simulation.
        Main additional functionality is in determining initial conditions.
        Variables finally stored are as in first order class.
    """ 
                                                      
    def __init__(self, ystart=None, tstart=0.0, tstartindex=None, tend=83.0, tstep_wanted=0.01,
                 k=None, ainit=None, solver="rkdriver_tsix", bgclass=None, foclass=None, 
                 potential_func=None, pot_params=None, simtstart=0, nfields=1, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
      
        #Initial conditions for each of the variables.
        if ystart is None:
            self.ystart= np.array([18.0/np.sqrt(nfields),-0.1/np.sqrt(nfields)]*nfields 
                                  + [0.0] + [1.0,0.0]*nfields**2)
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
                         solver=solver, 
                         potential_func=potential_func, 
                         pot_params=pot_params,
                         nfields=nfields, 
                         **kwargs)
        
        super(FOCanonicalTwoStage, self).__init__(**newkwargs)
        
        #Set field indices. These can be used to select only certain parts of
        #the y variable, e.g. y[self.bg_ix] is the array of background values.
        self.H_ix = self.nfields*2
        self.bg_ix = slice(0,self.nfields*2+1)
        self.phis_ix = slice(0,self.nfields*2,2)
        self.phidots_ix = slice(1,self.nfields*2,2)
        self.pert_ix = slice(self.nfields*2+1, None)
        self.dps_ix = slice(self.nfields*2+1, None, 2)
        self.dpdots_ix = slice(self.nfields*2+2, None, 2)
        
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
        Hend = self.bgmodel.yresult[self.fotendindex, self.H_ix]
        self.a_end = self.finda_end(Hend)
        self.ainit = self.a_end*np.exp(-self.bgmodel.tresult[self.fotendindex])
        
        
        #Find epsilon from bg model
        try:
            self.bgepsilon
        except AttributeError:            
            self.bgepsilon = self.bgmodel.getepsilon()
        #Set etainit, initial eta at n=0
        self.etainit = -1/(self.ainit*self.bgmodel.yresult[0,self.H_ix]*(1-self.bgepsilon[0]))
        
        #find k crossing indices
        kcrossings = self.findallkcrossings(self.bgmodel.tresult[:self.fotendindex], 
                            self.bgmodel.yresult[:self.fotendindex, self.H_ix])
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
            ys = self.ystart[self.bg_ix]
        elif len(self.ystart.shape) == 2:
            ys = self.ystart[self.bg_ix,0]
        #Choose tstartindex to be simply the first timestep.
        tstartindex = np.array([0])
        
        kwargs = dict(ystart=ys, 
                      tstart=self.tstart,
                      tstartindex=tstartindex, 
                      tend=self.tend,
                      tstep_wanted=self.tstep_wanted, 
                      solver=self.solver,
                      potential_func=self.potential_func, 
                      pot_params=self.pot_params,
                      nfields=self.nfields)
         
        self.bgmodel = self.bgclass(**kwargs)
        #Start background run
        self._log.info("Running background model...")
        try:
            self.bgmodel.run(saveresults=False)
        except ModelError:
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
                      solver=self.solver,
                      k=self.k, 
                      ainit=self.ainit, 
                      potential_func=self.potential_func, 
                      pot_params=self.pot_params,
                      nfields=self.nfields)
        
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
        
        #Save results in file
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()   
        
        if saveresults:
            try:
                self._log.info("Results saved in " + self.saveallresults())
            except IOError, er:
                self._log.exception("Error trying to save results! Results NOT saved.")        
        return
        
    def getfoystart(self, ts=None, tsix=None):
        """Model dependent setting of ystart"""
        if _debug:
            self._log.debug("Executing getfoystart to get initial conditions.")
        #Set variables in standard case:
        if ts is None or tsix is None:
            ts, tsix = self.fotstart, self.fotstartindex
            
        #Reset starting conditions at new time
        foystart = np.zeros(((2*self.nfields**2 + self.nfields*2 +1), len(self.k)), dtype=np.complex128)
        #set_trace()
        #Get values of needed variables at crossing time.
        astar = self.ainit*np.exp(ts)
        
        #Truncate bgmodel yresult down if there is an extra dimension
        if len(self.bgmodel.yresult.shape) > 2:
            bgyresult = self.bgmodel.yresult[..., 0]
        else:
            bgyresult = self.bgmodel.yresult
            
        Hstar = bgyresult[tsix,self.H_ix]
        Hzero = bgyresult[0,self.H_ix]
        
        epsstar = self.bgepsilon[tsix]
        etastar = -1/(astar*Hstar*(1-epsstar))
        try:
            etadiff = etastar - self.etainit
        except AttributeError:
            etadiff = etastar + 1/(self.ainit*Hzero*(1-self.bgepsilon[0]))
        keta = self.k*etadiff
        
        #Set bg init conditions based on previous bg evolution
        try:
            foystart[self.bg_ix] = bgyresult[tsix,:].transpose()
        except ValueError:
            foystart[self.bg_ix] = bgyresult[tsix,:][:, np.newaxis]
        
        #Find 1/asqrt(2k)
        arootk = 1/(astar*(np.sqrt(2*self.k)))
                
        #Only want to set the diagonal elements of the mode matrix
        #Use a.flat[::a.shape[1]+1] to set diagonal elements only
        #In our case already flat so foystart[slice,:][::nfields+1]
        #Set \delta\phi_1 initial condition
        foystart[self.dps_ix,:][::self.nfields+1] = arootk*np.exp(-keta*1j)
        #set \dot\delta\phi_1 ic

        foystart[self.dpdots_ix,:][::self.nfields+1] = -arootk*np.exp(-keta*1j)*(1 + (self.k/(astar*Hstar))*1j)
        
        return foystart
    
    def getdeltaphi(self):
        return self.deltaphi
    
    def getmodematrix(self, y, ix=None, ixslice=None):
        """Helper function to reshape flat nfield^2 long y variable into nfield*nfield mode
        matrix. Returns a view of the y array (changes will be reflected in underlying array).
        
        Parameters
        ----------
        ixslice: index slice, optional
            The index slice of y to use, defaults to full extent of y.
            
        Returns
        -------
        
        result: view of y array with shape nfield*nfield structure
        """
        if ix is None:
            #Use second dimension for index slice by default
            ix = 1
        if ixslice is None:
            #Assume slice is full extent if none given.
            ixslice = slice(None)
        indices = [Ellipsis]*len(y.shape)
        indices[ix] = ixslice
        modes = y[indices]
            
        s = list(modes.shape)
        #Check resulting array is correct shape
        if s[ix] != self.nfields**2:
            raise ModelError("Array does not have correct dimensions of nfields**2.")
        s[ix] = self.nfields
        s.insert(ix+1, self.nfields)
        result = modes.reshape(s)
        return result
    
    def flattenmodematrix(self, modematrix, ix1=None, ix2=None):
        """Flatten the mode matrix given into nfield^2 long vector."""
        s = modematrix.shape
        if s.count(self.nfields) < 2:
            raise ModelError("Mode matrix does not have two nfield long dimensions.")
        try:
            #If indices are not specified, use first two in order
            if ix1 is None:
                ix1 = s.index(self.nfields)
            if ix2 is None:
                #The second index is assumed to be after ix1
                ix2 = s.index(self.nfields, ix1+1)
        except ValueError:
            raise ModelError("Cannot determine correct indices for nfield long dimensions!")
        slist = list(s)
        ix2out = slist.pop(ix2)
        slist[ix1] = self.nfields**2
        return modematrix.reshape(slist) 
        
        
    @property
    def deltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times, fields and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi: array_like, dtype: complex128
                  Array of $\delta\phi$ values for all timesteps, fields and k modes.
        """
        
        if not hasattr(self, "_deltaphi") or recompute:
            self._deltaphi = self.yresult[:,self.dps_ix,:]
        return self._deltaphi
    
    @property
    def Pphi(self, recompute=False):
        """Return the spectrum of scalar perturbations P_phi for each field and k.
        
        This is the unscaled version $P_{\phi}$ which is related to the scaled version by
        $\mathcal{P}_{\phi} = k^3/(2pi^2) P_{\phi}$. Note that result is stored as the
        instance variable self.Pphi.
        For multifield systems the full crossterm matrix is returned which 
        has shape nfields*nfields flattened down to a vector of length nfields^2. 
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should value be recomputed even if already stored? Default is False.
        
        Returns
        -------
        Pphi: array_like, dtype: float64
              3-d array of Pphi values for all timesteps, fields and k modes
        """
        #Basic caching of result
        if not hasattr(self, "_Pphi") or recompute:        
            mdp = self.getmodematrix(self.yresult, 1, self.dps_ix)
            mPphi = (mdp[:,:,:,np.newaxis,:]*mdp[:,:,np.newaxis,:,:].conj()).sum(axis=2)
            self._Pphi = self.flattenmodematrix(mPphi, 1, 2) 
        return self._Pphi
    
    
    def findns(self, k=None, nefolds=3):
        """Return the value of n_s at the specified k mode, nefolds after horizon crossing."""
        
        #If k is not defined, get value at all self.k
        if k is None:
            k = self.k
        else:
            if k<self.k.min() and k>self.k.max():
                self._log.warn("Warning: Extrapolating to k value outside those used in spline!")
        
        ts = self.findallkcrossings(self.tresult, self.yresult[:,2], factor=1)[:,0] + nefolds/self.tstep_wanted #About nefolds after horizon exit
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
    
    def findHorizoncrossings(self, factor=1):
        """Find horizon crossing for all ks"""
        return self.findallkcrossings(self.tresult, self.yresult[:,2], factor)
         
    @property            
    def Pr(self, recompute=False):
        """Return the spectrum of curvature perturbations $P_R$ for each k.
        
        For a multifield model this is given by:
        
        Pr = (\Sum_K \dot{\phi_K}^2 )^{-2} 
                \Sum_{I,J} \dot{\phi_I} \dot{\phi_J} P_{IJ}
                
        where P_{IJ} = \Sum_K \chi_{IK} \chi_JK}
        and \chi are the mode matrix elements.  
        
        This is the unscaled version $P_R$ which is related to the scaled version by
        $\mathcal{P}_R = k^3/(2pi^2) P_R$. Note that result is stored as the instance variable
        self.Pr. 
        
        Parameters
        ----------
        recompute: boolean, optional
                   Should value be recomputed even if already stored? Default is False.
                   
        Returns
        -------
        Pr: array_like, dtype: float64
            Array of Pr values for all timesteps and k modes
        """
        #Basic caching of result
        if not hasattr(self, "_Pr") or recompute:        
            phidot = np.float64(self.yresult[:,self.phidots_ix,:]) #bg phidot
            phidotsumsq = (np.sum(phidot**2, axis=1))**2
            #Get mode matrix for Pphi as nfield*nfield
            Pphimatrix = self.getmodematrix(self.Pphi, 1, slice(None))
            #Multiply mode matrix by corresponding phidot value
            summatrix = phidot[:,np.newaxis,:,:]*phidot[:,:,np.newaxis,:]*Pphimatrix
            #Flatten mode matrix and sum over all nfield**2 values
            sumflat = np.sum(self.flattenmodematrix(summatrix, 1, 2), axis=1)
            #Divide by total sum of derivative terms
            self._Pr = sumflat/phidotsumsq
        return self._Pr
    
    @property            
    def calPr(self):
        """Return the spectrum of curvature perturbations $\mathcal{P}_\mathcal{R}$ 
        for each timestep and k mode.
        
        This is the scaled power spectrum which is related to the unscaled version by
        $\mathcal{P}_\mathcal{R} = k^3/(2pi^2) P_\mathcal{R}$. 
         
        Returns
        -------
        calPr: array_like
            Array of Pr values for all timesteps and k modes
        """
        #Basic caching of result
        return 1/(2*np.pi**2) * self.k**3 * self.Pr           
    
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
            
            #Set indices correctly
            self.H_ix = self.nfields*2
            self.bg_ix = slice(0,self.nfields*2+1)
            self.phis_ix = slice(0,self.nfields*2,2)
            self.phidots_ix = slice(1,self.nfields*2,2)
            self.pert_ix = slice(self.nfields*2+1, None)
            self.dps_ix = slice(self.nfields*2+1, None, 2)
            self.dpdots_ix = slice(self.nfields*2+2, None, 2)
            
            #Fix bgmodel to actual instance
            if self.ystart is not None:
                #Check ystart is in right form (1-d array of three values)
                if len(self.ystart.shape) == 1:
                    ys = self.ystart[self.bg_ix]
                elif len(self.ystart.shape) == 2:
                    ys = self.ystart[self.bg_ix,0]
            else:
                ys = self.foystart[self.bg_ix,0]
            self.bgmodel = self.bgclass(ystart=ys, tstart=self.tstart, tend=self.tend, 
                            tstep_wanted=self.tstep_wanted, solver=self.solver,
                            potential_func=self.potential_func, 
                            nfields=self.nfields, pot_params=self.pot_params)
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


class SOCanonicalThreeStage(MultiStageDriver):
    """Runs third stage calculation (typically second order perturbations) using
    a two stage model instance which could be wrapped from a file."""

    def __init__(self, second_stage, soclass=None, ystart=None, **soclassargs):
        """Initialize variables and check that tsmodel exists and is correct form."""
        
        #Test whether tsmodel is of correct type
        if not isinstance(second_stage, FOCanonicalTwoStage):
            raise ModelError("Need to provide a FOCanonicalTwoStage instance to get first order results from!")
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
            self.nfields = self.second_stage.nfields
        
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
                      solver="rkdriver_tsix",
                      potential_func=self.second_stage.potential_func,
                      pot_params=self.second_stage.pot_params,
                      nfields=self.nfields
                      )
        #Update sokwargs with any arguments from soclassargs
        if soclassargs is not None:
            kwargs.update(soclassargs)
            
        #Call superclass
        super(SOCanonicalThreeStage, self).__init__(**kwargs)
        
        if soclass is None:
            self.soclass = CanonicalSecondOrder
        else:
            self.soclass = soclass
        self.somodel = None
        
        #Set up source term
        if _debug:
            self._log.debug("Trying to set source term for second order model...")
        self.source = self.second_stage.source[:]
        if self.source is None:
            raise ModelError("First order model does not have a source term!")
        #Try to put yresult array in memory
        self.second_stage.yresultarr = self.second_stage.yresult
        self.second_stage.yresult = self.second_stage.yresultarr[:]
    
    def setup_soclass(self):
        """Initialize the second order class that will be used to run simulation."""
        sokwargs = {
        "ystart": self.ystart,
        "tstart": self.fotstart,
        "tstartindex": self.tstartindex,
        "simtstart": self.simtstart,
        "tend": self.tend,
        "tstep_wanted": self.tstep_wanted,
        "solver": self.solver,
        "k": self.k,
        "ainit": self.ainit,
        "potential_func": self.potential_func,
        "pot_params": self.pot_params,
        "cq": self.cq,
        "nfields": self.nfields,
        }
        
        
        self.somodel = self.soclass(**sokwargs)
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
        
        #Save results in file
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        if saveresults:
            try:
                self._log.info("Results saved in " + self.saveallresults())
            except IOError, er:
                self._log.exception("Error trying to save results! Results NOT saved.")        
        return
    
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
        
        if not hasattr(self, "_deltaphi") or recompute:
            dp1 = self.second_stage.yresult[:,3,:] + self.second_stage.yresult[:,5,:]*1j
            dp2 = self.yresult[:,0,:] + self.yresult[:,2,:]*1j
            self._deltaphi = dp1 + 0.5*dp2
        return self._deltaphi
    
class SOHorizonStart(SOCanonicalThreeStage):
    """Runs third stage calculation (typically second order perturbations) using
    a two stage model instance which could be wrapped from a file.
    
    Second order calculation starts at horizon crossing.
    """
    

    def __init__(self, second_stage, soclass=None, ystart=None, **soclassargs):
        """Initialize variables and check that tsmodel exists and is correct form."""
        
        
        #Test whether tsmodel is of correct type
        if not isinstance(second_stage, FOCanonicalTwoStage):
            raise ModelError("Need to provide a FOCanonicalTwoStage instance to get first order results from!")
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
        
        fohorizons = np.array([second_stage.findkcrossing(second_stage.k[kix],
                                                         second_stage.bgmodel.tresult,
                                                         second_stage.bgmodel.yresult[:,2],
                                                         factor=1) for kix in np.arange(len(second_stage.k)) ])
        fohorizonindex = fohorizons[:,0]
        fohorizontimes = fohorizons[:,1]
        
        sotstartindex = np.around(fohorizonindex*(fotstep/sotstep) + sotstep/2).astype(np.int)
        
        kwargs = dict(ystart=ystart,
                      tstart=self.second_stage.tresult[0],
                      tstartindex=sotstartindex,
                      simtstart=self.simtstart,
                      tend=self.second_stage.tresult[-1],
                      tstep_wanted=sotstep,
                      solver="rkdriver_new",
                      potential_func=self.second_stage.potential_func,
                      pot_params=self.second_stage.pot_params
                      )
        #Update sokwargs with any arguments from soclassargs
        if soclassargs is not None:
            kwargs.update(soclassargs)
            
        #Call superclass
        super(SOCanonicalThreeStage, self).__init__(**kwargs)
        
        if soclass is None:
            self.soclass = CanonicalSecondOrder
        else:
            self.soclass = soclass
        self.somodel = None
        
        #Set up source term
        if _debug:
            self._log.debug("Trying to set source term for second order model...")
        self.source = self.second_stage.source[:]
        if self.source is None:
            raise ModelError("First order model does not have a source term!")
        #Try to put yresult array in memory
        self.second_stage.yresultarr = self.second_stage.yresult
        self.second_stage.yresult = self.second_stage.yresultarr[:]
        
class CombinedCanonicalFromFile(MultiStageDriver):
    """Model class for combined first and second order data, assumed to be used with a file wrapper."""
    
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

class FixedainitTwoStage(FOCanonicalTwoStage):
    """First order driver class with ainit fixed to a specified value.
    This is useful for comparing models which have different H behaviour and
    therefore different ainit values. 
    
    It should be remembered that these models with fixed ainit are equivalent
    to changing the number of efoldings from the end of inflation until today.
    
    Extra argument in initialisation
    --------------------------------
    ainit - float, value of ainit to fix no matter what the value of H at the
            end of inflation.
    
    """ 

    def __init__(self, *args, **kwargs):
        """Extra keyword argument ainit is used to set value of ainit no matter
        what the value of H at the end of inflation."""
        super(FixedainitTwoStage, self).__init__(*args, **kwargs)
        if "ainit" in kwargs:
            self.fixedainit = kwargs["ainit"]
        else:
            #Set with ainit from standard msqphisq potential run.
            self.fixedainit = 7.837219134630212e-65
            
    def setfoics(self):
        """After a bg run has completed, set the initial conditions for the 
            first order run."""
        
        #Find initial conditions for 1st order model
        #Use fixed ainit in this class instead of calculating from aend
        self.ainit = self.fixedainit
                
        #Find epsilon from bg model
        try:
            self.bgepsilon
        except AttributeError:            
            self.bgepsilon = self.bgmodel.getepsilon()
        #Set etainit, initial eta at n=0
        self.etainit = -1/(self.ainit*self.bgmodel.yresult[0,self.H_ix]*(1-self.bgepsilon[0]))
        
        #find k crossing indices
        kcrossings = self.findallkcrossings(self.bgmodel.tresult[:self.fotendindex], 
                            self.bgmodel.yresult[:self.fotendindex,self.H_ix])
        kcrossefolds = kcrossings[:,1]
                
        #If mode crosses horizon before t=0 then we will not be able to propagate it
        if any(kcrossefolds==0):
            raise ModelError("Some k modes crossed horizon before simulation began and cannot be initialized!")
        
        #Find new start time from earliest kcrossing
        self.fotstart, self.fotstartindex = kcrossefolds, kcrossings[:,0].astype(np.int)
        self.foystart = self.getfoystart()
        return
        