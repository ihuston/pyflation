"""cosmomodels.py - Cosmological Model simulations

Provides classes for modelling cosmological inflationary scenarios.
Especially important classes are:

* :class:`FOCanonicalTwoStage` - drives first order calculation 
* :class:`SOCanonicalThreeStage` - drives second order calculation
* :class:`CanonicalFirstOrder` - the class containing derivatives for first order calculation
* :class:`CanonicalRampedSecondOrder` - the class containing derivatives and ramped \
source term for second order calculation.



"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.



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
import analysis
import provenance

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
    
    
    """
    solverlist = ["rkdriver_tsix", "rkdriver_append", "rkdriver_rkf45"]
    
    def __init__(self, ystart=None, simtstart=0.0, tstart=0.0, tstartindex=None, 
                 tend=83.0, tstep_wanted=0.01, solver="rkdriver_tsix", 
                 potential_func=None, pot_params=None, nfields=1, **kwargs):
        """Initialize model variables, some with default values. 
        
        Parameters
        ----------
        ystart : array_like
                initial values for y variables
                
        simtstart : float, optional
                   initial overall time for simulation to start, default is 0.0.
                   
        tstart : array, optional
                individual start times for each k mode, default is 0.0
                
        tstartindex : array, optional
                     individual start time indices for each k mode, default is 0.0.
                     
        tend : float, optional
              overall end time for simulation, default is 83.0
              
        tstep_wanted : float, optional
                      size of time step to use in evolution, default is 0.01
                      
        solver : string, optional
                the name of the rk4 driver function to use, default is "rkdriver_tsix"
                
        potential_func : string, optional
                        the name of the potential function in cmpotentials,
                        default is msqphisq
                        
        pot_params : dict, optional
                    contains modifications to the default parameters in the potential,
                    default is empty dictionary.
                    
        nfields : int, optional
                 the number of fields in the model, default is 1.
        
        """
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
        self.potentials = getattr(cmpotentials, potential_func)
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
    
    def run(self, saveresults=True, yresarr=None, tresarr=None):
        """Execute a simulation run using the parameters already provided."""
            
        # Use python lists instead of pytables earray if not available
        if yresarr is None:
            yresarr = []
        if tresarr is None:
            tresarr = []
        # Check whether postprocess function exists or pass None
        postprocess = getattr(self, "postprocess", None)
        
        # Set up results variables
        self.yresult = yresarr
        self.tresult = tresarr
        if self.solver in ["rkdriver_tsix", "rkdriver_append"]:
            solverargs = dict(ystart=self.ystart, 
                            simtstart=self.simtstart, 
                            tsix=self.tstartindex, 
                            tend=self.tend,
                            h=self.tstep_wanted, 
                            derivs=self.derivs,
                            yarr=self.yresult,
                            xarr=self.tresult,
                            postprocess=postprocess)
        elif self.solver in ["rkdriver_rkf45"]:
            solverargs = dict(ystart=self.ystart, 
                            xstart=self.tstart, 
                            xend=self.tend,
                            h=self.tstep_wanted, 
                            derivs=self.derivs,
                            yarr=self.yresult,
                            xarr=self.tresult,
                            hmax=getattr(self, "hmax", self.tstep_wanted*1e4),
                            hmin=getattr(self, "hmin", self.tstep_wanted*1e-10),
                            abstol=getattr(self, "abstol", 0),
                            reltol=getattr(self, "reltol", 1e-6),
                            postprocess=postprocess)
        if self.solver in self.solverlist:
            if not hasattr(self, "tstartindex"):
                raise ModelError("Need to specify initial starting indices!")
            if _debug:
                self._log.debug("Starting simulation with %s.", self.solver)
            solver = getattr(rk4, self.solver)
            try:
                self.tresult, self.yresult = solver(**solverargs)
            except StandardError:
                self._log.exception("Error running %s!", self.solver)
                raise
            #Change lists back into arrays
            if isinstance(self.yresult, list):
                self.yresult = np.vstack(self.yresult)
            if isinstance(self.tresult, list):
                self.tresult = np.hstack(self.tresult)
        

            ###################################
        else:
            raise ModelError("Unknown solver!")            
        
        if self.solver in ["rkdriver_tsix"]:
            #Need to save results if called with saveresults=True
   
            if saveresults:
                try:
                    self._log.info("Appending results to opened file")
                    self.appendresults(yresarr, tresarr)
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
    
    def get_provenance_values(self):
        """Returns dictionary of provenance details."""      
        #Form dictionary of inputs
        rundir = os.getcwd()
        codedir = os.path.abspath(os.path.dirname(provenance.__file__))
        provdict = provenance.provenance(rundir, codedir)
        self.provenance = provdict
        return provdict
               

    def openresultsfile(self, filename=None, filetype="hf5", yresultshape=None, **kwargs):
        """Open a results file and create the necessary structure.
        
        Parameters
        ----------
        filename : string
                   full path to file
            
        filetype : string
                   filetype to open (currently only "hdf5")
                   
        yresultshape : tuple
                       shape of one row the yresult array, first entry should 
                       be 0.
                       
        kwargs : additional arguments for createhdf5structure method.
        
        Returns
        -------
        
        rf : file handle
        
        grpname : name of results group in file
        
        filename : path to file 
        
        yresarr : handle to EArray
                  array for saving y results
                  
        tresarr : handle to EArray
                  array for saving t results
        
        
        """
        
        if not filename:
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
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
            raise IOError("Directory %s does not exist" % os.path.dirname(filename))
        if yresultshape is None:
            yresultshape = list(self.yresult.shape)
            yresultshape[0] = 0
    #Check whether we should store ks and set group name accordingly
        if self.k is None:
            grpname = "bgresults"
        else:
            grpname = "results"
        if filetype is "hf5":
            try:
                if filemode == "w":
                    rf, yresarr, tresarr = self.createhdf5structure(filename, grpname, yresultshape, **kwargs)
                elif filemode == "a":
                    rf = tables.openFile(filename, filemode)
                    yresarr = rf.getNode(rf.root, grpname + ".yresult")
                    tresarr = rf.getNode(rf.root, grpname + ".tresult")
            except IOError:
                raise
        else:
            raise NotImplementedError("Saving results in format %s is not implemented." % filetype)
        return rf, grpname, filename, yresarr, tresarr

    def saveallresults(self, filename=None, filetype="hf5", yresultshape=None, **kwargs):
        """Saves results already calculated into a file."""
        
        rf, grpname, filename, yresarr, tresarr = self.openresultsfile(filename, filetype, yresultshape, **kwargs)
        
        #Try to save results
        try:
            resgrp = self.saveparamsinhdf5(rf, grpname)
            self.appendresults(yresarr, tresarr)
            self.closehdf5file(rf)
        except IOError:
            self._log.error("Error saving results to %s" % rf)
            raise
        
        return filename
    
    def createhdf5structure(self, filename, grpname="results", yresultshape=None, hdf5complevel=2, hdf5complib="blosc"):
        """Create a new hdf5 file with the structure capable of holding results.
           
        Parameters
        ----------
        filename : string
                   Path including filename of file to create
    
        grpname : string, optional
                  Name of the HDF5 group to create, default is "results"
    
        yresultshape : tuple, optional
                       Shape of yresult variable to store
          
        hdf5complevel :  integer, optional
                         Compression level to use with PyTables, default 2.
    
        hdf5complib : string, optional
                      Compression library to use with PyTables, default "blosc".
    
        Returns
        -------
        rf : file handle
             Handle of file created 
             
        yresarr : handle to EArray
                  array for saving y results
                  
        tresarr : handle to EArray
                  array for saving t results
        """
                    
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
            
            #Add in potential parameters pot_params as a table
            potparamsshape = {"name": tables.StringCol(255),
                              "value": tables.Float64Col()}
            potparamstab = rf.createTable(resgroup, "pot_params", 
                                          potparamsshape, filters=filters)
            #Add provenance information
            provshape = {"name": tables.StringCol(255),
                        "value": tables.StringCol(255)}
            provtab = rf.createTable(resgroup, "provenance", 
                                          provshape, filters=filters)
            
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
                ystartarr = rf.createArray(resgroup, "ystart", self.ystart)
                if hasattr(self, "bgystart"):
                    bgystartarr = rf.createArray(resgroup, "bgystart", self.bgystart)
                if hasattr(self, "foystart"):
                    foystarr = rf.createArray(resgroup, "foystart", self.foystart)
                    fotstarr = rf.createArray(resgroup, "fotstart", self.fotstart)
                    fotsxarr = rf.createArray(resgroup, "fotstartindex", self.fotstartindex)
            else:
                #Only make bg results array
                yresarr = rf.createEArray(resgroup, "yresult", tables.Float64Atom(), yresultshape, filters=filters, expectedrows=8300)
        except IOError:
            raise
        
        return rf, yresarr, tresarr
        



    def saveparamsinhdf5(self, rf, grpname="results"):
        """Save simulation parameters in a HDF5 format file with filename.
        
        Parameters
        ----------
        rf : filelike
            File to save results in

        grpname : string, optional
                 Name of the HDF5 results group
                 
        Returns
        -------
        resgrp : handle for results group in HDF5 file

        """
        try:
            #Get tables and array handles
            resgrp = rf.getNode(rf.root, grpname)
            paramstab = rf.createTable(resgrp, "parameters", 
                                       self.gethf5paramsdict(), 
                                       filters=resgrp.tresult.filters)
            #Now save data
            #Save parameters
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
             
            #Save provenance
            provtab = resgrp.provenance
            provtabrow = provtab.row
            provvalues = self.get_provenance_values()
            for key in provvalues:
                provtabrow["name"] = key
                provtabrow["value"] = provvalues[key]
                provtabrow.append()
            provtab.flush()
            
            #Log success
            if _debug:
                self._log.debug("Successfully wrote parameters to file " + rf.filename)
        except IOError:
            raise
        return resgrp
        
    def closehdf5file(self, rf):
        #Flush saved results to file
        rf.flush() #Close file
        rf.close()
        if _debug:
            self._log.debug("File successfully closed")
        return

    def appendresults(self, yresarr, tresarr):
        #Append y results
        yresarr.append(self.yresult)
        if _debug:
            self._log.debug("yresult array succesfully written.")
        #Save tresults
        tresarr.append(self.tresult)
        if _debug:
            self._log.debug("tresult array successfully written.")
        return
            
class TestModel(CosmologicalModel):
    """Test class defining a very simple function"""
            
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
        
       y[0] - phi_0 : Background inflaton
       y[1] - dphi_0/deta : First deriv of \phi
       y[2] - a : Scale Factor
    """
    
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
       y[1] - d\phi_0/dn : First deriv of \phi
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
        
        #Set the field indices to use
        self.setfieldindices()
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart= np.array([18.0,-0.1]*self.nfields + [0.0] + [1.0,0.0]*self.nfields)
        
        #Set initial H value if None
        if np.all(self.ystart[self.H_ix] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[self.H_ix] = self.findH(U, self.ystart)

    def setfieldindices(self):
        """Set field indices. These can be used to select only certain parts of
        the y variable, e.g. y[self.bg_ix] is the array of background values."""
        self.H_ix = self.nfields * 2
        self.bg_ix = slice(0, self.nfields * 2 + 1)
        self.phis_ix = slice(0, self.nfields * 2, 2)
        self.phidots_ix = slice(1, self.nfields * 2, 2)
        self.pert_ix = slice(self.nfields * 2 + 1, None)
        self.dps_ix = slice(self.nfields * 2 + 1, None, 2)
        self.dpdots_ix = slice(self.nfields * 2 + 2, None, 2)
        return
                       
    def derivs(self, y, t, **kwargs):
        """Return derivatives of fields in y at time t."""
        #If k not given select all
        if "k" not in kwargs or kwargs["k"] is None:
            k = self.k
        else:
            k = kwargs["k"]
        
        #Set up variables    
        phidots = y[self.phidots_ix]
        lenk = len(k)
        #Get a
        a = self.ainit*np.exp(t)
        H = y[self.H_ix]
        nfields = self.nfields    
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y[self.bg_ix,0], self.pot_params)[0:3]        
        
        #Set derivatives taking care of k type
        if type(k) is np.ndarray or type(k) is list: 
            dydx = np.zeros((2*nfields**2 + 2*nfields + 1,lenk), dtype=y.dtype)
            innerterm = np.zeros((nfields,nfields,lenk), dtype=y.dtype)
        else:
            dydx = np.zeros(2*nfields**2 + 2*nfields + 1, dtype=y.dtype)
            innerterm = np.zeros((nfields,nfields), y.dtype)
        
        #d\phi_0/dn = y_1
        dydx[self.phis_ix] = phidots
        #dphi^prime/dn
        dydx[self.phidots_ix] = -(U*phidots+ dUdphi[...,np.newaxis])/(H**2)
        #dH/dn Do sum over fields not ks so use axis=0
        dydx[self.H_ix] = -0.5*(np.sum(phidots**2, axis=0))*H
        #d\delta \phi_I / dn
        dydx[self.dps_ix] = y[self.dpdots_ix]
        
        #Set up delta phis in nfields*nfields array        
        dpmodes = y[self.dps_ix].reshape((nfields, nfields, lenk))
        #This for loop runs over i,j and does the inner summation over l
        for i in range(nfields):
            for j in range(nfields):
                #Inner loop over fields
                for l in range(nfields):
                    innerterm[i,j] += (d2Udphi2[i,l] + (phidots[i]*dUdphi[l] 
                                        + dUdphi[i]*phidots[l] 
                                        + phidots[i]*phidots[l]*U))*dpmodes[l,j]
        #Reshape this term so that it is nfields**2 long        
        innerterm = innerterm.reshape((nfields**2,lenk))
        #d\deltaphi_1^prime/dn
        dydx[self.dpdots_ix] = -(U * y[self.dpdots_ix]/H**2 + (k/(a*H))**2 * y[self.dps_ix]
                                + innerterm/H**2)
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
            
        References
        ----------
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
        Hend : scalar
               value of Hubble parameter at end of inflation.
               
        Hreh : scalar, optional
               value of Hubble parameter at end of reheating, default is Hend.
                
        a_0 : scalar, optional
              value of scale factor today, default is 1. 
        
        Returns
        -------
        a_end : scalar
                scale factor at the end of inflation.
        
        """
        N_after = self.find_efolds_after_inflation(Hend, Hreh)
        a_end = a_0*np.exp(-N_after)
        return a_end
    
    def finda_0(self, Hend, Hreh=None, a_end=None):
        """Given the Hubble parameter at the end of inflation and at the end of reheating,
        and the scale factor at the end of inflation, calculate the scale factor today.
        
        Parameters
        ----------
        Hend : scalar
               value of Hubble parameter at end of inflation
               
        Hreh : scalar, optional
               value of Hubble parameter at end of reheating, default is Hend.
                
        a_end : scalar, optional
                value of scale factor at the end of inflation, default is calculated
                from results. 
        
        Returns
        -------
        a_0 : scalar
              scale factor today
        
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
        """Given k, time variable and Hubble parameter, find when mode k crosses the horizon.
        
        Parameters
        ----------
        k : float
            Single k value to compute crossing time with

        t : array
            Array of time values

        H : array
            Array of values of the Hubble parameter

        factor : float, optional
                 coefficient of crossing k = a*H*factor

        Returns
        -------
        kcrindex, kcrefold : tuple
                             Tuple containing k cross index (in t variable) and the efold number
                             e.g. t[kcrindex]

        """
        #threshold
        err = 1.0e-26
        if factor is None:
            factor = self.cq #time before horizon crossing
        #get aHs
        if len(H.shape) > 1:
            #Use only one dimensional H
            H = np.squeeze(H)
        aH = self.ainit*np.exp(t)*H
        try:
            kcrindex = np.where(np.sign(k - (factor*aH))<0)[0][0]
        except IndexError, ex:
            raise ModelError("k mode " + str(k) + " crosses horizon after end of inflation!")
        kcrefold = t[kcrindex]
        return kcrindex, kcrefold
    
    def findallkcrossings(self, t, H):
        """Iterate over findkcrossing to get full list
        
        Parameters
        ----------
        t : array
            Array of t values to calculate over

        H : array
            Array of Hubble parameter values, should be the same shape as t

        Returns
        -------
        kcrossings : array
                     Array of (kcrindex, kcrefold) pairs of index (in to t) and efold number
                     at which each k in self.k crosses the horizon (k=a*H).
        """
        return np.array([self.findkcrossing(onek, t, H) for onek in self.k])
    
    def findHorizoncrossings(self, factor=1):
        """Find horizon crossing time indices and efolds for all ks
        
        Parameters
        ----------
        factor : float
                 Value of coefficient to calculate crossing time, k=a*H*factor

        Returns
        -------
        hcrossings : array
                     Array of (kcrindex, kcrefold) pairs of time index 
                     and efold number pairs
                
        
        """
        return np.array([self.findkcrossing(onek, self.tresult, oneH, factor) for onek, oneH in zip(self.k, np.rollaxis(self.yresult[:,2,:], -1,0))])
    
    @property
    def deltaphi(self, recompute=False):
        """Return the value of deltaphi for this model, recomputing if necessary."""
        pass
        
    @property
    def Pr(self):
        """The power spectrum of comoving curvature perturbation.
        This is the unscaled spectrum P_R calculated for all timesteps and ks. 
        Calculated using the pyflation.analysis package.
        """
        return analysis.Pr(self)
    
    @property
    def Pzeta(self):
        """The power spectrum of the curvature perturbation on uniform energy
        density hypersurfaces.
        
        Calculated using the pyflation.analysis package.
        """
        return analysis.Pzeta(self)
    
    @property
    def scaled_Pr(self):
        """The power spectrum of comoving curvature perturbation.
        
        Calculated using the pyflation.analysis package.
        """
        return analysis.scaled_Pr(self)
    
    @property
    def scaled_Pzeta(self):
        """The power spectrum of comoving curvature perturbation.
        
        Calculated using the pyflation.analysis package.
        """
        return analysis.scaled_Pzeta(self)
                
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


class FODriver(MultiStageDriver): 
    """Generic class for two stage model with a background and first order pass."""
       
    def __init__(self, *args, **kwargs):
        super(FODriver, self).__init__(*args, **kwargs)
        
    def find_ainit(self):
        """Find initial conditions for 1st order model
           Find a_end using instantaneous reheating
        """
        Hend = np.asscalar(self.bgmodel.yresult[self.inflendindex, self.H_ix])
        self.a_end = np.asscalar(self.finda_end(Hend))
        self.ainit = np.asscalar(self.a_end*np.exp(-self.bgmodel.tresult[self.inflendindex]))
        return
    
    def setfoics(self):
        """After a bg run has completed, set the initial conditions for the 
            first order run."""
        #debug
        #set_trace()
        # Make sure ainit has been calculated.
        self.find_ainit()
            
        #Find epsilon from bg model
        try:
            self.bgepsilon
        except AttributeError:            
            self.bgepsilon = self.bgmodel.getepsilon()
        #Set etainit, initial eta at n=0
        self.etainit = -1/(self.ainit*self.bgmodel.yresult[0,self.H_ix]*(1-self.bgepsilon[0]))
        
        #find k crossing indices
        kcrossings = self.findallkcrossings(self.bgmodel.tresult[:self.inflendindex], 
                            self.bgmodel.yresult[:self.inflendindex, self.H_ix])
        kcrossefolds = kcrossings[:,1]
                
        #If mode crosses horizon before t=0 then we will not be able to propagate it
        if any(kcrossefolds==0):
            raise ModelError("Some k modes crossed horizon before simulation began and cannot be initialized!")
        
        #Find new start time from earliest kcrossing
        self.fotstart, self.fotstartindex = kcrossefolds, kcrossings[:,0].astype(np.int)
        self.foystart = self.getfoystart()
        return
    
    #These should be implemented in concrete classes
    def getbgargs(self):
        pass
    
    def getfoargs(self):
        pass
    
    def postfo_cleanup(self):
        pass
    
    def find_fotend(self):
        """Set the end time of first order run.
            Override to use different end time e.g. after reheating"""
        self.fotend = self.inflation_end
        self.fotendindex = self.inflendindex
        return
    
    def runbg(self):
        """Run bg model after setting initial conditions."""

        kwargs = self.getbgargs()
         
        self.bgmodel = self.bgclass(**kwargs)
        #Start background run
        self._log.info("Running background model...")
        try:
            self.bgmodel.run(saveresults=False)
        except ModelError:
            self._log.exception("Error in background run, aborting!")
        #Find end of inflation
        self.inflation_end, self.inflendindex = self.bgmodel.findinflend()
        self._log.info("Background run complete, inflation ended " + str(self.inflation_end) + " efoldings after start.")
        self.find_fotend()
        return
    
    def runfo(self, saveresults, yresarr, tresarr):
        """Run first order model after setting initial conditions."""

        kwargs = self.getfoargs()
        
        self.firstordermodel = self.foclass(**kwargs)

        #Start first order run
        self._log.info("Beginning first order run...")
        try:
            self.firstordermodel.run(saveresults=saveresults, yresarr=yresarr,
                                     tresarr=tresarr)
        except ModelError, er:
            raise ModelError("Error in first order run, aborting! Message: " + er.message)
        
        #Set results to current object
        self.tresult, self.yresult = self.firstordermodel.tresult, self.firstordermodel.yresult
        self.postfo_cleanup()
        return
    
    def run(self, saveresults=True, saveargs=None):
        """Run the full model.
        
        The background model is first run to establish the end time of inflation and the start
        times for the k modes. Then the initial conditions are set for the first order variables.
        Finally the first order model is run and the results are saved if required.
        
        Parameters
        ----------
        saveresults : boolean, optional
                      Should results be saved at the end of the run. Default is True.
                      
        saveargs : dict, optional
                   Dictionary of keyword arguments to pass to file saving routines.
                   See Cosmomodels.openresultsfile, .saveallresults, 
                   .createhdf5structure, .saveparamsinhdf5 for more arguments.
                     
        Returns
        -------
        filename : string
                   name of the results file if any
        """
        #Run bg model
        self.runbg()
        
        #Set initial conditions for first order model
        self.setfoics()
        
        if saveargs is None:
            saveargs = {}
        
        if saveresults:
            ystartshape = list(self.foystart.shape)
            ystartshape.insert(0, 0)
            saveargs["yresultshape"] = ystartshape 
            #Set up results file
            rf, grpname, filename, yresarr, tresarr = self.openresultsfile(**saveargs)
            self._log.info("Opened results file %s.", filename)
        else:
            yresarr = None
            tresarr = None
            filename = None
        #Run first order model
        self.runfo(saveresults, yresarr, tresarr)
        
        #Save results in file
        if saveresults:
            try:
                resgrp = self.saveparamsinhdf5(rf, grpname)
                self._log.info("Saved parameters in file.")
                self._log.info("Closing file")
                self.closehdf5file(rf)
            except IOError:
                self._log.exception("Error trying to close file! Results may not be saved.")
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()        
        return filename
    
    def getdeltaphi(self):
        return self.deltaphi
    
    def deltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times, fields and modes.
        For multifield systems this is the quantum matrix of solutions:
        
        \hat{\delta\phi} = \Sum_{\alpha, I} xi_{\alpha I} \hat{a}_I
        
        The result is stored as the instance variable m.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute : boolean, optional
                    Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi : array_like, dtype: complex128
                   Array of $\delta\phi$ values for all timesteps, fields and k modes.
        """
        
        if not hasattr(self, "_deltaphi") or recompute:
            self._deltaphi = self.yresult[:,self.dps_ix,:]
        return self._deltaphi
    
    #Helper functions to access results variables
    @property
    def phis(self):
        """Background fields \phi_i"""
        return self.yresult[:,self.phis_ix]
    
    @property
    def phidots(self):
        """Derivatives of background fields w.r.t N \phi_i^\dagger"""
        return self.yresult[:,self.phidots_ix]
    
    @property
    def H(self):
        """Hubble parameter"""
        return self.yresult[:,self.H_ix]
    
    @property
    def dpmodes(self):
        """Quantum modes of first order perturbations"""
        return self.yresult[:,self.dps_ix]
    
    @property
    def dpdotmodes(self):
        """Quantum modes of derivatives of first order perturbations"""
        return self.yresult[:,self.dpdots_ix]
    
    @property
    def a(self):
        """Scale factor of the universe"""
        return self.ainit*np.exp(self.tresult)

class FOCanonicalTwoStage(FODriver):
    """Uses a background and firstorder class to run a full (first-order) simulation.
        Main additional functionality is in determining initial conditions.
        Variables finally stored are as in first order class.
    """ 
                                                      
    def __init__(self, bgystart=None, tstart=0.0, tstartindex=None, tend=83.0, tstep_wanted=0.01,
                 k=None, ainit=None, solver="rkdriver_tsix", bgclass=None, foclass=None, 
                 potential_func=None, pot_params=None, simtstart=0, nfields=1, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
      
        #Initial conditions for each of the variables.
        if bgystart is None:
            self.bgystart = np.array([18.0/np.sqrt(nfields),-0.1/np.sqrt(nfields)]*nfields 
                                  + [0.0])
        else:
            self.bgystart = bgystart
        #Lengthen bgystart to add perturbed fields.
        self.ystart= np.append(self.bgystart, [0.0,0.0]*nfields**2)
            
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
        
        #Set the field indices
        self.setfieldindices()
        
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

    def setfieldindices(self):
        """Set field indices. These can be used to select only certain parts of
        the y variable, e.g. y[self.bg_ix] is the array of background values."""
        self.H_ix = self.nfields * 2
        self.bg_ix = slice(0, self.nfields * 2 + 1)
        self.phis_ix = slice(0, self.nfields * 2, 2)
        self.phidots_ix = slice(1, self.nfields * 2, 2)
        self.pert_ix = slice(self.nfields * 2 + 1, None)
        self.dps_ix = slice(self.nfields * 2 + 1, None, 2)
        self.dpdots_ix = slice(self.nfields * 2 + 2, None, 2)
        return
    
    def getbgargs(self):
        #Check ystart is in right form (1-d array of three values)
        if len(self.ystart.shape) == 1:
            ys = self.ystart[self.bg_ix]
        elif len(self.ystart.shape) == 2:
            ys = self.ystart[self.bg_ix,0]
        #Choose tstartindex to be simply the first timestep.
        tstartindex = np.array([0])
        
        args = dict(ystart=ys, 
                      tstart=self.tstart,
                      tstartindex=tstartindex, 
                      tend=self.tend,
                      tstep_wanted=self.tstep_wanted, 
                      solver=self.solver,
                      potential_func=self.potential_func, 
                      pot_params=self.pot_params,
                      nfields=self.nfields)
        return args
        
    def getfoargs(self):
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
        return kwargs
    
    
        
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
        
    def setfieldindices(self):
        """Set field indices. These can be used to select only certain parts of
        the y variable, e.g. y[self.bg_ix] is the array of background values."""
        # Indices for use with self.second_stage.yresult
        self.H_ix = self.nfields * 2
        self.bg_ix = slice(0, self.nfields * 2 + 1)
        self.phis_ix = slice(0, self.nfields * 2, 2)
        self.phidots_ix = slice(1, self.nfields * 2, 2)
        self.pert_ix = slice(self.nfields * 2 + 1, None)
        self.dps_ix = slice(self.nfields * 2 + 1, None, 2)
        self.dpdots_ix = slice(self.nfields * 2 + 2, None, 2)
        
        #Indices of second order quantities, to use with self.yresult
        self.dp2s_ix = slice(0, None, 2)
        self.dp2dots_ix = slice(1, None, 2)
        return
                    
    
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
    
    def runso(self, saveresults, yresarr, tresarr):
        """Run second order model."""
        
        #Initialize second order class
        self.setup_soclass()
        #Start second order run
        self._log.info("Beginning second order run...")
        try:
            self.somodel.run(saveresults, yresarr, tresarr)
            pass
        except ModelError:
            self._log.exception("Error in second order run, aborting!")
            raise
        
        self.tresult, self.yresult = self.somodel.tresult, self.somodel.yresult
        return
    
    def run(self, saveresults=True, saveargs=None):
        """Run the full model.
        
        The second order model is run in full using the first order and convolution results.
        
        Parameters
        ----------
        saveresults : boolean, optional
                      Should results be saved at the end of the run. Default is True.
                      
        saveargs : dict, optional
                   Dictionary of keyword arguments to pass to file saving routines.
                   See Cosmomodels.openresultsfile, .saveallresults, 
                   .createhdf5structure, .saveparamsinhdf5 for more arguments.
                     
        Returns
        -------
        filename : string
                   name of the results file if any
        """
        
        
        #Save results in file
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()

        if saveargs is None:
            saveargs = {}

        if saveresults:
            ystartshape = list(self.ystart.shape)
            ystartshape.insert(0, 0)
            saveargs["yresultshape"] = ystartshape 
            #Set up results file
            rf, grpname, filename, yresarr, tresarr = self.openresultsfile(**saveargs)
            self._log.info("Opened results file %s.", filename)
            resgrp = self.saveparamsinhdf5(rf, grpname)
            self._log.info("Saved parameters in file.")
        
        #Run second order model
        self.runso(saveresults, yresarr, tresarr)
        
        if saveresults:
            try:
                self._log.info("Closing file")
                self.closehdf5file(rf)
            except IOError:
                self._log.exception("Error trying to save results! Results NOT saved.")        
        return filename
    
    @property
    def deltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute : boolean, optional
                    Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi : array_like
                   Array of $\delta\phi$ values for all timesteps and k modes.
        """
        
        if not hasattr(self, "_deltaphi") or recompute:
            dp1 = self.second_stage.yresult[:,3,:] + self.second_stage.yresult[:,5,:]*1j
            dp2 = self.yresult[:,0,:] + self.yresult[:,2,:]*1j
            self._deltaphi = dp1 + 0.5*dp2
        return self._deltaphi
    
    #Helper functions to access results variables
    @property
    def phis(self):
        """Background fields \phi_i"""
        return self.second_stage.yresult[:,self.phis_ix]
    
    @property
    def phidots(self):
        """Derivatives of background fields w.r.t N \phi_i^\dagger"""
        return self.second_stage.yresult[:,self.phidots_ix]
    
    @property
    def H(self):
        """Hubble parameter"""
        return self.second_stage.yresult[:,self.H_ix]
    
    @property
    def dpmodes(self):
        """Quantum modes of first order perturbations"""
        return self.second_stage.yresult[:,self.dps_ix]
    
    @property
    def dpdotmodes(self):
        """Quantum modes of derivatives of first order perturbations"""
        return self.second_stage.yresult[:,self.dpdots_ix]
    
    @property
    def dp2modes(self):
        """Quantum modes of second order perturbations"""
        return self.yresult[:,self.dp2s_ix]
    
    @property
    def dp2dotmodes(self):
        """Quantum modes of derivatives of second order perturbations"""
        return self.yresult[:,self.dp2dots_ix]
    
    @property
    def a(self):
        """Scale factor of the universe"""
        return self.ainit*np.exp(self.tresult)
    
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
            self.bgclass = kwargs["bgclass"]
        if "foclass" not in kwargs or kwargs["foclass"] is None:
            self.foclass = CanonicalFirstOrder
        else:
            self.foclass = kwargs["foclass"]
    
    @property
    def deltaphi(self, recompute=False):
        """Return the calculated values of $\delta\phi$ for all times and modes.
        
        The result is stored as the instance variable self.deltaphi but will be recomputed
        if `recompute` is True.
        
        Parameters
        ----------
        recompute : boolean, optional
                    Should the values be recomputed? Default is False.
                   
        Returns
        -------
        deltaphi : array_like
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
    
    """ 

    def __init__(self, *args, **kwargs):
        """Extra keyword argument ainit is used to set value of ainit no matter
        what the value of H at the end of inflation.
        
        Parameters
        ----------
        ainit : float
                value of ainit to fix no matter what the value of H at the
                end of inflation.
    
        """
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
        kcrossings = self.findallkcrossings(self.bgmodel.tresult[:self.inflendindex], 
                            self.bgmodel.yresult[:self.inflendindex,self.H_ix])
        kcrossefolds = kcrossings[:,1]
                
        #If mode crosses horizon before t=0 then we will not be able to propagate it
        if any(kcrossefolds==0):
            raise ModelError("Some k modes crossed horizon before simulation began and cannot be initialized!")
        
        #Find new start time from earliest kcrossing
        self.fotstart, self.fotstartindex = kcrossefolds, kcrossings[:,0].astype(np.int)
        self.foystart = self.getfoystart()
        return
        
        
class FONoPhase(FOCanonicalTwoStage):
    """First order two stage class which does not include a phase in the initial
    conditions for the first order field."""
    
    def __init__(self, *args, **kwargs):
        super(FONoPhase, self).__init__(*args, **kwargs)
        
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
#        Hzero = bgyresult[0,self.H_ix]
#        
#        epsstar = self.bgepsilon[tsix]
#        etastar = -1/(astar*Hstar*(1-epsstar))
#        try:
#            etadiff = etastar - self.etainit
#        except AttributeError:
#            etadiff = etastar + 1/(self.ainit*Hzero*(1-self.bgepsilon[0]))
#        keta = self.k*etadiff
        
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
        foystart[self.dps_ix,:][::self.nfields+1] = arootk
        #set \dot\delta\phi_1 ic

        foystart[self.dpdots_ix,:][::self.nfields+1] = -arootk*(1 + (self.k/(astar*Hstar))*1j)
        
        return foystart   
    

class FOSuppressOneField(FOCanonicalTwoStage):
    """First order two stage class which does not include a phase in the initial
    conditions for the first order field."""
    
    def __init__(self, suppress_ix=0, *args, **kwargs):
        
        self.suppress_ix = suppress_ix
        super(FOSuppressOneField, self).__init__(*args, **kwargs)
        
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
        
        #Suppress one field using self.suppress_ix
        foystart[self.dps_ix,:][::self.nfields+1][self.suppress_ix] = 0
        foystart[self.dpdots_ix,:][::self.nfields+1][self.suppress_ix] = 0
                
        return foystart
