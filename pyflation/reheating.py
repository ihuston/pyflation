"""reheating.py - Cosmological models for reheating simulations

Provides classes for modelling cosmological reheating scenarios.
Especially important classes are:

* :class:`ReheatingBackground` - the class containing derivatives for first order calculation
* :class:`ReheatingFirstOrder` - the class containing derivatives for first order calculation
* :class:`FOReheatingTwoStage` - drives first order calculation 

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.

from __future__ import division

#system modules
import numpy as np
from scipy import interpolate
import logging
import tables
import datetime

#local modules from pyflation
from pyflation import cosmomodels as c

#Start logging
root_log_name = logging.getLogger().name
module_logger = logging.getLogger(root_log_name + "." + __name__)



class ReheatingModels(c.PhiModels):
    '''
    Base class for background and first order reheating model classes.
    '''


    def __init__(self, *args, **kwargs):
        '''
        Constructor
        '''
        super(ReheatingModels, self).__init__(*args, **kwargs)
        
        # Set default transfer coefficient values if not specified
        # Default is no transfer to fluids from fields
        self.transfers = kwargs.get("transfers", np.zeros((self.nfields,2)))
        if self.transfers.shape != (self.nfields,2):
            raise ValueError("Shape of transfer coefficient is array is wrong.")
        self.transfers_on = np.zeros_like(self.transfers, dtype=np.bool)
        self.transfers_on_times = np.zeros_like(self.transfers)
        self.last_pdot_sign = np.zeros((self.nfields,))
        
        # Set default value of rho_limit. If ratio of energy density of 
        # fields to total energy density falls below this the fields are not
        # included in the calculation any longer.
        self.rho_limit = kwargs.get("rho_limit", 1e-5)
        # This setting means the fields will be evolved initially.
        self.fields_off = False
 
    def findH(self, U, y):
        """Return value of Hubble variable, H at y for given potential."""
        phidot = y[self.phidots_ix]
        rhomatter = y[self.rhomatter_ix]
        rhogamma = y[self.rhogamma_ix]
        
        #Expression for H
        H = np.sqrt((rhomatter + rhogamma + U)/(3.0-0.5*(np.sum(phidot**2))))
        return H
    
    def getepsilon(self):
        """Return an array of epsilon = -\dot{H}/H values for each timestep."""
        #Find Hdot
        if len(self.yresult.shape) == 3:
            phidots = self.yresult[:,self.phidots_ix,0]
            rhogamma = self.yresult[:,self.rhogamma_ix,0]
            rhomatter = self.yresult[:,self.rhomatter_ix,0]
            Hsq = self.yresult[:,self.H_ix,0]**2
        else:
            phidots = self.yresult[:,self.phidots_ix]
            rhogamma = self.yresult[:,self.rhogamma_ix]
            rhomatter = self.yresult[:,self.rhomatter_ix]
            Hsq = self.yresult[:,self.H_ix]**2
        #Make sure to do sum across only phidot axis (1 in this case)
        epsilon = (0.5*rhomatter + 2.0/3.0*rhogamma)/Hsq + 0.5*np.sum(phidots**2, axis=1)
        return epsilon 
    
    def getrho_fields(self):
        """Return an array of the total energy density of the scalar
        fields and the total energy density for each timestep.
        
        Returns
        -------
        rho_fields : array
                     Energy density of the scalar fields at all timesteps
                     
        total_rho : array
                    Total energy density of the system at all timesteps
        """
        #Find Hdot
        if len(self.yresult.shape) == 3:
            Hsq = self.yresult[:,self.H_ix,0]**2
            phidots = self.yresult[:,self.phidots_ix,0]
            
        else:
            Hsq = self.yresult[:,self.H_ix]**2
            phidots = self.yresult[:,self.phidots_ix]
        pdotsq = 0.5*np.sum(phidots**2, axis=1)
        U = np.array([self.potentials(myr, self.pot_params)[0] for myr in self.yresult])
        #Make sure to do sum across only phidot axis (1 in this case)
        rho_fields = 0.5*Hsq*pdotsq + U 
        return rho_fields, 3*Hsq
    
    def find_reheating_end(self, fraction=0.01):
        """Find the efold time where reheating ends,
            i.e. the energy density of inflaton fields < 1% of total.
            Returns tuple of endefold and endindex (in tresult)."""
        
        rho_fields, total_rho = self.getrho_fields()
        
        rho_fraction = rho_fields/total_rho
        
        if not any(self.epsilon>1):
            raise c.ModelError("Reheating did not end during specified number of efoldings.")
        
        endindex = np.where(rho_fraction<=fraction)[0][0]
        
        #Interpolate results to find more accurate endpoint
        tck = interpolate.splrep(self.tresult[:endindex], rho_fraction[:endindex])
        t2 = np.linspace(self.tresult[endindex-1], self.tresult[endindex], 100)
        y2 = interpolate.splev(t2, tck)
        endindex2 = np.where(y2 < 0.01)[0][0]
        #Return efold of more accurate endpoint
        endefold = t2[endindex2]
        
        return endefold, endindex
    
class ReheatingBackground(ReheatingModels):
    """Model of background equations for reheating in a two field, two fluid system
    """
        
    def __init__(self,  *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(ReheatingBackground, self).__init__(*args, **kwargs)
        
        
        #set field indices
        self.setfieldindices()
        
        #Set initial H value if None
        if np.all(self.ystart[self.H_ix] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[self.H_ix] = self.findH(U, self.ystart)
            

    
    def setfieldindices(self):
        """Set field indices. These can be used to select only certain parts of
        the y variable, e.g. y[self.bg_ix] is the array of background values.
        """
        self.H_ix = self.nfields*2
        self.bg_ix = slice(0,self.nfields*2+3)
        self.phis_ix = slice(0,self.nfields*2,2)
        self.phidots_ix = slice(1,self.nfields*2,2)
        self.rhogamma_ix = self.nfields*2 + 1
        self.rhomatter_ix = self.nfields*2 + 2
        
        #Indices for transfer array
        self.tgamma_ix = 0
        self.tmatter_ix = 1
    
    def derivs(self, y, t, **kwargs):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
        #Set local variables
        rhogamma = y[self.rhogamma_ix]
        rhomatter = y[self.rhomatter_ix]
        
        #Set derivatives
        dydx = np.zeros_like(y)
        
        if self.fields_off:
            #Set H without fields 
            H = y[self.H_ix]
            
            #Calculate H derivative now as we need it later
            Hdot = -((0.5*rhomatter + 2.0/3.0*rhogamma)/H)
            #d\phi_0/dn = y_1
            dydx[self.phis_ix] = 0
            
            #dphi^prime/dn
            dydx[self.phidots_ix] = 0
            
            #dH/dn We do not evolve H at each step, only saving the result 
            #of the Friedmann constraint equation at each step.
            dydx[self.H_ix] = 0
            
            # Fluids
            dydx[self.rhogamma_ix] = -4*rhogamma
            
            dydx[self.rhomatter_ix] = -3*rhomatter
        else: #Fields are on and need to be used
            #get potential from function
            U, dUdphi = self.potentials(y, self.pot_params)[0:2]       
            # Get field dependent variables
            phidots = y[self.phidots_ix]
            pdotsq = np.sum(phidots**2, axis=0)
            active_transfers = self.transfers*self.transfers_on
            tgamma = active_transfers[:,self.tgamma_ix][...,np.newaxis]
            tmatter = active_transfers[:,self.tmatter_ix][...,np.newaxis]
            
            #Use H from y variable
            H = y[self.H_ix]
            
            #Calculate H derivative now as we need it later
            Hdot = -((0.5*rhomatter + 2.0/3.0*rhogamma)/H + 0.5*H*pdotsq)
            #d\phi_0/dn = y_1
            dydx[self.phis_ix] = phidots
            
            #dphi^prime/dn
            dydx[self.phidots_ix] = -((3 + Hdot/H + 0.5/H * (tgamma + tmatter))*phidots 
                                       + dUdphi[...,np.newaxis]/(H**2))
            
            #dH/dn We do not evolve H in this case, only storing the 
            #result from the Friedmann constraint at each time step.
            dydx[self.H_ix] = 0
            
            # Fluids
            dydx[self.rhogamma_ix] = -4*rhogamma + 0.5*H*np.sum(tgamma*phidots**2, axis=0)
            
            dydx[self.rhomatter_ix] = -3*rhomatter + 0.5*H*np.sum(tmatter*phidots**2, axis=0)
        

        return dydx
    
    def postprocess(self, y, t):
        """Postprocess step takes place after RK4 step and can modify y values
        at the end of that step.
        
        This is used to turn off scalar fields when they drop below a specified
        level of the matter and radiation energy densities.
        The value of H is also recalculated to improve numerical stability.
        
        Parameters
        ----------
        y : array_like
            array of this timesteps y values
            
        t : float
            value of t variable at this timestep
            
        Returns
        -------
        y : array_like
            modified array of y values.
            
        """
        
        #Set local variables
        rhogamma = y[self.rhogamma_ix]
        rhomatter = y[self.rhomatter_ix]
        
        #Check whether transfers should be on
        if not np.all(self.transfers_on):
            #Switch on any transfers if minimum is passed
            signchanged = self.last_pdot_sign*y[self.phidots_ix,0] < 0
            self.transfers_on_times[signchanged[:,np.newaxis]*(np.logical_not(self.transfers_on))] = t
            self.transfers_on[signchanged] = True
            self.last_pdot_sign = np.sign(y[self.phidots_ix,0])
                
        # Only do check if fields are still being used
        if self.fields_off:
            #Fields are off but set to zero anyway
            Hsq = (rhogamma + rhomatter)/(3)
            H = np.sqrt(Hsq)
            y[self.H_ix] = H
            y[self.phis_ix] = 0
            y[self.phidots_ix] = 0
        else:
            #get potential from function
            U = self.potentials(y, self.pot_params)[0]       
            # Get field dependent variables
            phidots = y[self.phidots_ix]
            pdotsq = np.sum(phidots**2, axis=0)
            #Calculate rho for the fields to check if it's not negligible
            #Update H to use fields
            Hsq = (rhogamma + rhomatter + U)/(3-0.5*pdotsq)
            H = np.sqrt(Hsq)
            y[self.H_ix] = H
            
            rho_fields = 0.5*Hsq*pdotsq + U
            rho_total = 3*Hsq
            
            if rho_fields/rho_total < self.rho_limit:
                self.fields_off = True
                self.fields_off_time = t
                #Set fields at this timestep to be zero.
                y[self.phis_ix] = 0
                y[self.phidots_ix] = 0
        
        return y
        


class ReheatingFirstOrder(ReheatingModels):
    """First order model with two fluids for reheating.
    
    nfields holds the number of fields and the yresult variable is then laid
    out as follows:
    
    yresult[0:nfields*2] : background fields and derivatives
    yresult[nfields*2] : Hubble variable H
    yresult[nfields*2 + 1:] : perturbation fields and derivatives
       """
            
    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(ReheatingFirstOrder, self).__init__(*args, **kwargs)
        
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
                
        self.phis_ix = slice(0, self.nfields * 2, 2)
        self.phidots_ix = slice(1, self.nfields * 2, 2)
        
        self.H_ix = slice(self.phidots_ix.stop, self.phidots_ix.stop + 1)
                
        self.rhogamma_ix = slice(self.H_ix.stop, self.H_ix.stop + 1)
        self.rhomatter_ix = slice(self.rhogamma_ix.stop, self.rhogamma_ix.stop + 1)
        
        self.bg_ix = slice(0, self.rhomatter_ix.stop)
        
        self.pert_ix = slice(self.bg_ix.stop, None)
        self.dps_ix = slice(self.pert_ix.start, self.pert_ix.start + 2*self.nfields**2, 2)
        self.dpdots_ix = slice(self.dps_ix.start + 1, self.dps_ix.stop, 2)
        
        #Fluid perturbations
        self.dgamma_ix = slice(self.dpdots_ix.stop, self.dpdots_ix.stop + self.nfields, 2)
        self.dmatter_ix = slice(self.dgamma_ix.stop, self.dgamma_ix.stop + self.nfields, 2)
        self.Vgamma_ix = slice(self.dmatter_ix.stop, self.dmatter_ix.stop + self.nfields, 2)
        self.Vmatter_ix = slice(self.Vgamma_ix.stop, self.Vgamma_ix.stop + self.nfields, 2)
        
        #Indices for transfer array
        self.tgamma_ix = 0
        self.tmatter_ix = 1
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
        #Set local variables
        rhogamma = y[self.rhogamma_ix]
        rhomatter = y[self.rhomatter_ix]
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
        
        if self.fields_off:
            #Set H without fields 
            H = y[self.H_ix]
            dydx[self.phis_ix] = 0
            dydx[self.phidots_ix] = 0
            #Do not save result of dH/dN at each step, see postprocess method
            dydx[self.H_ix] = 0
            dydx[self.rhogamma_ix] = -4*rhogamma
            dydx[self.rhomatter_ix] = -3*rhomatter
            #Perturbations without fields
            
            
        else: # Fields are on and need to be used
            pdotsq = np.sum(phidots**2, axis=0)
            active_transfers = self.transfers*self.transfers_on
            tgamma = active_transfers[:,self.tgamma_ix][...,np.newaxis]
            tmatter = active_transfers[:,self.tmatter_ix][...,np.newaxis]
            #Calculate H derivative now as we need it later
            Hdot = -((0.5*rhomatter + 2.0/3.0*rhogamma)/H + 0.5*H*pdotsq)
            #d\phi_0/dn = y_1
            dydx[self.phis_ix] = phidots
            #dphi^prime/dn
            dydx[self.phidots_ix] = -((3 + Hdot/H + 0.5/H * (tgamma + tmatter))*phidots 
                                       + dUdphi[...,np.newaxis]/(H**2))
            #dH/dn is not recorded, see postprocess method
            dydx[self.H_ix] = 0
            
            # Background Fluids
            dydx[self.rhogamma_ix] = -4*rhogamma + 0.5*H*np.sum(tgamma*phidots**2, axis=0)
            dydx[self.rhomatter_ix] = -3*rhomatter + 0.5*H*np.sum(tmatter*phidots**2, axis=0)
            
            #****************************
            # Perturbations
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
    

class ReheatingTwoStage(c.FOCanonicalTwoStage):
    """Run a first order simulation taking account of the times that transfer
    coefficients should be turned on, after the first time each field passes
    through its minimum.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize model"""
        super(ReheatingTwoStage, self).__init__(*args, **kwargs)
            
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
        
        
        #Run should reach last minima
        #FIXME Need to use normal end times
        self.fotend = max(self.fotend, np.max(self.minima_times))
        self.fotendindex = max(self.fotendindex, np.max(self.minima_indices))
        
        #Set initial conditions for first order model
        self.setfoics()
        
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()   
        
        if saveargs is None:
            saveargs = {}
        
        if saveresults:
            ystartshape = list(self.foystart.shape)
            ystartshape.insert(0, 0)
            saveargs["yresultshape"] = ystartshape 
            #Set up results file
            rf, grpname, filename, yresarr, tresarr = self.openresultsfile(**saveargs)
            self._log.info("Opened results file %s.", filename)
            resgrp = self.saveparamsinhdf5(rf, grpname)
            self._log.info("Saved parameters in file.")
        else:
            yresarr = None
            tresarr = None
            filename = None
        #Run first order model
        self.runfo(saveresults, yresarr, tresarr)
        
        #Save results in file
        if saveresults:
            try:
                self._log.info("Closing file")
                self.closehdf5file(rf)
            except IOError:
                self._log.exception("Error trying to close file! Results may not be saved.")        
        return filename
        
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
                  "nfields":self.nfields,
                  "transfers":self.transfers,
                  "transfers_on_times":self.transfers_on_times
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
        "nfields" : tables.IntCol(),
        "transfers":tables.Float64Col(np.shape(self.transfers)),
        "transfers_on_times":tables.Float64Col(np.shape(self.transfers_on_times))
        }
        return params