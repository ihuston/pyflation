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

#local modules from pyflation
from configuration import _debug
import cmpotentials
import analysis
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
            rhogamma = self.yresult[:,self.rhogamma_ix,0]
            rhomatter = self.yresult[:,self.rhomatter_ix,0]
            Hsq = self.yresult[:,self.H_ix,0]**2
        else:
            rhogamma = self.yresult[:,self.rhogamma_ix]
            rhomatter = self.yresult[:,self.rhomatter_ix]
            Hsq = self.yresult[:,self.H_ix]**2
        #Make sure to do sum across only phidot axis (1 in this case)
        rho_fields = 3*Hsq - rhogamma - rhomatter
        return rho_fields, 3*Hsq
    
    def find_reheating_end(self):
        """Find the efold time where reheating ends,
            i.e. the energy density of inflaton fields < 1% of total.
            Returns tuple of endefold and endindex (in tresult)."""
        
        rho_fields, total_rho = self.getrho_fields()
        
        rho_fraction = rho_fields/total_rho
        
        if not any(self.epsilon>1):
            raise c.ModelError("Reheating did not end during specified number of efoldings.")
        
        endindex = np.where(rho_fraction<=0.01)[0][0]
        
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
        
                
        #get potential from function
        U, dUdphi = self.potentials(y, self.pot_params)[0:2]       
        
        #Set local variables
        phidots = y[self.phidots_ix]
        H = y[self.H_ix]
        rhogamma = y[self.rhogamma_ix]
        rhomatter = y[self.rhomatter_ix]
        tgamma = self.transfers[:,self.tgamma_ix][...,np.newaxis]
        tmatter = self.transfers[:,self.tmatter_ix][...,np.newaxis]
        
        #Calculate H derivative now as we need it later
        Hdot = -((0.5*rhomatter + 2.0/3.0*rhogamma)/H
                 + 0.5*H*np.sum(phidots**2,axis=0))
        
        #Set derivatives
        dydx = np.zeros_like(y)
        
        #d\phi_0/dn = y_1
        dydx[self.phis_ix] = phidots
        
        #dphi^prime/dn
        dydx[self.phidots_ix] = -((3 + Hdot/H + 0.5/H * (tgamma + tmatter))*phidots 
                                   + dUdphi[...,np.newaxis]/(H**2))
        
        #dH/dn
        dydx[self.H_ix] = Hdot
        
        # Fluids
        dydx[self.rhogamma_ix] = -4*rhogamma + 0.5*H*np.sum(tgamma*phidots**2, axis=0)
        
        dydx[self.rhomatter_ix] = -3*rhomatter + 0.5*H*np.sum(tmatter*phidots**2, axis=0)

        return dydx
