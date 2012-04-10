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
        self.dgamma_ix = slice(self.dpdots_ix.stop, self.dpdots_ix.stop + 2*self.nfields, 2)
        self.dgammadot_ix = slice(self.dpdots_ix.stop + 1, self.dpdots_ix.stop + 2*self.nfields, 2)
        self.dmatter_ix = slice(self.dgammadot_ix.stop, self.dgammadot_ix.stop + 2*self.nfields, 2)
        self.dmatterdot_ix = slice(self.dgammadot_ix.stop + 1, self.dgammadot_ix.stop + 2*self.nfields, 2)
        
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