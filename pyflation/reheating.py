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
        
        if not any(rho_fraction<=fraction):
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
                module_logger.info("Fields turned off at time %f.", t)
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
        
        #Set number of fluids and length of variable array
        nfluids = 2 # only implemented for 2 fluids at the moment
        self.nvars = 2*self.nfields*(self.nfields + nfluids + 1) + nfluids + 1
        
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
        self.dgamma_ix = slice(self.dpdots_ix.stop, self.dpdots_ix.stop + self.nfields)
        self.dmatter_ix = slice(self.dgamma_ix.stop, self.dgamma_ix.stop + self.nfields)
        self.Vgamma_ix = slice(self.dmatter_ix.stop, self.dmatter_ix.stop + self.nfields)
        self.Vmatter_ix = slice(self.Vgamma_ix.stop, self.Vgamma_ix.stop + self.nfields)
        
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
        Vmatter = y[self.Vmatter_ix]
        Vgamma = y[self.Vgamma_ix]
        dgamma = y[self.dgamma_ix]
        dmatter = y[self.dmatter_ix]
        #Get a
        a = self.ainit*np.exp(t)
        H = y[self.H_ix]
        
        nfields = self.nfields    
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y[self.bg_ix,0], self.pot_params)[0:3]        
        
        #Set derivatives taking care of k type
        if type(k) is np.ndarray or type(k) is list: 
            dydx = np.zeros((self.nvars,lenk), dtype=y.dtype)
            innerterm = np.zeros((nfields,nfields,lenk), dtype=y.dtype)
        else:
            dydx = np.zeros(self.nvars, dtype=y.dtype)
            innerterm = np.zeros((nfields,nfields), y.dtype)
        
        if self.fields_off:
            #Calculate H derivative now as we need it later
            Hdot = -(0.5*rhomatter + 2.0/3.0*rhogamma)/H
            dydx[self.phis_ix] = 0
            dydx[self.phidots_ix] = 0
            #Do not save result of dH/dN at each step, see postprocess method
            dydx[self.H_ix] = 0
            dydx[self.rhogamma_ix] = -4*rhogamma
            dydx[self.rhomatter_ix] = -3*rhomatter
            #Perturbations without fields
            metric_phi = -1/(2*H) * (rhomatter*Vmatter + 4/3.0 * rhogamma*Vgamma)
            V_full = -2*H*metric_phi / (rhomatter + 4/3.0*rhogamma)
            drho_full = dmatter + dgamma
            
            #Vmatter and Vgamma perturbation equations
            dydx[self.Vmatter_ix] = -metric_phi/H
            dydx[self.Vgamma_ix] = Vgamma - metric_phi/H - dgamma/(4*H*rhogamma)
            
            #Metric_phi_dot
            metric_phi_dot = (-metric_phi*Hdot/H 
                              -1/(2*H)*(dydx[self.rhomatter_ix]*Vmatter 
                                        + rhomatter*dydx[self.Vmatter_ix])
                              -2/(3.0*H)*(dydx[self.rhogamma_ix]*Vgamma 
                                        + rhogamma*dydx[self.Vgamma_ix]))
            
            #Fluid perturbations
            dydx[self.dgamma_ix] = -(4*dgamma + 4*k**2/(3*H*a**2)*Vgamma*rhogamma
                                     -2/(3*H**2) * rhogamma*drho_full - 4*rhogamma*metric_phi)
            dydx[self.dmatter_ix] = -(3*dmatter + k**2/(H*a**2)*Vmatter*rhomatter
                                     -1/(2*H**2) * rhomatter*drho_full - 3*rhomatter*metric_phi)
            
            #Field perturbations
            dydx[self.dps_ix] = 0
            dydx[self.dpdots_ix] = 0
            
            
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
            dpdotmodes = y[self.dpdots_ix].reshape((nfields, nfields, lenk))
            
            # Set up metric phi, V and drho_full
            metric_phi = -0.5 * (1/H * (rhomatter*Vmatter + 4/3.0 * rhogamma*Vgamma)
                                 - np.sum(phidots[:,np.newaxis] * dpmodes, axis=0))
            V_full = -2*H*metric_phi / (rhomatter + 4/3.0*rhogamma + H**2*pdotsq)
            drho_full = (dmatter + dgamma 
                         + np.sum(H**2*phidots[:,np.newaxis]*dpdotmodes 
                                  -H**2*phidots[:,np.newaxis]**2*metric_phi
                                  -dUdphi[:,np.newaxis]*dpmodes, axis=0))
            
            #Vmatter and Vgamma perturbation equations
            dydx[self.Vmatter_ix] = (-metric_phi/H 
                                     -1/(2*rhomatter)*np.sum(H*tmatter*phidots**2, axis=0)*(
                                      Vmatter - V_full))
            dydx[self.Vgamma_ix] = (Vgamma - metric_phi/H - dgamma/(4*H*rhogamma)
                                    -1/(2*rhogamma)*np.sum(H*tgamma*phidots**2, axis=0)*(
                                      Vgamma - 0.75*V_full))
            
            #Metric_phi_dot
            metric_phi_dot = (-metric_phi*Hdot/H 
                              -1/(2*H)*(dydx[self.rhomatter_ix]*Vmatter 
                                        + rhomatter*dydx[self.Vmatter_ix])
                              -2/(3.0*H)*(dydx[self.rhogamma_ix]*Vgamma 
                                        + rhogamma*dydx[self.Vgamma_ix])
                              +0.5*np.sum((dydx[self.dpdots_ix][:,np.newaxis] 
                                         + Hdot/H*phidots[:,np.newaxis])*dpmodes
                                        + phidots[:,np.newaxis]*dpdotmodes, axis=0)
                              )
            
            #Fluid perturbations
            dydx[self.dgamma_ix] = (-4*dgamma - 4*k**2/(3*H*a**2)*Vgamma*rhogamma
                                     +2/(3*H**2) * rhogamma*drho_full + 4*rhogamma*metric_phi
                                     +np.sum(H*tgamma*(phidots[:,np.newaxis]*dpdotmodes
                                        -0.5*phidots[:,np.newaxis]**2*metric_phi[np.newaxis,:]), axis=0))
            
            dydx[self.dmatter_ix] = (-3*dmatter - k**2/(H*a**2)*Vmatter*rhomatter
                                     +1/(2*H**2) * rhomatter*drho_full + 3*rhomatter*metric_phi
                                     +np.sum(H*tmatter*(phidots[:,np.newaxis]*dpdotmodes
                                        -0.5*phidots[:,np.newaxis]**2*metric_phi[np.newaxis,:]), axis=0))
            #This for loop runs over i,j and does the inner summation over l
            for i in range(nfields):
                for j in range(nfields):
                    #Inner loop over fields
                    for l in range(nfields):
                        innerterm[i,j] += (d2Udphi2[i,l])*dpmodes[l,j]
            #Reshape this term so that it is nfields**2 long        
            innerterm = innerterm.reshape((nfields**2,lenk))
            #d\deltaphi_1^prime/dn
            dydx[self.dpdots_ix] = -((3+Hdot/H +1/(2*H)*(tgamma+tmatter))*y[self.dpdots_ix] 
                                     + (k/(a*H))**2 * y[self.dps_ix]
                                     + innerterm/H**2
                                     + (2/H**2 * dUdphi[:,np.newaxis] 
                                        + 0.5/H*phidots[:,np.newaxis]*(tgamma+tmatter)
                                        - 3*phidots[:,np.newaxis])*metric_phi[np.newaxis,:]
                                     - phidots[:,np.newaxis]*metric_phi_dot[np.newaxis,:]
                                     - 1/(2*H**2) * phidots[:,np.newaxis]*drho_full[np.newaxis,:]
                                     )
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
            y[self.dps_ix] = 0
            y[self.dpdots_ix] = 0
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
                module_logger.info("Fields turned off at time %f.", t)
                #Set fields at this timestep to be zero.
                y[self.phis_ix] = 0
                y[self.phidots_ix] = 0
                y[self.dps_ix] = 0
                y[self.dpdots_ix] = 0
        
        return y

class ReheatingTwoStage(c.FODriver):
    """Uses a background and firstorder class to run a full (first-order) reheating
        simulation.
        
        Main additional functionality is in determining initial conditions.
        Variables finally stored are as in first order class.
    """ 
                                                      
    def __init__(self, bgystart=None, tstart=0.0, tstartindex=None, tend=83.0, tstep_wanted=0.01,
                 k=None, ainit=None, solver="rkdriver_rkf45", bgclass=None, foclass=None, 
                 potential_func=None, pot_params=None, simtstart=0, nfields=1, 
                 transfers=None, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
      
        #Set number of fluids and length of variable array
        nfluids = 2 # only implemented for 2 fluids at the moment
        self.nvars = 2*nfields*(nfields + nfluids + 1) + nfluids + 1
        
        if transfers is None:
            self.transfers = np.zeros((nfields,nfluids))
        else:
            self.transfers = transfers
        
        #Initial conditions for each of the variables.
        if bgystart is None:
            self.bgystart = np.array([18.0/np.sqrt(nfields),-0.1/np.sqrt(nfields)]*nfields 
                                  + [0.0]*(nfluids+1))
        else:
            self.bgystart = bgystart
        #Lengthen bgystart to add perturbed fields.
        self.ystart= np.append(self.bgystart, ([0.0])*(2*nfields**2 + 2*nfluids*nfields))
            
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
        
        super(ReheatingTwoStage, self).__init__(**newkwargs)
        
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
            self.bgclass = ReheatingBackground
        else:
            self.bgclass = bgclass
        if foclass is None:
            self.foclass = ReheatingFirstOrder
        else:
            self.foclass = foclass
        
        #Setup model variables    
        self.bgmodel = self.firstordermodel = None
        return

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
        self.dgamma_ix = slice(self.dpdots_ix.stop, self.dpdots_ix.stop + self.nfields)
        self.dmatter_ix = slice(self.dgamma_ix.stop, self.dgamma_ix.stop + self.nfields)
        self.Vgamma_ix = slice(self.dmatter_ix.stop, self.dmatter_ix.stop + self.nfields)
        self.Vmatter_ix = slice(self.Vgamma_ix.stop, self.Vgamma_ix.stop + self.nfields)
        
        #Indices for transfer array
        self.tgamma_ix = 0
        self.tmatter_ix = 1
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
                      nfields=self.nfields,
                      transfers=self.transfers)
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
                      nfields=self.nfields,
                      transfers=self.transfers)
        return kwargs
    
    def find_ainit(self):
        """Find initial conditions for 1st order model
           Find a_end using the correct reheating temperature
        """
        #Find when reheating ended
        
        Hend = self.bgmodel.yresult[self.inflendindex, self.H_ix]
        Hreh = self.bgmodel.yresult[self.rehendindex, self.H_ix]
        self.a_end = self.finda_end(Hend, Hreh)
        self.ainit = self.a_end*np.exp(-self.bgmodel.tresult[self.inflendindex])
        return
        
    def find_fotend(self):
        """Set the end time of first order run.
            Find reheating time and set fotend to it"""
        self.reheating_end, self.rehendindex = self.bgmodel.find_reheating_end()
        self.fotend = self.reheating_end
        self.fotendindex = self.rehendindex
        return
        
    def getfoystart(self, ts=None, tsix=None):
        """Model dependent setting of ystart"""
        #Set variables in standard case:
        if ts is None or tsix is None:
            ts, tsix = self.fotstart, self.fotstartindex
            
        #Reset starting conditions at new time
        foystart = np.zeros((self.nvars, len(self.k)), dtype=np.complex128)
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
        
        #Set fluid perturbations to be zero
        foystart[self.dgamma_ix,:] = 0
        foystart[self.dmatter_ix,:] = 0
        
        #Set Vm and Vgamma to be zero
        foystart[self.Vgamma_ix,:] = 0
        foystart[self.Vmatter_ix,:] = 0
        
        return foystart
        
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