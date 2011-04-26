'''multifield.py - Classes for multi field cosmological models

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''

import numpy as np
from scipy import interpolate

import cosmomodels as c

class MultiFieldModels(c.CosmologicalModel):
    '''
    Parent class for all multifield models. 
    '''


    def __init__(self, *args, **kwargs):
        """Call superclass init method."""
        super(MultiFieldModels, self).__init__(*args, **kwargs)
        
        #Set the number of fields using keyword argument, defaults to 1.
        self.nfields = kwargs.get("nfields", 1)
        
        #Set field indices. These can be used to select only certain parts of
        #the y variable, e.g. y[self.bg_ix] is the array of background values.
        self.H_ix = self.nfields*2
        self.bg_ix = slice(0,self.nfields*2)
        self.phis_ix = slice(0,self.nfields*2,2)
        self.phidots_ix = slice(1,self.nfields*2,2)
        self.pert_ix = slice(self.nfields*2+1, None)
        self.dps_ix = slice(self.nfields*2+1, None, 2)
        self.dpdots_ix = slice(self.nfields*2+2, None, 2)
        
    def findH(self, U, y):
        """Return value of Hubble variable, H at y for given potential.
        
        y - array_like: variable array at one time step. So y[1] should be phidot
                        for the first field and y[n+1] for each subsequent field. 
        """
        #Get phidots from y, should be second variable for each field.
        phidots = y[self.phidots_ix]
        
        
        #Expression for H
        H = np.sqrt(U/(3.0-0.5*(np.sum(phidots**2))))
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
            raise c.ModelError("Inflation did not end during specified number of efoldings. Increase tend and try again!")
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
        
class MultiFieldBackground(MultiFieldModels):
    """Basic model with background equations for multiple fields
        Array of dependent variables y is given by:
        
       y[0] - \phi_a : Background inflaton
       y[1] - d\phi_a/d\n : First deriv of \phi_a
       ...
       y[self.nfields*2] - H: Hubble parameter
    """
        
    def __init__(self,  *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(MultiFieldBackground, self).__init__(*args, **kwargs)
        
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
        dydx[self.phidots_ix] = -(U*y[self.phidots_ix] + dUdphi)/(y[self.H_ix]**2)
        
        #dH/dn
        dydx[self.H_ix] = -0.5*(np.sum(y[self.phidots_ix]**2))*y[self.H_ix]

        return dydx
    
class MultiFieldFirstOrder(MultiFieldModels):
    """First order model using efold as time variable with multiple fields.
    
    nfields holds the number of fields and the yresult variable is then laid
    out as follows:
    
    yresult[0:nfields*2] : background fields and derivatives
    yresult[nfields*2] : Hubble variable H
    yresult[nfields*2 + 1:] : perturbation fields and derivatives
       """

    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(MultiFieldFirstOrder, self).__init__(*args, **kwargs)
        
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
        
        #Set initial H value if None
        if np.all(self.ystart[self.H_ix] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[self.H_ix] = self.findH(U, self.ystart)
                        
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
            dydx = np.zeros((4*self.nfields + 1,len(k)), dtype=y.dtype)
        else:
            dydx = np.zeros(4*self.nfields + 1, dtype=y.dtype)
            
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
        term = (d2Udphi2[:,np.newaxis,:] 
                + y[self.phidots_ix,:,np.newaxis]*dUdphi 
                + dUdphi * (y[self.phidots_ix].T[np.newaxis,...])
                + y[self.phidots_ix,:,np.newaxis]*y[self.phidots_ix].T[np.newaxis,...]*U )
        
        #d\deltaphi_1^prime/dn  
        # Do sum over second field index so axis=-1
        dydx[self.dpdots_ix] = -(U * y[self.dpdots_ix]/H**2 + (k/(a*H))**2 * y[self.dps_ix]
                                + np.sum(term, axis=-1)) 
                
        return dydx