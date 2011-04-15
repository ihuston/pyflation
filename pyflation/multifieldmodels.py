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
        self.pert_ix = slice(self.nfields*2+1, None)
        
    def findH(self, U, y):
        """Return value of Hubble variable, H at y for given potential.
        
        y - array_like: variable array at one time step. So y[1] should be phidot
                        for the first field and y[n+1] for each subsequent field. 
        """
        #Get phidots from y, should be second variable for each field.
        phidots = y[self.bg_ix][1::2]
        
        
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
            Hdot = np.array(map(self.derivs, self.yresult, self.tresult))[:,self.H_ix,0]
            epsilon = - Hdot/self.yresult[:,self.H_ix,0]
        else:
            Hdot = np.array(map(self.derivs, self.yresult, self.tresult))[:,self.H_ix]
            epsilon = - Hdot/self.yresult[:,self.H_ix]
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
        U, dUdphi, d2Udphi2 = self.potentials(y, self.pot_params)[0:3]       
        
        #Set derivatives
        dydx = np.zeros_like(y)
        
        bg_indices = arange(0,self.H_ix,2)
        firstderiv_indices = arange(1,self.H_ix,2)
        
        #d\phi_0/dn = y_1
        dydx[bg_indices] = y[firstderiv_indices] 
        
        #dphi^prime/dn
        dydx[firstderiv_indices] = -(U*y[firstderiv_indices] + dUdphi)/(y[self.H_ix]**2)
        
        #dH/dn
        dydx[self.H_ix] = -0.5*(y[firstderiv_indices]**2)*y[self.H_ix]

        return dydx