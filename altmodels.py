# -*- coding: iso-8859-15 -*-
""" alt-models.py Alternate cosmological models using cosmomodels.py classes.
Author: Ian Huston
$Id: altmodels.py,v 1.1 2009/02/05 20:43:27 ith Exp $
"""
import numpy as N
import cosmomodels as c

class OneZeroIcsTwoStage(c.TwoStageModel):
    """Implementation of First Order Canonical two stage model with standard initial conditions for phi.
    """
                    
    def __init__(self, *args, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
        #Call superclass
        super(OneZeroIcsTwoStage, self).__init__(*args, **kwargs)
        
    def getfoystart(self, ts=None, tsix=None):
        """Model dependent setting of ystart"""
        self._log.debug("Executing getfoystart to get initial conditions.")
        #Set variables in standard case:
        if ts is None or tsix is None:
            ts, tsix = self.fotstart, self.fotstartindex
            
        #Reset starting conditions at new time
        foystart = N.zeros((len(self.ystart), len(self.k)))
        
        #Set bg init conditions based on previous bg evolution
        try:
            foystart[0:3] = self.bgmodel.yresult[tsix,:].transpose()
        except ValueError:
            foystart[0:3] = self.bgmodel.yresult[tsix,:][:, N.newaxis]
        
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
        #Raise error if first order not run yet
        self.checkruncomplete()
        
        if not hasattr(self, "deltaphi") or recompute:
            self.deltaphi = self.yresult[:,3,:] + self.yresult[:,5,:]*1j #complex deltaphi
        return self.deltaphi 
        