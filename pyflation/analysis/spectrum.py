''' pyflation.analysis.spectrum - Explanation

Author: ith
'''

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
        #Get into mode matrix form, over first axis   
        mdp = self.getmodematrix(self.yresult, 1, self.dps_ix)
        #Take tensor product of modes and conjugate, summing over second mode
        #index.
        mPphi = np.zeros_like(mdp)
        #Do for loop as tensordot to memory expensive
        nfields=self.nfields
        for i in range(nfields):
            for j in range(nfields):
                for k in range(nfields):
                    mPphi[:,i,j] += mdp[:,i,k]*mdp[:,j,k].conj() 
        #Flatten back into vector form
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
