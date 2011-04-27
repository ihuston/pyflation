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
                + y[self.phidots_ix,:,np.newaxis]*y[self.phidots_ix].T[np.newaxis,...]*U )*y[self.dps_ix].T
        
        #d\deltaphi_1^prime/dn  
        # Do sum over second field index so axis=-1
        dydx[self.dpdots_ix] = -(U * y[self.dpdots_ix]/H**2 + (k/(a*H))**2 * y[self.dps_ix]
                                + np.sum(term, axis=-1)/H**2) 
                
        return dydx
    
class MultiFieldTwoStage(c.MultiStageDriver):
    """Uses a background and firstorder class to run a full (first-order) simulation.
        Main additional functionality is in determining initial conditions.
        Variables finally stored are as in first order class.
    """
                                                  
    def __init__(self, ystart=None, tstart=0.0, tstartindex=None, tend=83.0, tstep_wanted=0.01,
                 k=None, ainit=None, solver="rkdriver_tsix", bgclass=None, foclass=None, 
                 potential_func=None, pot_params=None, simtstart=0, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
      
        #Initial conditions for each of the variables.
        if ystart is None:
            #Initial conditions for all variables
            self.ystart = np.array([18.0, # \phi_0
                                   -0.1, # \dot{\phi_0}
                                    0.0, # H - leave as 0.0 to let program determine
                                    1.0, # Re\delta\phi_1
                                    0.0, # Re\dot{\delta\phi_1}
                                    1.0, # Im\delta\phi_1
                                    0.0  # Im\dot{\delta\phi_1}
                                    ])
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
                         **kwargs)
        
        super(FOCanonicalTwoStage, self).__init__(**newkwargs)
        
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
        Hend = self.bgmodel.yresult[self.fotendindex,2]
        self.a_end = self.finda_end(Hend)
        self.ainit = self.a_end*np.exp(-self.bgmodel.tresult[self.fotendindex])
        
        
        #Find epsilon from bg model
        try:
            self.bgepsilon
        except AttributeError:            
            self.bgepsilon = self.bgmodel.getepsilon()
        #Set etainit, initial eta at n=0
        self.etainit = -1/(self.ainit*self.bgmodel.yresult[0,2]*(1-self.bgepsilon[0]))
        
        #find k crossing indices
        kcrossings = self.findallkcrossings(self.bgmodel.tresult[:self.fotendindex], 
                            self.bgmodel.yresult[:self.fotendindex,2])
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
            ys = self.ystart[0:3]
        elif len(self.ystart.shape) == 2:
            ys = self.ystart[0:3,0]
        #Choose tstartindex to be simply the first timestep.
        tstartindex = np.array([0])
        
        kwargs = dict(ystart=ys, 
                      tstart=self.tstart,
                      tstartindex=tstartindex, 
                      tend=self.tend,
                      tstep_wanted=self.tstep_wanted, 
                      solver=self.solver,
                      potential_func=self.potential_func, 
                      pot_params=self.pot_params)
         
        self.bgmodel = self.bgclass(**kwargs)
        #Start background run
        self._log.info("Running background model...")
        try:
            self.bgmodel.run(saveresults=False)
        except ModelError, er:
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
                      pot_params=self.pot_params)
        
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
        
        #Save results in resultlist and file
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        
        self.resultlist.append([self.lastparams, self.tresult, self.yresult])        
        
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
        foystart = np.zeros((len(self.ystart), len(self.k)))
        #set_trace()
        #Get values of needed variables at crossing time.
        astar = self.ainit*np.exp(ts)
        
        #Truncate bgmodel yresult down if there is an extra dimension
        if len(self.bgmodel.yresult.shape) > 2:
            bgyresult = self.bgmodel.yresult[..., 0]
        else:
            bgyresult = self.bgmodel.yresult
            
        Hstar = bgyresult[tsix,2]
        Hzero = bgyresult[0,2]
        
        epsstar = self.bgepsilon[tsix]
        etastar = -1/(astar*Hstar*(1-epsstar))
        try:
            etadiff = etastar - self.etainit
        except AttributeError:
            etadiff = etastar + 1/(self.ainit*Hzero*(1-self.bgepsilon[0]))
        keta = self.k*etadiff
        
        #Set bg init conditions based on previous bg evolution
        try:
            foystart[0:3] = bgyresult[tsix,:].transpose()
        except ValueError:
            foystart[0:3] = bgyresult[tsix,:][:, np.newaxis]
        
        #Find 1/asqrt(2k)
        arootk = 1/(astar*(np.sqrt(2*self.k)))
        #Find cos and sin(-keta)
        csketa = np.cos(-keta)
        snketa = np.sin(-keta)
        
        #Set Re\delta\phi_1 initial condition
        foystart[3,:] = csketa*arootk
        #set Re\dot\delta\phi_1 ic
        foystart[4,:] = -arootk*(csketa - (self.k/(astar*Hstar))*snketa)
        #Set Im\delta\phi_1
        foystart[5,:] = snketa*arootk
        #Set Im\dot\delta\phi_1
        foystart[6,:] = -arootk*((self.k/(astar*Hstar))*csketa + snketa)
        
        return foystart
    
    def getdeltaphi(self):
        return self.deltaphi
    
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
            self._deltaphi = self.yresult[:,3,:] + self.yresult[:,5,:]*1j
        return self._deltaphi