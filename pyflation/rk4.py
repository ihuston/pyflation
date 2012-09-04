"""rk4.py - Runge-Kutta ODE solver

Provides Runge-Kutta based ODE solvers for use with pyflation models.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division # Get rid of integer division problems, i.e. 1/2=0

import numpy as np
import logging

from configuration import _debug

#if not "profile" in __builtins__:
#    def profile(f):
#        return f

#Start logging
root_log_name = logging.getLogger().name
rk_log = logging.getLogger(root_log_name + "." + __name__)

#Constants for rkf45
rkc = np.array([0.25, #[0] for K2 
                3/8.0, 3/32.0, 9/32.0, #[1-3] for K3 
                12/13.0, 1932/2197.0, -7200/2197.0, 7296/2197.0, #[4-7] for K4
                439/216.0, -8.0, 3680/513.0, -845/4104.0, #[8-11] for K5
                0.5, -8/27.0, 2.0, -3544/2565.0, 1859/4104.0, -11/40.0, #[12-17] for K6
                1/360.0, -128/4275.0, -2197/75240.0, 1/50.0, 2/55.0, # [18-22] for R
                25/216.0, 1408/2565.0, 2197/4104.0, -0.2, # [23-26] for yout
                ])

#@profile
def rk4stepks(x, y, h, dargs, derivs, postprocess=None):
    '''Do one step of the classical 4th order Runge Kutta method,
    starting from y at x with time step h and derivatives given by derivs'''
    
    hh = h*0.5 #Half time step
    h6 = h/6.0 #Sixth of time step
    xh = x + hh # Halfway point in x direction
    
    dydx = derivs(y, x, **dargs)
    #First step, using dydx
    yt = y + hh*dydx
    
    #Second step, get new derivatives
    dyt = derivs(yt, xh, **dargs)
    
    yt = y + hh*dyt
    
    #Third step
    dym = derivs(yt, xh, **dargs)
    
    yt = y + h*dym
    dym = dym + dyt
    
    #Fourth step
    dyt = derivs(yt, x+h, **dargs)
    
    #Accumulate increments with proper weights
    yout = y + h6*(dydx + dyt + 2*dym)
    
    if postprocess is not None:
        #Allow post processing function to change y depending on y and x
        yout = postprocess(yout, x+h)
    
    return yout

def rkf45(x, y, h, dargs, derivs):
    '''Do one internal step of the Runge Kutta Fehlberg 4-5 method,
    starting from y at x with time step h and derivatives given by derivs.
    
    Parameters
    ----------
    x : float
        time to start from
        
    y : array_like
        array of y values to start from
        
    h : float
        step size to use in this step
        
    dargs : dict
            dictionary of extra arguments to pass to derivs function
            
    derivs : function
             derivatives function called using derivs(y, x, **dargs) signature
    
    Returns
    -------
    yout : array_like
           array of possible y values (not accepted yet)
           
    xout : float
           possible new x value (not accepted yet)
           
    R : array_like
        value of R variable to test against tolerance value
        
    
    Notes
    -----
    The Runge-Kutta-Fehlberg method used here is as outlined in Algorithm 5.3 of 
    "Numerical Analysis" by Burden & Faires, 3rd edition.
    '''
    
    K1 = h*derivs(y, x, **dargs)
    
    K2 = h*derivs(y + rkc[0]*K1, x + rkc[0]*h, **dargs)
    
    K3 = h*derivs(y + rkc[2]*K1 + rkc[3]*K2, x + rkc[1]*h, **dargs)
    
    K4 = h*derivs(y + rkc[5]*K1 + rkc[6]*K2 + rkc[7]*K3, x + rkc[4]*h, **dargs)
    
    K5 = h*derivs(y + rkc[8]*K1 + rkc[9]*K2 + rkc[10]*K3 + rkc[11]*K4, 
                  x + h, **dargs)
    
    K6 = h*derivs(y + rkc[13]*K1 + rkc[14]*K2 + rkc[15]*K3 + rkc[16]*K4 + rkc[17]*K5, 
                  x + rkc[12]*h, **dargs)
    
    #Calculate difference R
    R = np.abs(rkc[18]*K1 + rkc[19]*K3 + rkc[20]*K4 + rkc[21]*K5 + rkc[22]*K6)/h
    
    #Possible yout and xout which have yet to be accepted
    yout = y + rkc[23]*K1 + rkc[24]*K3 + rkc[25]*K4 + rkc[26]*K5
    xout = x + h
    
    
    
    return yout, xout, R

#@profile
def rkdriver_tsix(ystart, simtstart, tsix, tend, h, derivs, yarr, xarr, postprocess=None):
    """Driver function for classical Runge Kutta 4th Order method.
    Uses indexes of starting time values instead of actual times.
    Indexes are number of steps of size h away from initial time simtstart."""
    #Make sure h is specified
    if h is None:
        raise SimRunError("Need to specify h.")
    
    #Set up x counter and index for x
    xix = 0 # first index
    
    #The number of steps is now calculated using around. This matches with the
    #expression used in second order classes to calculate the first order timestep.
    #Around rounds .5 values towards even numbers so 0.5->0 and 1.5->2.
    #The added one is because the step at simtstart should also be counted.
    number_steps = np.int(np.around((tend - simtstart)/h) + 1)
    if np.any(tsix>number_steps):
        raise SimRunError("Start times outside range of steps.")
    
    # We do not use the given yarr and xarr variables but initialize new ones
    #Set up x results array
    xarr = np.zeros((number_steps,))
    #Record first x value
    xarr[xix] = simtstart
    
    first_real_step = np.int(tsix.min())
    if first_real_step > xix:
        if _debug:
            rk_log.debug("rkdriver_tsix: Storing x values for steps from %d to %d", xix+1, first_real_step+1)
        xarr[xix+1:first_real_step+1] = simtstart + np.arange(xix+1, first_real_step+1)*h
        xix = first_real_step
    
    #Get the last start step. Only need to check for NaNs before this.
    last_start_step = tsix.max()    
    
    #Check whether ystart is one dimensional and change to at least two dimensions
    if ystart.ndim == 1:
        ystart = ystart[..., np.newaxis]
    v = np.ones_like(ystart)*np.nan
    
    #New y results array
    yshape = [number_steps]
    yshape.extend(v.shape)
    yarr = np.ones(yshape, dtype=ystart.dtype)*np.nan
    
    #Change yresults at each timestep in tsix to value in ystart
    #The transpose of ystart is used so that the start_value variable is an array
    #of all the dynamical variables at the start time given by timeindex.
    #Test whether the timeindex array has more than one value, i.e. more than one k value
    for kindex, (timeindex, start_value) in enumerate(zip(tsix, ystart.transpose())):
        yarr[timeindex, ..., kindex] = start_value
    
    for xix in range(first_real_step + 1, number_steps):
        if _debug:
            rk_log.debug("rkdriver_tsix: xix=%f", xix)
        if xix % 1000 == 0:
            rk_log.info("Step number %i of %i", xix, number_steps)
        
        # xix labels the current timestep to be saved
        current_x = simtstart + xix*h
        #last_x is the timestep before, which we will need to use for calc
        last_x = simtstart + (xix-1)*h
        
        #Setup any arguments that are needed to send to derivs function
        dargs = {}
        #Do a rk4 step starting from last time step
        v = rk4stepks(last_x, yarr[xix-1], h, dargs, derivs, postprocess)
        #This masks all the NaNs in the v result so that they are not copied
        if xix <= last_start_step:
            v_nonan = ~np.isnan(v)
            #Save current result without overwriting with NaNs
            yarr[xix, v_nonan] = v[v_nonan]
        else:
            yarr[xix] = np.copy(v)
        #Save current timestep
        xarr[xix] = np.copy(current_x)
        
    #Get results 
    rk_log.info("Execution of Runge-Kutta method has finished.")
    return xarr, yarr
   

#@profile
def rkdriver_append(ystart, simtstart, tsix, tend, h, derivs, yarr, xarr, postprocess=None):
    """Driver function for classical Runge Kutta 4th Order method.
    Results for y and x are appended to the yarr and xarr variables to allow
    for buffering in the case of PyTables writing to disk.
     
    Uses indexes of starting time values instead of actual times.
    Indexes are number of steps of size h away from initial time simtstart."""
    #Make sure h is specified
    if h is None:
        raise SimRunError("Need to specify h.")
    
    #The number of steps is now calculated using around. This matches with the
    #expression used in second order classes to calculate the first order timestep.
    #Around rounds .5 values towards even numbers so 0.5->0 and 1.5->2.
    #The added one is because the step at simtstart should also be counted.
    number_steps = np.int(np.around((tend - simtstart)/h) + 1)
    if np.any(tsix>number_steps):
        raise SimRunError("Start times outside range of steps.")
    
    #Check whether ystart is one dimensional and change to at least two dimensions
    if ystart.ndim == 1:
        ystart = ystart[..., np.newaxis]
    v = np.ones_like(ystart)*np.nan
    
    #Set up x counter and index for x
    xix = 0 # first index
    #Record first x value
    xarr.append(np.atleast_1d(simtstart))
    
    first_real_step = np.int(tsix.min())
    if first_real_step > xix:
        if _debug:
            rk_log.debug("rkdriver_append: Storing x values for steps from %d to %d", xix+1, first_real_step+1)
        xarr.append(simtstart + np.arange(xix+1, first_real_step+1)*h)
        #Add in first_real_step ystart value
        #Need to append a result for step xix unlike in xarr case
        yarr.append(np.tile(v, (first_real_step - xix, 1, 1)))
        #Move pointer up to first_real_step
        xix = first_real_step
        
    #Save first set of y values
    ks_starting = np.where(tsix == xix)
    y_to_save = np.copy(v[np.newaxis])
    y_to_save[..., ks_starting] = ystart[..., ks_starting]
    yarr.append(y_to_save)
    last_y = y_to_save[0]
        
    # Go through all remaining timesteps
    for xix in range(first_real_step + 1, number_steps):
        if _debug:
            rk_log.debug("rkdriver_append: xix=%f", xix)
        if xix % 1000 == 0:
            rk_log.info("Step number %i of %i", xix, number_steps)
        
        # xix labels the current timestep to be saved
        current_x = simtstart + xix*h
        #last_x is the timestep before, which we will need to use for calc
        last_x = simtstart + (xix-1)*h
        
        #Setup any arguments that are needed to send to derivs function
        dargs = {}
        
        #Do a rk4 step starting from last time step
        v = rk4stepks(last_x, last_y, h, dargs, derivs, postprocess)
    
        #Check whether next time step has new ystart values
        ks_starting = np.where(tsix == xix)[0]
        y_to_save = np.copy(v[np.newaxis])
        if len(ks_starting) > 0:
            y_to_save[..., ks_starting] = ystart[..., ks_starting]
        yarr.append(y_to_save)
        #Save current timestep
        xarr.append(np.atleast_1d(current_x))
        #Save last y value but remove first axis
        last_y = y_to_save[0]
        
    #Get results 
    rk_log.info("Execution of Runge-Kutta method has finished.")
    return xarr, yarr

def rkdriver_rkf45(ystart, xstart, xend, h, derivs, yarr, xarr, 
                   hmax, hmin, abstol, reltol, postprocess=None):
    """Driver function for Runge Kutta Fehlberg 4-5 method.
    Results for y and x are appended to the yarr and xarr variables to allow
    for buffering in the case of PyTables writing to disk.
    Initial conditions must be at a single time for all k modes.
    
    Parameters
    ----------
    ystart : array_like
             Array of initial values in ystart for time xstart.
             
    xstart : float
             initial start time of this run
             
    xend : float
           end time of the run (included in result)
           
    h : float
        initial step size to use
        
    derivs : function
             derivatives function called using derivs(y, x, **dargs) signature
             
    yarr : list_like
           output for y variable, should implement append method
           
    xarr : list_like
           output for x variable, should implement append method
           
    hmax : float
           maximum allowed value of step size
           
    hmin : float
           minimum allowed value of step size, ValueError is raised if step
           size needs to be below this.
           
    abstol : float
             absolute tolerance value to be applied in acceptance of new y value
             
    reltol : float
             relative tolerance value to be applied in acceptance of new y value
             
    postprocess : function, optional
                  if specified this function with signature (y, x) is run after
                  a new y value has been accepted but before it is saved.
                  This allows any post-processing to be applied.
                  
    Returns
    -------
    xarr : list_like
           container for x results
           
    yarr : list_like
           container for y result
    
    Notes
    -----
    The Runge-Kutta-Fehlberg method used here is as outlined in Algorithm 5.3 of 
    "Numerical Analysis" by Burden & Faires, 3rd edition.
    """
    #Make sure h is specified
    if h is None:
        raise SimRunError("Need to specify h.")
    
    #Check whether ystart is one dimensional and change to at least two dimensions
    if ystart.ndim == 1:
        ystart = ystart[..., np.newaxis]
    
    
    #Record first x value
    xarr.append(np.atleast_1d(xstart))
    last_x = xstart
    xdiff = xend - xstart
    
    #Save first set of y values
    y_to_save = np.copy(ystart[np.newaxis])
    yarr.append(y_to_save)
    last_y = ystart
        
    # Go through all remaining timesteps
    while(last_x < xend):
        if _debug:
            rk_log.debug("rkdriver_rkf45: last_x=%f", last_x)
        if last_x % (xdiff/10) == 0:
            rk_log.info("Last saved time step %f", last_x)
        
        # Align stepsize with end of x range if needed
        h = min(h, xend-last_x)
        
        #Setup any arguments that are needed to send to derivs function
        dargs = {}
        
        #Do a rkf45 step starting from last time step
        yout, xout, R = rkf45(last_x, last_y, h, dargs, derivs)
        Rmax = R.max()
        #Get the tolerance in terms of abstol and reltol 
        tol = abstol + reltol*np.max([np.abs(yout), np.abs(last_y)], axis=0)
        
        if np.all(R <= tol):
            if postprocess:
                #Allow post processing function to change y depending on y and x
                yout = postprocess(yout, xout)
            # Approximation has been accepted
            y_to_save = np.copy(yout[np.newaxis])
            yarr.append(y_to_save)
            #Save current timestep
            xarr.append(np.atleast_1d(xout))
            #Save last values
            last_y = yout
            last_x = xout
        
        # Change timestep for next attempt
        delta = np.nanmin(0.84*(tol/R)**(0.25))
        if delta <= 0.1:
            h = 0.1*h
        elif delta >= 4:
            h = 4*h
        else:
            h = delta*h
        h = min(h, hmax)
        if h < hmin:
            rk_log.warn("Step size needed is smaller than minimum. Run halted!")
            return xarr, yarr
        
        
    #Get results 
    rk_log.info("Execution of Runge-Kutta method has finished.")
    return xarr, yarr

class SimRunError(StandardError):
    """Generic error for model simulating run. Attributes include current results stack."""
    pass

