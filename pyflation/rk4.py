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


#@profile
def rk4stepks(x, y, h, dydx, dargs, derivs):
    '''Do one step of the classical 4th order Runge Kutta method,
    starting from y at x with time step h and derivatives given by derivs'''
    
    hh = h*0.5 #Half time step
    h6 = h/6.0 #Sixth of time step
    xh = x + hh # Halfway point in x direction
    
    #First step, we already have derivatives from dydx
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
    
    return yout

#@profile
def rkdriver_tsix(ystart, simtstart, tsix, tend, h, derivs, yarr, xarr):
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
        #Find first derivative term for the last time step
        dv = derivs(yarr[xix-1], last_x, **dargs)
        #Do a rk4 step starting from last time step
        v = rk4stepks(last_x, yarr[xix-1], h, dv, dargs, derivs)
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
def rkdriver_append(ystart, simtstart, tsix, tend, h, derivs, yarr, xarr):
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
        yarr.append(np.tile(v, (first_real_step - xix, 1)))
        #Move pointer up to first_real_step
        xix = first_real_step
        
    #Save first set of y values
    ks_starting = np.where(tsix == xix)
    y_to_save = np.copy(v)
    y_to_save[..., ks_starting] = ystart[..., ks_starting]
    yarr.append(y_to_save)
    last_y = y_to_save
        
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
        #Find first derivative term for the last time step
        dv = derivs(last_y, last_x, **dargs)
        #Do a rk4 step starting from last time step
        v = rk4stepks(last_x, last_y, h, dv, dargs, derivs)
    
        #Check whether next time step has new ystart values
        ks_starting = np.where(tsix == xix)
        y_to_save = np.copy(v)
        y_to_save[..., ks_starting] = ystart[..., ks_starting]
        yarr.append(y_to_save)
        #Save current timestep
        xarr.append(np.copy(current_x))
        #Save last y value
        last_y = y_to_save
        
    #Get results 
    rk_log.info("Execution of Runge-Kutta method has finished.")
    return xarr, yarr

class SimRunError(StandardError):
    """Generic error for model simulating run. Attributes include current results stack."""
    pass

