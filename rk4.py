#
#Runge-Kutta ODE solver
#Author: Ian Huston
#CVS: $Id: rk4.py,v 1.38 2010/01/18 16:57:02 ith Exp $
#
from __future__ import division # Get rid of integer division problems, i.e. 1/2=0

import numpy as np
import logging

from helpers import seq #Proper sequencing of floats
import helpers
from configuration import _debug

if not "profile" in __builtins__:
    def profile(f):
        return f

#Start logging
root_log_name = logging.getLogger().name
rk_log = logging.getLogger(root_log_name + "." + __name__)



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

def rk4stepxix(x, y, h, dargs, derivs):
    '''Do one step of the classical 4th order Runge Kutta method,
    starting from y at x with time step h and derivatives given by derivs'''
    
    hh = h*0.5 #Half time step
    h6 = h/6.0 #Sixth of time step
    xh = x + hh # Halfway point in x direction
            
    dydx = derivs(y, x, **dargs)
    #First step, we already have derivatives from dydx
    yt = y + hh*dydx
    
    if "tix" in dargs:
        dargs["tix"] += 1
    #Second step, get new derivatives
    dyt = derivs(yt, xh, **dargs)
    
    yt = y + hh*dyt
    
    #Third step
    dym = derivs(yt, xh, **dargs)
    
    yt = y + h*dym
    dym = dym + dyt
    
    if "tix" in dargs:
        dargs["tix"] += 1
    #Fourth step
    dyt = derivs(yt, x+h, **dargs)
    
    #Reset dargs["tix"]
    if "tix" in dargs:
        dargs["tix"] -= 2
        
    #Accumulate increments with proper weights
    yout = y + h6*(dydx + dyt + 2*dym)
    
    return yout

def rkdriver_tsix(ystart, simtstart, tsix, tend, allks, h, derivs):
    """Driver function for classical Runge Kutta 4th Order method.
    Uses indexes of starting time values instead of actual times.
    Indexes are number of steps of size h away from initial time simtstart."""
    #Make sure h is specified
    if h is None:
        raise SimRunError("Need to specify h.")
    
    #Set up x counter and index for x
    xix = 0 # first index
    
    #The number of steps could be either the floor or ceiling of the following calc
    #In the previous code, the floor was used, but then the rk step add another step on
    #Additional +1 is to match with extra step as x is incremented at beginning of loop
    number_steps = np.ceil((tend - simtstart)/h) + 1#floor might be needed for compatibility
    if np.any(tsix>number_steps):
        raise SimRunError("Start times outside range of steps.")
    
    #Set up x results array
    xarr = np.zeros((number_steps,))
    #Record first x value
    xarr[xix] = simtstart
        
    #Check whether ystart is one dimensional and change to at least two dimensions
    if ystart.ndim == 1:
        ystart = ystart[..., np.newaxis]
    v = np.ones_like(ystart)*np.nan
    
    #New y results array
    yshape = [number_steps]
    yshape.extend(v.shape)
    yarr = np.ones(yshape)*np.nan
    
    #Change yresults at each timestep in tsix to value in ystart
    #The transpose of ystart is used so that the start_value variable is an array
    #of all the dynamical variables at the start time given by timeindex.
    #Test whether the timeindex array has more than one value, i.e. more than one k value
    for kindex, (timeindex, start_value) in enumerate(zip(tsix, ystart.transpose())):
        yarr[timeindex, ..., kindex] = start_value
    
    for xix in range(1, number_steps):
        if _debug:
            rk_log.debug("rkdriver_tsix: xix=%f", xix)
        # xix labels the current timestep to be saved
        current_x = simtstart + xix*h
        #last_x is the timestep before, which we will need to use for calc
        last_x = simtstart + (xix-1)*h
        
        #Setup any arguments that are needed to send to derivs function
        dargs = {"k": allks}
        #Find first derivative term for the last time step
        dv = derivs(yarr[xix-1], last_x, **dargs)
        #Do a rk4 step starting from last time step
        v = rk4stepks(last_x, yarr[xix-1], h, dv, dargs, derivs)
        #This masks all the NaNs in the v result so that they are not copied
        v_nonan = ~np.isnan(v)
        
        #Save current timestep
        xarr[xix] = np.copy(current_x)
        #Save current result without overwriting with NaNs
        yarr[xix, v_nonan] = np.copy(v)[v_nonan]
    #Get results 
    
    return xarr, yarr
   
@profile 
def rkdriver_withks(vstart, simtstart, ts, te, allks, h, derivs):
    """Driver function for classical Runge Kutta 4th Order method. 
    Starting at x1 and proceeding to x2 in nstep number of steps.
    Copes with multiple start times for different ks if they are sorted in terms of starting time."""
    
    
    #Make sure h is specified
    if h is None:
        raise SimRunError("Need to specify h.")
   
    if allks is not None:
        #set_trace()
        if not isinstance(ts, np.ndarray):
                raise SimRunError("Need more than one start time for different k modes.")
        #Set up x counter and index for x
        xix = 0 # first index
        #The number of steps could be either the floor or ceiling of the following calc
        #In the previous code, the floor was used, but then the rk step add another step on
        #Additional +1 is to match with extra step as x is incremented at beginning of loop
        number_steps = np.ceil((te - simtstart)/h) + 1#floor might be needed for compatibility
        #set up x results array
        xarr = np.zeros((number_steps,))
        #Set up x results
        
        x1 = simtstart #Find simulation start time
        if not all(ts[ts.argsort()] == ts):
            raise SimRunError("ks not in order of start time.") #Sanity check
        
        #New array 
        xarr[xix] = x1
        
        #Set up start and end list for each section
        xslist = np.empty((len(ts)+1))
        xslist[0] = simtstart
        xslist[1:] = ts[:]
        xelist = np.empty((len(ts)+1)) #create empty array (which will be written over)
        xelist[:-1] = ts[:] - h #End list is one time step before next start time
        xelist[-1] = np.floor(te.copy()/h)*h # end time can only be in steps of size h
        v = np.ones_like(vstart)*np.nan
        
        #New y results array
        yshape = [number_steps]
        yshape.extend(v.shape)
        yarr = np.ones(yshape)*np.nan
        
        #First result is initial condition
        firstkix = np.where(x1>=ts)[0]
        for anix in firstkix:
            if np.any(np.isnan(v[:,anix])):
                v[:,anix] = vstart[:,anix]
        yarr[xix] = v.copy()
        #Need to start at different times for different k modes
        for xstart, xend in zip(xslist,xelist):
            #set_trace()
            #Set up initial values
            kix = np.where(xstart>=ts)[0]
            ks = allks[kix]
            if len(kix):
                kmax = kix.max()
                v[:,:kmax+1][np.isnan(v[:,:kmax+1])] = vstart[:,:kmax+1][np.isnan(v[:,:kmax+1])]
                #Change last y result to hold initial condition
                yarr[xix-1][:,kix] = v[:,kix]
            for x in seq(xstart, xend, h):
                xix += 1
                xarr[xix] = x.copy() + h
                if len(kix) > 0:
                    #Only complete if there is some k being calculated
                    dargs = {"k": ks}
                    dv = derivs(v[:,kix], x, **dargs)
                    v[:,kix] = rk4stepks(x, v[:,kix], h, dv, dargs, derivs)
                yarr[xix] = v.copy()
        #Get results 
        xx = xarr
        y = yarr
    else: #No ks to iterate over
        nstep = np.ceil((te-ts)/h).astype(int) #Total number of steps to take
        xx = np.zeros(nstep+1) #initialize 1-dim array for x
        xx[0] = x = ts # set both first xx and x to ts
        
        v = vstart
        y = [v.copy()] #start results list
        ks = None
        for step in xrange(nstep):
            dargs = {"k": ks}
            dv = derivs(v, x, **dargs)
            v = rk4stepks(x, v, h, dv, dargs, derivs)
            x = xx[step+1] = x + h
            y.append(v.copy())
        y = np.concatenate([y], 0) #very bad performance wise
    #Return results    
    return xx, y

def rkdriver_new(vstart, simtstart, ts, te, allks, h, derivs):
    """Driver function for classical Runge Kutta 4th Order method. 
    Starting at x1 and proceeding to x2 in nstep number of steps.
    Copes with multiple start times for different ks if they are sorted in terms of starting time."""
    #set_trace()
    
    #Make sure h is specified
    if h is None:
        raise SimRunError("Need to specify h.")
   
    if allks is not None:
        if not isinstance(ts, np.ndarray):
                raise SimRunError("Need more than one start time for different k modes.")
        #Set up x results
        
        x1 = simtstart #Find simulation start time
        if not all(ts[ts.argsort()] == ts):
            raise SimRunError("ks not in order of start time.") #Sanity check
        xx = []
        xx.append(x1) #Start x value
        
        #Set up start and end list for each section
        #ts = np.where(np.abs(ts%h - h) < h/10.0, ts, ts+h/2) #Put all start times on even steps
        ts = np.around(ts/h + h/2)*h
        #Need to remove duplicates
        tsnd = helpers.removedups(ts)
        xslist = np.empty((len(tsnd)+1))
        xslist[0] = simtstart
        xslist[1:] = tsnd[:]
        
        xelist = np.empty((len(tsnd)+1)) #create empty array (which will be written over)
        xelist[:-1] = tsnd[:] - h #End list is one time step before next start time
        # end time can only be in steps of size h one before end.
        xelist[-1] = np.floor(te.copy()/h + 0.1)*h - h# Fix bug#3
        xix = 0 #Index of x in first order tresult
        v = np.ones_like(vstart)*np.nan
        y = [] #start results list
        #First result is initial condition
        #Need to use ts for kix tests to get all indices
        firstkix = np.where(x1>=ts)[0]
        for anix in firstkix:
            if np.any(np.isnan(v[:,anix])):
                v[:,anix] = vstart[:,anix]
        y.append(v.copy()) #Add first result
#         xix+=2
                    
        #Need to start at different times for different k modes
        for xstart, xend in zip(xslist,xelist):
            #set_trace()
            #Set up initial values
            kix = np.where(xstart>=ts-h/2)[0]
            ks = allks[kix]
            for oneix in kix:
                if np.any(np.isnan(v[:,oneix])):
                    v[:,oneix] = vstart[:,oneix]
            #Change last y result to initial conditions
            y[-1][:,kix] = v[:,kix]
            if _debug:    
                rk_log.debug("rkdriver_new: xstart=%f, xend=%f", xstart, xend)
            for x in seq(xstart, xend, h):
                if _debug:
                    rk_log.debug("rkdriver_new: x=%f, xix=%d", x, xix)
                xx.append(x.copy() + h)
                if len(kix) != 0:
                    dargs = {"k": ks, "kix":kix, "tix":xix}
                    v[:,kix] = rk4stepxix(x, v[:,kix], h, dargs, derivs)
                xix+=2 #Increment x index counter
                y.append(v.copy())
        #Get results in right shape
        xx = np.array(xx)
        y = np.concatenate([y], 0)
    else: #No ks to iterate over
        nstep = np.ceil((te-ts)/h).astype(int) #Total number of steps to take
        xx = np.zeros(nstep+1) #initialize 1-dim array for x
        xx[0] = x = ts # set both first xx and x to ts
        
        v = vstart
        y = [v.copy()] #start results list
        ks = None
        for step in xrange(nstep):
            dargs = {"k": ks}
            dv = derivs(v, x, **dargs)
            v = rk4stepks(x, v, h, dv, dargs, derivs)
            x = xx[step+1] = x + h
            y.append(v.copy())
        y = np.concatenate([y], 0)
    #Return results    
    return xx, y
        


class SimRunError(StandardError):
    """Generic error for model simulating run. Attributes include current results stack."""
    pass

