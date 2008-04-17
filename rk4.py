#
#Runge-Kutta 4th order step
#Author: Ian Huston
#CVS: $Id: rk4.py,v 1.2 2008/04/17 19:04:19 ith Exp $
#

import numpy as N

def rk4step(x, y, h, dydx, derivs):
    '''Do one step of the classical 4th order Runge Kutta method,
    starting from y at x with time step h and derivatives given by derivs'''
    
    hh = h*0.5 #Half time step
    h6 = h/6.0 #Sixth of time step
    xh = x + hh # Halfway point in x direction
    
    #First step, we already have derivatives from dydx
    yt = y +hh*dydx
    
    #Second step, get new derivatives
    dyt = derivs(xh,yt)
    
    yt = y + hh*dyt
    
    #Third step
    dym = derivs(xh,yt)
    
    yt = y + h*dym
    dym = dym + dyt
    
    #Fourth step
    dyt = derivs(x+h, yt)
    
    #Accumulate increments with proper weights
    yout = y + h6*(dydx + dyt + 2*dym)
    
    return yout

def rkdriver_dumb(vstart, x1, x2, nstep, derivs):
    """Driver function for classical Runge Kutta 4th Order method. 
    Starting at x1 and proceeding to x2 in nstep number of steps."""
    
    v = vstart
    y = [v] #start results list
    xx = N.zeros(nstep+1) #initialize 1-dim array for x
    xx[0] = x = x1 # set both first xx and x to x1
    
    h = (x2-x1)/nstep
    
    for k in range(nstep):
        dv = derivs(x,v)
        v = rk4step(x, v, h, dv, derivs)
        x = xx[k+1] = x + h
        y.append(v)
    
    return xx, y
    
    
    