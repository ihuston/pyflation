#
#Runge-Kutta ODE solver
#Author: Ian Huston
#CVS: $Id: rk4.py,v 1.12 2008/07/09 15:58:27 ith Exp $
#

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import sys

#Constants
#Cash Karp coefficients for Runge Kutta.
CKP = {"A2":0.2, "A3":0.3, "A4":0.6, "A5":1.0, "A6":0.875,
                "B21":0.2, "B31":3.0/40.0, "B32":9.0/40.0, 
                "B41":0.3, "B42":-0.9, "B43":1.2, 
                "B51":-11.0/54.0, "B52":2.5, "B53":-70.0/27.0,
                "B54":35.0/27.0, "B61":1631.0/55296.0, "B62":175.0/512.0,
                "B63":575.0/13824.0, "B64":44275.0/110592.0, "B65":253.0/4096.0,
                "C1":37.0/378.0, "C3":250.0/621.0, "C4":125.0/594.0, "C6":512.0/1771.0,
                "DC1":(37.0/378.0-2825.0/27648.0), "DC3":(250.0/621.0-18575.0/48384.0),
                "DC4":(125.0/594.0-13525.0/55296.0), "DC5":-277.0/14336.0,
                "DC6":(512.0/1771.0-0.25)}
                
MAXSTP = 10000 #Maximum number of steps to take
TINY = 1.0e-30 #Tiny difference to add 

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

def rkck(y, dydx, x, h,derivs):
    """Take a Cash-Karp Runge-Kutta step."""
    
    global CKP
    
    #First step
    ytemp = y + CKP["B21"]*h*dydx
    
    #Second step
    ak2 = derivs(x+CKP["A2"]*h, ytemp)
    ytemp = y + h*(CKP["B31"]*dydx + CKP["B32"]*ak2)
    
    #Third step
    ak3 = derivs(x + CKP["A3"]*h, ytemp)
    ytemp = y + h*(CKP["B41"]*dydx + CKP["B42"]*ak2 + CKP["B43"]*ak3)
    
    #Fourth step
    ak4 = derivs(x + CKP["A4"]*h, ytemp)
    ytemp = y + h*(CKP["B51"]*dydx + CKP["B52"]*ak2 + CKP["B53"]*ak3 + CKP["B54"]*ak4)
    
    #Fifth step
    ak5 = derivs(x + CKP["A5"]*h, ytemp)
    ytemp = y + h*(CKP["B61"]*dydx + CKP["B62"]*ak2 + CKP["B63"]*ak3 + CKP["B64"]*ak4 + CKP["B65"]*ak5)
    
    #Sixth step
    ak6 = derivs(x + CKP["A6"]*h, ytemp)
    
    #Accumulate increments with proper weights.
    yout = y + h*(CKP["C1"]*dydx + CKP["C3"]*ak3 + CKP["C4"]*ak4 + CKP["C6"]*ak6)
    
    #Estimate error between fourth and fifth order methods
    yerr = h*(CKP["DC1"]*dydx + CKP["DC3"]*ak3 + CKP["DC4"]*ak4 + CKP["DC5"]*ak5 + CKP["DC6"]*ak6)
    
    return yout, yerr

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
    
def rkqs(y, dydx, x, htry, eps, yscal, derivs):
    """Takes one quality controlled RK step using rkck"""
    
    #Parameters for quality control
    SAFETY = 0.9
    PGROW = -0.2
    PSHRINK = -0.25
    ERRCON = 1.89e-4
    
    h = htry #Set initial stepsize
    
    while 1:
        ytemp, yerr = rkck(y, dydx, x, h, derivs) # Take a Cash-Karp Runge-Kutta step
        
        #Evaluate accuracy
        errmax = 0.0
        errmax = max(errmax, abs(yerr/yscal).max()) # Changed to version in C book
        
#         This does not work in all cases, e.g. single variables
        #for yerritem, yscalitem in zip(yerr, yscal):
        #    errmax = max(errmax, abs(yerritem/yscalitem).max())
        
        errmax = errmax/eps #Scale relative to required tolerance
        
        if errmax <= 1.0:
            break #Step succeeded. Compute size of next step
        
        htemp = SAFETY*h*(errmax**PSHRINK)
        #Truncation error too large, reduce step size
        #h = max(abs(htemp), 0.1*abs(h))*N.sign(h)#Old version from Fortran book
        if h >= 0.0:
            h = max(htemp, 0.1*h)
        else:
            h = min(htemp, 0.1*h)
            
        xnew = x + h
        assert xnew != x, "Stepsize underflow"
        #end of while loop
        
    if errmax > ERRCON:
        hnext = SAFETY*h*(errmax**PGROW)
    else:
        hnext = 5.0*h #No more than a factor 5 increase
    hdid = h
    x = x + h
    y = ytemp
    
    return x, y, hdid, hnext
        
def odeint(ystart, x1, x2, h1, hmin, derivs, eps=1.0e-6, dxsav=0.0):
    
    x = x1
    h = N.sign(x2-x1)*abs(h1)
    nok = nbad = 0
    
    xp = [] # Lists that will hold intermediate results
    yp = []
    
    y = ystart
    
    xsav = x - 2.0*dxsav #always save first step
    
    for i in xrange(MAXSTP):
        dydx = derivs(x, y)
        
        yscal = abs(y) + abs(h*dydx) + TINY
        
        if abs(x-xsav) >= abs(dxsav):
            xp.append(x)
            yp.append(y)
            xsav = x
        
        if (x + h - x2)*(x + h - x1) > 0.0:
            h = x2 - x
            
        x, y, hdid, hnext = rkqs(y, dydx, x, h, eps, yscal, derivs)
        if hdid == h:
            nok += 1
        else:
            nbad += 1
        
        #Are we done?
        if (x - x2)*(x2 -x1) >= 0.0:
            
            #Next line would set ystart to end value, but we are 
            #saving all steps anyway so not needed.
            #ystart = y 
            
            xp.append(x) #always save last step
            yp.append(y)
            
            #Want to return arrays, not lists so convert them
            xp = N.array(xp)
            yp = N.array(yp)
            return xp, yp, nok, nbad
        
        if abs(hnext) <= hmin:
            print >> sys.stderr, "Step size ", hnext, " smaller than ", hmin, " in odeint, trying again."
        
        h = hnext
        
    simerror = SimRunError("Too many steps taken in odeint!")
    simerror.tresult = xp
    simerror.yresult = yp
    raise simerror


class SimRunError(StandardError):
    """Generic error for model simulating run. Attributes include current results stack."""
    pass

