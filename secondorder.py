"""Second Order Cosmological Model simulations by Ian Huston
    $Id: secondorder.py,v 1.5 2008/11/14 13:46:12 ith Exp $
    
    Provides generic class CosmologicalModel that can be used as a base for explicit models."""

from cosmomodels import *
import numpy as N
from pdb import set_trace
import cmpotentials

class CanonicalSecondOrder(PhiModels):
    """First order model using efold as time variable.
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - H : Hubble parameter
       y[3] - \delta\varphi_1 : First order perturbation [Real Part]
       y[4] - \delta\varphi_1^\prime : Derivative of first order perturbation [Real Part]
       y[5] - \delta\varphi_1 : First order perturbation [Imag Part]
       y[6] - \delta\varphi_1^\prime : Derivative of first order perturbation [Imag Part]
       y[7] - \delta\varphi_2 : Second order perturbation [Real Part]
       y[8] - \delta\varphi_2^\prime : Derivative of second order perturbation [Real Part]
       y[9] - \delta\varphi_2 : Second order perturbation [Imag Part]
       y[10] - \delta\varphi_2^\prime : Derivative of second order perturbation [Imag Part]
       """
    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalSecondOrder, self).__init__(*args, **kwargs)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start if not given
        if k is None:
            self.k = 10**(N.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = N.array([15.0,-0.1,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
        
        #Set initial H value if None
        if N.all(self.ystart[2] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[2] = self.findH(U, self.ystart)
            
        #Text for graphs
        self.plottitle = "Complex First Order Malik Model in Efold time"
        self.tname = r"$n$"
        self.ynames = [r"$\varphi_0$",
                        r"$\dot{\varphi_0}$",
                        r"$H$",
                        r"Real $\delta\varphi_1$",
                        r"Real $\dot{\delta\varphi_1}$",
                        r"Imag $\delta\varphi_1$",
                        r"Imag $\dot{\delta\varphi_1}$",
                        r"Real $\delta\varphi_2$",
                        r"Real $\dot{\delta\varphi_2}$",
                        r"Imag $\delta\varphi_2$",
                        r"Imag $\dot{\delta\varphi_2}$"]
                    
    def derivs(self, y, t, k=None):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        #If k not given select all
        if k is None:
            k = self.k
            
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y, self.pot_params)        
        
        #Set derivatives taking care of k type
        if type(k) is N.ndarray or type(k) is list: 
            dydx = N.zeros((11,len(k)))
        else:
            dydx = N.zeros(11)
            
        
        #d\phi_0/dn = y_1
        dydx[0] = y[1] 
        
        #dphi^prime/dn
        dydx[1] = -(U*y[1] + dUdphi)/(y[2]**2)
        
        #dH/dn
        dydx[2] = -0.5*(y[1]**2)*y[2]
        
        #d\deltaphi_1/dn = y[4]
        dydx[3] = y[4]
        
        #Get a
        a = self.ainit*N.exp(t)
        
        #d\deltaphi_1^prime/dn  #
        dydx[4] = (-(3 + dydx[2]/y[2])*y[4] - ((k/(a*y[2]))**2)*y[3]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[3]/(y[2]**2)))
        #print dydx[4]
        
        #Complex parts
        dydx[5] = y[6]
        
        #
        dydx[6] = (-(3 + dydx[2]/y[2])*y[6]  - ((k/(a*y[2]))**2)*y[5]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[5]/(y[2]**2)))
        #
        #Second Order perturbations
        #
        #d\deltaphi_2/dn Real
        dydx[7] = y[8]
        
        #d\deltaphi_2^\prime/dn Real
        dydx[8] = (-(3 + dydx[2]/y[2])*y[8] - ((k/(a*y[2]))**2)*y[7]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[7]/(y[2]**2)))
        
        #d\deltaphi_2/dn Imag
        dydx[9] = y[10]
        
        #d\deltaphi_2^\prime/dn Imag
        dydx[10] = (-(3 + dydx[2]/y[2])*y[10] - ((k/(a*y[2]))**2)*y[9]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[9]/(y[2]**2)))
        
        return dydx
        
        
class SOCanonicalTwoStage(CanonicalTwoStage):
    """Implementation of Second Order Canonical two stage model with standard initial conditions for phi.
    """
                    
    def __init__(self, ystart=None, foclass=CanonicalSecondOrder, *args,**kwargs):
        """Initialize model and ensure initial conditions are sane."""
        
        #Initial conditions for each of the variables.
        if ystart is None:
            #Initial conditions for all variables
            self.ystart = ystart = N.array([18.0, # \phi_0
                                   -0.1, # \dot{\phi_0}
                                    0.0, # H - leave as 0.0 to let program determine
                                    1.0, # Re\delta\phi_1
                                    0.0, # Re\dot{\delta\phi_1}
                                    1.0, # Im\delta\phi_1
                                    0.0, # Im\dot{\delta\phi_1}
                                    0.0, # Re\delta\phi_2
                                    0.0, # Re\dot{\delta\phi_2}
                                    0.0, # Im\delta\phi_2
                                    0.0  # Im\dot{\delta\phi_2}
                                    ])
        else:
            self.ystart = ystart
        
        #Call superclass
        super(SOCanonicalTwoStage, self).__init__(ystart=ystart, foclass=foclass, *args, **kwargs)
        
    def getfoystart(self):
        """Model dependent setting of ystart"""
        #Reset starting conditions at new time
        foystart = N.zeros((len(self.ystart), len(self.k)))
        
        #set_trace()
        #Get values of needed variables at crossing time.
        astar = self.ainit*N.exp(self.fotstart)
        Hstar = self.bgmodel.yresult[self.fotstartindex,2]
        epsstar = self.bgepsilon[self.fotstartindex]
        etastar = -1/(astar*Hstar*(1-epsstar))
        etainit = -1/(self.ainit*self.bgmodel.yresult[0,2]*(1-self.bgepsilon[0]))
        etadiff = etastar - etainit
        keta = self.k*etadiff
        
        #Set bg init conditions based on previous bg evolution
        foystart[0:3] = self.bgmodel.yresult[self.fotstartindex,:].transpose()
        
        #Find 1/asqrt(2k)
        arootk = 1/(astar*(N.sqrt(2*self.k)))
        #Find cos and sin(-keta)
        csketa = N.cos(-keta)
        snketa = N.sin(-keta)
        
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
        """Return the total perturbation \delta\phi taking into 
            account up to second order perturbations.
           """
        #Raise error if first order not run yet
        self.checkfirstordercomplete()
        
        #Set nice variable names
        deltaphi1 = self.yresult[:,3,:] + self.yresult[:,5,:]*1j #complex deltaphi1
        deltaphi2 = self.yresult[:,7,:] + self.yresult[:,9,:]*1j #complex deltaphi2
        deltaphi = deltaphi1 + 0.5*deltaphi2
        return deltaphi
        