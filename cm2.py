#Cosmomodels2 - testing classes
import numpy as N
from ipdb import set_trace
from cosmomodels import *

class RingevalFirstOrder(MalikModels):
    """First order Malik-Wands model using efold as time variable.
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - H : Hubble parameter
       y[3] - \delta\varphi_1 : First order perturbation [Real Part]
       y[4] - \delta\varphi_1^\prime : Derivative of first order perturbation [Real Part]
       y[5] - \delta\varphi_1 : First order perturbation [Imag Part]
       y[6] - \delta\varphi_1^\prime : Derivative of first order perturbation [Imag Part]
       """
    def __init__(self, ystart=None, tstart=0.0, tend=80.0, tstep_wanted=0.01, tstep_min=0.0001, k=None, ainit=None, solver="scipy_odeint", mass=6.3267e-6):
        """Initialize all variables and call ancestor's __init__ method."""
        self.mass = mass
        super(RingevalFirstOrder, self).__init__(ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        
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
            self.ystart = N.array([15.0,-0.1,0.0,1.0,0.0,1.0,0.0])   
        
        #Set initial H value if None
        if N.all(self.ystart[2] == 0.0):
            U = self.potentials(self.ystart)[0]
            self.ystart[2] = self.findH(U, self.ystart)
            
        #Text for graphs
        self.plottitle = "Standard Ringeval evolution of real field in Efold time"
        self.tname = r"$n$"
        self.ynames = [r"$\varphi_0$",
                        r"$\dot{\varphi_0}$",
                        r"$H$",
                        r"$\delta\varphi_1$",
                        r"$\dot{\delta\varphi_1}$",
                        r"$\Psi$",
                        r"$\dot{\Psi}$"]
                    
    def derivs(self, y, t):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)        
        
        #Set derivatives taking care of k type
        if type(self.k) is N.ndarray or type(self.k) is list: 
            dydx = N.zeros((7,len(self.k)))
        else:
            dydx = N.zeros(7)
            
        
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
        
        #d\deltaphi_1^prime/dn
        dydx[4] = (-(3 + dydx[2]/y[2] + 2*y[1])*y[4] - ((self.k/(a*y[2]))**2)*y[3] 
                    -(d2Udphi2/(y[2]**2) + (y[1]**2))*y[3] + 4*y[6]*y[1] - 2*y[5]*dUdphi/(y[2]**2)   ) 
        
        #\Upsi
        dydx[5] = y[6]
        
        #d\upsi/dn
        dydx[6] = -(7 + dydx[2]/y[2])*y[6] - ( 2*U/(y[2]**2) + (k/(a*y[2]))**2 )*y[5] - dUdphi*y[3]/(y[2]**2)
        
        return dydx

class RingevalTwoStage(TwoStageModel):
    """Implementation of Ringeval two stage model with standard initial conditions for phi.
    """                
    def __init__(self, *args, **kwargs):
        """Initialize model and ensure initial conditions are sane."""
        #Call superclass
        super(RingevalTwoStage, self).__init__(*args, **kwargs)
              
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
    
    def findspectrum(self):
        """Find the spectrum of perturbations for each k. 
           Return Pr.
           """
        #Check if bg run is completed
        if self.firstordermodel.runcount == 0:
            raise ModelError("First order system must be run trying to find spectrum!")
        
        #Set nice variable names
        dphi = self.yresult[:,3,:] + self.yresult[:,5,:]*1j #complex dphi
        phidot = self.yresult[:,1,:] #bg phidot
        
        Pphi = (self.k**3/(2*N.pi**2))*(dphi*dphi.conj())
        Pr = Pphi/(phidot**2) #change if bg evol is different  
        return Pr
    
class MalikFirstOrder2(MalikModels):
    """First order model using efold as time variable.
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - H : Hubble parameter
       y[3] - \delta\varphi_1 : First order perturbation [Real Part]
       y[4] - \delta\varphi_1^\prime : Derivative of first order perturbation [Real Part]
       y[5] - \delta\varphi_1 : First order perturbation [Imag Part]
       y[6] - \delta\varphi_1^\prime : Derivative of first order perturbation [Imag Part]
       """
    def __init__(self, ystart=None, tstart=0.0, tend=80.0, tstep_wanted=0.01, tstep_min=0.0001, k=None, ainit=None, solver="scipy_odeint", mass=6.3267e-6):
        """Initialize all variables and call ancestor's __init__ method."""
        self.mass = mass
        super(MalikFirstOrder2, self).__init__(ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        
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
            self.ystart = N.array([15.0,-0.1,0.0,1.0,0.0,1.0,0.0])   
        
        #Set initial H value if None
        if N.all(self.ystart[2] == 0.0):
            U = self.potentials(self.ystart)[0]
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
                        r"Imag $\dot{\delta\varphi_1}$"]
                    
    def derivs(self, y, t):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)        
        
        #Set derivatives taking care of k type
        if type(self.k) is N.ndarray or type(self.k) is list: 
            dydx = N.zeros((7,len(self.k)))
        else:
            dydx = N.zeros(7)
            
        
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
        dydx[4] = (-(3 + dydx[2]/y[2])*y[4] - ((self.k/(a*y[2]))**2)*y[3]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[3]/(y[2]**2)))
        #print dydx[4]
        
        #Complex parts
        dydx[5] = y[6]
        
        #
        dydx[6] = (-(3 + dydx[2]/y[2])*y[6]  - ((self.k/(a*y[2]))**2)*y[5]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[5]/(y[2]**2)))
        
        return dydx