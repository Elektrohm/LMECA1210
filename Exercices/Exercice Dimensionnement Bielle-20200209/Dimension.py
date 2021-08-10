#Programme destiné à la résolution du travail sur le moteur, cours LMECA1210
#Réalisé par le groupe 12 : Arnaud Deckers, Théo Denis, Jonathan Dessy
#Version finale - 28/04/2020.

#https://www.ijert.org/research/design-and-analysis-of-connecting-rod-for-weight-and-stress-reduction-IJERTCONV7IS03008.pdf

#=================================Import===================================================

import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#=================================INPUTS===================================================

tau = 9 #@valeur taux compression@ [-]
D = 0.086 #@valeur alesage@ [m]
C = 0.094 #@valeur course@ [m] 
L = 0.141 #@valeur longueur bielle@ [m]
mpiston = 0.434 #@valeur masse piston@ [kg]
mbielle = 0.439 #@valeur masse bielle@ [kg]
Q = 2800*10**3 #@valeur chaleur emise par fuel par kg de melance admis@ #[J/kg_inlet gas] (ess=2800, die=1650)

#================================CONSTANTE=================================================

R = L/3                  #[m]
gam = 1.3                #[-]
M = 28.96                #[g/mol]
B = L/R                  #[-]
T = 303.15               #[K] 
p0 = 100000              #[Pa]
Vc = np.pi*D**2/4*2*R    #[m³]
m = p0*Vc*M/(8.3145*T)   #[g]  (doit être multiplié par s dans la suite du code)

#=================================Code=====================================================
#Calcul de la pression

def V(Theta):
    """
    Fonction décrivant la valeur du volume dans le cylindre en fonction de l'angle du vilebrequin
    @pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
    @post : retourne int correspondant à la valeur du volume 
    """
    v = Vc/2 * (1-np.cos(Theta)+B-np.sqrt(B**2-(np.sin(Theta))**2)) + 1/(tau-1)*Vc
    return v

def Chal(Theta,thetaC, deltaThetaC,s):
    """
    Fonction décrivant la valeur d'énergie de combustion apportée en fonction de l'angle du vilebrequin
    @pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
    @post : retourne int correspondant à la valeur de l'énergie de combustion
    """
    PI = np.pi
    if (Theta<-thetaC  or Theta>-thetaC+deltaThetaC):
        return 0
    q = s*m*(Q/10**3)/2*(1-np.cos(PI*(Theta-(-thetaC))/deltaThetaC))
    return q

def dV(Theta):
    """
    Fonction calculant la dérivée du volume en fonction de l'angle du vilebrequin
    @pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
    @post : retourne int correspondant à la valeur de la dérivée du volume 
    """ 
    dv =  Vc/2*np.sin(Theta)*(1+np.cos(Theta)/(np.sqrt((B**2)-(np.sin(Theta))**2)))
    return dv

def dQ(Theta,thetaC, deltaThetaC,s): 
    """
    Fonction calculant la dérivée de l'énergie de combustion en fonction de l'angle du vilebrequin
    @pre: - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
    @post : retourne int correspondant à la valeur de la dérivée de l'énergie de combustion
    """
    PI = np.pi
    if (Theta<-thetaC or Theta>-thetaC+deltaThetaC):
        return 0
    dq = (PI*(Q/10**3)*s*m/(2*deltaThetaC))*np.sin(PI*(Theta-(-thetaC))/deltaThetaC)
    return dq


def dP(p,Theta,thetaC, deltaThetaC,s):
    """
    Fonction calculant la dérivée de la pression à l'intérieur du cylindre en fonction de l'angle du vilebrequin
    @pre: - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
    @post: retourne int correspondant à la valeur de la dérivée de la pression dans le cylindre
    """
    v = V(Theta)
    dv = dV(Theta)
    dq = dQ(Theta,thetaC, deltaThetaC,s)
    dp = -gam*(p/v)*dv+(gam-1)*(1/v)*dq
    return dp



def Pression(Theta,thetaC, deltaThetaC,s):
    """
    Fonction calculant de la pression à l'intérieur du cylindre en fonction de l'angle du vilebrequin
    @pre: - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
    @post: retourne int correspondant à la valeur de la pression dans le cylindre
    """
    thetaC, deltaThetaC = thetaC*np.pi/180, deltaThetaC*np.pi/180
    p = odeint(dP, s*p0,Theta, args=(thetaC, deltaThetaC,s))[:, 0]
    return p


#Calcul des forces
def Fpied(Theta,thetaC,deltaThetaC,s,Omega):
    """
    Fonction calculant la force s'exerçant sur le pied de la bielle
    @pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
           - Omega, int, la vitesse angulaire
    @post : retourne int correspondant à la force exercée sur le pied de la bielle
    """
    PI = np.pi
    p = Pression(Theta,thetaC,deltaThetaC,s)
    F = (PI*D**2)/4*p-mpiston*R*Omega**2*np.cos(Theta*np.pi/180)
    return F

def Ftete(Theta,thetaC,deltaThetaC,s,Omega):
    """
    Fonction calculant la force s'exerçant sur le pied de la bielle
    @pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
           - Omega, int, la vitesse angulaire
    @post : retourne int correspondant à la force exercée sur le pied de la bielle
    """
    PI = np.pi
    p = Pression(Theta,thetaC,deltaThetaC,s)
    F = -(PI*D**2)/4*p + (mpiston+mbielle)*R*Omega**2*np.cos(Theta*np.pi/180)
    return F

def Fpression(Theta,thetaC,deltaThetaC,s,Omega):
    """
    Fonction calculant la force de pression s'exerçant sur la tête de la bielle
    @pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
           - Omega, int, la vitesse angulaire
    @post : retourne int correspondant à la force de pression exercée sur la tête de la bielle
    """
    PI = np.pi
    p = Pression(Theta,thetaC,deltaThetaC,s)
    F = (PI*D**2)/4*p
    return F

def Fbielle(Theta,thetaC,deltaThetaC,s,Omega):
    """
    Fonction calculant la force résultante exercée sur la bielle
    @pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
           - Omega, int, la vitesse angulaire
    @post : retourne int correspondant à la force résultante exercée sur la bielle
    """
    F = mbielle*R*Omega**2*np.cos(Theta*np.pi/180)
    return F

#Dimensionnement

def myfunc(rpm, s, thetaC, deltaThetaC):
    Theta = np.linspace(-np.pi,np.pi,1000) 
    E, Comp = 200*10**9, 450*10**6
    Omega = rpm*2*np.pi/60
    
    Ft = max(-Ftete(Theta,thetaC,deltaThetaC,s,Omega))
    Fpi= max(Fpied(Theta,thetaC,deltaThetaC,s,Omega))
    Fcrit = max(Ft,Fpi)
    
    #Pour Ixx
    a = 419/(12*Fcrit)
    b = -419/(132*Comp)
    c = -(L/np.pi)**2*1/E
    
    d = (b**2) - (4*a*c)

    sol1 = np.sqrt((-b+np.sqrt(d))/(2*a)) if d >= 0 else 0 
    
    #Pour Iyy
    a = 131/(12*Fcrit)
    b = -131/(132*Comp)
    c = -(L/np.pi)**2*1/(4*E)
    
    d = (b**2) - (4*a*c)
    
    sol2 = np.sqrt((-b+np.sqrt(d))/(2*a)) if d >= 0 else 0
    
    t = max(sol1,sol2)
    return t

Theta = np.linspace(-np.pi,np.pi,1000)
myfunc(2974,2.3,28,45)

