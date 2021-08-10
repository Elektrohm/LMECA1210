#Programme destiné à la résolution du travail sur le moteur, cours LMECA1210
#Réalisé par le groupe 12 : Arnaud Deckers, Théo Denis, Jonathan Dessy
#Version 3 - 21/03/2020.


#Source utilisée
#https://fr.wikipedia.org/wiki/Syst%C3%A8me_bielle-manivelle ; pour déterminer la longueur de la bielle et de la manivelle
#http://forum-auto.caradisiac.com/automobile-pratique/discussions-libres/sujet382592.htm  ; pour avoir une valeur de volume au PMB et PMH

import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#============================================================================================================================================
# DONNEEs DU PROBLEME 

#Constante dans les calculs : 
Qess = 2800                 #[J/g]           Quantité d'énergie apportée par la combustion pour un moteur essence
Qdie = 1650                 #[J/g]           Quantité d'énergie apportée par la combustion pour un moteur diesel
R = 0.10                    #[m]             Longueur de la manivelle (=moitié de la course du piston dans le cylindre)
L = 3*R                     #[m]             Longueur de la bielle
Dcomb = 50*np.pi/180        #[rad]           durée de la combustion, appartient à [40°;75°]
Angd = -30*np.pi/180        #[rad]           angle de dérmarrage de combustion, appartient à [15°;45°] avant PMH
gam = 1.3                   #[-]             coéfficient d'isentropie, égale au rapport des capacités thermiques à pression et volume constant.
M = 28.96                   #[g/mol]         masse atomique de l'air
mpist = 0.400               #[kg]            masse du piston
mbielle = 0.500             #[kg]            masse de la bielle
D = 0.1                     #[m]             diamètre de la tête du piston

#Relation utile :
B = L/R                     #[-]             rapport entre la longueur de la bielle et de la manivelle
Tau = 10                    #[-]             taux de compression, pour un moteur essence appartient à [8,13], pour un moteur diesel appartient à [16,25]
Vc = np.pi*D**2/4*2*R       #[m³]            le volume balayé par le piston lors d’une course complète (= cylindrée)

#Variable pour l'intégration :
p0 = 101325                 #[Pa]            pression initiale dans le moteur 
T = 300                     #[K]             température initiale dans le moteur
m = p0*Vc*M/(8.3145*T)      #[g]             masse d'air qui parcourt le cycl

#==============================================================================================================================================
# EQUATIONS CONSTITUTRICES
# Les fonctions ont été écrites pour un moteur essence, cependant, il suffit de changer les valeurs caractéristiques pour l'adapter au diesel


"""
Fonction décrivant la valeur du volume dans le cylindre en fonction de l'angle du vilebrequin
@pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
@post : retourne int correspondant à la valeur du volume 
"""
def V(Theta):
    v = Vc/2 * (1-np.cos(Theta)+B-np.sqrt(B**2-(np.sin(Theta))**2)) + 1/(Tau-1)*Vc
    return v


"""
Fonction décrivant la valeur d'énergie de combustion apportée en fonction de l'angle du vilebrequin
@pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
@post : retourne int correspondant à la valeur de l'énergie de combustion
"""
def Q(Theta):
    PI = np.pi
    if (Theta<Angd  or Theta>Angd+Dcomb):
        return 0
    q = m*Qess/2*(1-np.cos(PI*(Theta-Angd)/Dcomb))
    return q


"""
Fonction calculant la dérivée du volume en fonction de l'angle du vilebrequin
@pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
@post : retourne int correspondant à la valeur de la dérivée du volume 
"""
def dV(Theta):
    dv =  Vc/2*np.sin(Theta)*(1+np.cos(Theta)/(np.sqrt((B**2)-(np.sin(Theta))**2)))
    return dv

"""
Fonction calculant la dérivée de l'énergie de combustion en fonction de l'angle du vilebrequin
@pre: - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
@post : retourne int correspondant à la valeur de la dérivée de l'énergie de combustion
"""
def dQ(Theta):
    PI = np.pi
    if (Theta<Angd or Theta>Angd+Dcomb):
        return 0
    dq = (PI*Qess*m/(2*Dcomb))*np.sin(PI*(Theta-Angd)/Dcomb)
    return dq

"""
Fonction calculant la dérivée de la pression à l'intérieur du cylindre en fonction de l'angle du vilebrequin
@pre: - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
@post: retourne int correspondant à la valeur de la dérivée de la pression dans le cylindre
"""
def dP(p,Theta):
    v = V(Theta)
    dv = dV(Theta)
    dq = dQ(Theta)
    dp = -gam*(p/v)*dv+(gam-1)*(1/v)*dq
    return dp


"""
Fonction calculant de la pression à l'intérieur du cylindre en fonction de l'angle du vilebrequin
@pre: - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
@post: retourne int correspondant à la valeur de la pression dans le cylindre
"""
def P(Theta):
    th = np.linspace(-np.pi,np.pi,len(Theta)//2)
    sol = odeint(dP, p0,th)[:, 0]
    return sol


#=============================================================================================================
# Partie effort sur la bielle

"""
Fonction calculant la force s'exerçant sur le pied de la bielle
@pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
       - Omega, int, la vitesse angulaire
@post : retourne int correspondant à la force exercée sur le pied de la bielle
"""
def Fpied(Theta, Omega):
    PI = np.pi
    p = P(Theta)
    index = 0
    i = 0
    F = np.zeros(len(Theta))
    for th in Theta:
        if (th<-np.pi or th>np.pi):
            F[index] = (PI*D**2)/4*p0-mpist*R*Omega**2*np.cos(th)
        else:
            F[index] = (PI*D**2)/4*p[i]-mpist*R*Omega**2*np.cos(th)
            i+=1
        index+=1
    return F

"""
Fonction calculant la force s'exerçant sur le pied de la bielle
@pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
       - Omega, int, la vitesse angulaire
@post : retourne int correspondant à la force exercée sur le pied de la bielle
"""
def Ftete(Theta,Omega):
    PI = np.pi
    p = P(Theta)
    index = 0
    i = 0
    F = np.zeros(len(Theta))
    for th in Theta:
        if (th<=-np.pi or th>=np.pi):
            F[index] = -(PI*D**2)/4*p0 + (mpist+mbielle)*R*Omega**2*np.cos(th)
        else:
            F[index] = -(PI*D**2)/4*p[i] + (mpist+mbielle)*R*Omega**2*np.cos(th)
            i+=1
        index+=1
    return F


"""
Fonction calculant l'accélération s'exerçant sur la bielle
@pre : - Theta, int, est l'angle du vilebrequin, il vaut 180° au PMB et 0° au PMH
       - Omega, int, la vitesse angulaire
@post : retourne int correspondant à l'accélération de la bielle
"""
def acc(Theta, Omega):
    return R*Omega**2*np.cos(Theta)

#=========================================================================================================================
#
"""
Fonction qui affiche le résultat de l'intégration de la fonction dP/dTheta
"""
def plotFig(Theta):
    sol = odeint(dP, p0, Theta)[:, 0]
    
    p = np.zeros(2*len(sol))
    p[:len(sol)//2] = p0
    p[3*len(sol)//2:] = p0
    p[len(sol)//2:3*len(sol)//2]=sol
    th = np.linspace(-2*np.pi,2*np.pi,2*len(Theta))
    
    plt.plot(th*180/np.pi, p*10**-5, 'b', label=r'$P(\Theta)$')
    plt.title("Evolution de la pression en fonction de l'angle de vilebrequin")
    plt.legend(loc='best')
    plt.xlabel(r'$\Theta [°]$')
    plt.ylabel(r'$P(\Theta) [bar]$')
    plt.grid()
    plt.show()
    
def plotVol(Theta):
    """
    Fonction qui affiche l'évolution du volume dans le cylindre
    """
    v = np.zeros(len(Theta))
    for i in range(0,len(Theta)):
        v[i] = V(Theta[i])
    plt.plot(Theta,v*10**6, 'b', label=r'$V(\Theta)$')
    plt.title("Evolution du volume en fonction de l'angle de vilebrequin")
    plt.legend(loc='best')
    plt.xlabel(r'$\Theta$')
    plt.ylabel(r'$V(\Theta) [cm^3]$')
    plt.grid()
    plt.show()
    
"""
Fonction qui affiche l'évolution de la variation de l'énergie de combustion
"""
def plotChal(Theta):
    dq = np.zeros(len(Theta))
    for i in range(0,len(Theta)):
        dq[i] = dQ(Theta[i])
    plt.plot(Theta,dq, 'b', label=r'$dQ/d\Theta$')
    plt.title("Variation de l'énergie de combustion en fonction de l'angle de vilebrequin")
    plt.legend(loc='best')
    plt.xlabel(r'$\Theta$')
    plt.ylabel(r'$dQ/d\Theta [J/g.rad]$')
    plt.grid()
    plt.show()
    
"""
Fonction qui affiche le diagramme (P,V) du moteur
"""
def plotDiagPV(Theta):
    th = np.linspace(-2*np.pi,2*np.pi,2*len(Theta))
    v = V(th)
    sol = odeint(dP, p0, Theta)[:, 0]
    p = np.zeros(2*len(sol))
    p[:len(sol)//2] = p0
    p[3*len(sol)//2:] = p0
    p[len(sol)//2:3*len(sol)//2]=sol
    
    plt.plot(v*10**6,p*10**-5, 'b', label=r'$P(\Theta)$')
    plt.title("Diagramme P-V")
    plt.legend(loc='best')
    plt.xlabel(r'$V(\Theta) [cm^3]$')
    plt.ylabel(r'$P(\Theta) [bar]$')
    plt.grid()
    plt.show()
    
def plotEvol(Theta):
    """
    Fonction qui affiche les diagrammes de P,V,Q en fonction de l'angle de vilebrequin
    """ 
    fig, axs = plt.subplots(3)
    fig.suptitle(r'Variation de P,V,Q en fonction de $\Theta$')
    plt.subplots_adjust(hspace=0.8)
    
    v = V(Theta)
    p = odeint(dP, p0, Theta)
    q = np.zeros(len(Theta))
    for i in range(0,len(Theta)):
        q[i] = Q(Theta[i])
    
    #Graphe de la pression
    axs[0].plot(Theta, p[:, 0]*10**-5, 'b', label=r'$P(\Theta)$')
    axs[0].set_title("Pression")
    axs[0].legend(loc = "best")
    
    #Graphe du volume
    axs[1].plot(Theta,v*10**6, 'b', label=r'$V(\Theta)$')
    axs[1].set_title("Volume")
    axs[1].legend(loc = "best")
    
    #Graphe de l'énergie de combustion
    axs[2].plot(Theta,q, 'b', label=r'$Q(\Theta)$')
    axs[2].set_title("Energie de combustion")
    axs[2].legend(loc = "best")

    plt.xlabel(r'$\Theta$')
    
    plt.show()
    
def ForceBielle(Theta, rpm):
    Omega = rpm*2*np.pi/60
    fpied = Fpied(Theta,Omega)
    ftete = Ftete(Theta, Omega)
    
    sol = odeint(dP, p0, np.linspace(-np.pi, np.pi,500))[:, 0]
    
    p = np.zeros(2*len(sol))
    p[:len(sol)//2] = p0
    p[3*len(sol)//2:] = p0
    p[len(sol)//2:3*len(sol)//2]=sol
    
    plt.figure()
    plt.title("Schéma des forces à {} rpm".format(rpm))
    plt.plot(Theta*180/np.pi, fpied*10**-3, color='red',label="Force pied")
    plt.plot(Theta*180/np.pi, ftete*10**-3, color='black', label="Force tête")
    plt.plot(Theta*180/np.pi, p*np.pi*D**2/4*10**-3, 'b', label=r'$P(\Theta)$')
    plt.xlabel(r"Angle de vilebrequin $\Theta$ [°]")
    plt.ylabel("Force [kN]")
    plt.grid()
    plt.legend()

    plt.show()
    

Thet = np.linspace(-np.pi,np.pi,1000)
plotFig(Thet)
plotDiagPV(Thet)

th = np.linspace(-2*np.pi, 2*np.pi,1000)
ForceBielle(th,3000)
ForceBielle(th,10000)



    
    




