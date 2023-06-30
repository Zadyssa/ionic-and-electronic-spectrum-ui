# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:49:06 2020

Raymond Diab
"""
#Import all necessary modules
import numpy as np
from scipy.special.orthogonal import p_roots

# Computes the derivative of the plasma dispertion function between -15 and 15 
#and stores the data in z_prime.npy
#Used by the zprime functuon in thomson.py

Npoints= 30000 #number of points where to evaluate the integral
# 30000 --> precision of 10^-3
z_prime = np.zeros((2,Npoints),dtype = 'complex') #table to store the value pf z_prime

class elementary_rule:
    def __init__(self,x,w): # initialize weigths and nodes
        self.x = x
        self.w = w
    def Int(self, f, a, b): # computing the integral of from a to b
        # change of variable
        x = (a+b)/2 + (b-a)*self.x/2 
        w = (b-a) * self.w/2
        y = f(x)
        return np.sum(w*y)
    
class composite_rule:
    #Class for composite numerical integration (the interval is divided into smaller intervals)
    def __init__(self,Elem,M): # initialization
        self.Elem = Elem #elementary rule
        self.M = M #number of subdividing intervals
    def Int(self, f, a, b): #computing the integral using composite rule
        M = self.M
        x = np.linspace(a, b, M+1)
        # computation of the approximation
        I = 0
        for i in range(M):
            I += self.Elem.Int(f,x[i],x[i+1])
        return I
    
# function computing the nodes and weigths for gaussian quadrature rule of order n
def Gauss_quadrature(n):
    [x,w] = p_roots(n)
    return (x,w)

x,w = Gauss_quadrature(100) #gaussian quadrature of order 12
GaussElem = elementary_rule(x,w)
GaussComp = composite_rule(GaussElem,100)

phi = 1e-6
xmin = -15
xmax = 15
points = np.linspace(xmin,xmax,Npoints) #points where we compute the integral
for i in range(len(points)):
    zeta = points[i]
    def f(x):
        return np.exp(-x**2)/(x-zeta)
    NEG = GaussComp.Int(f,-2000,zeta-phi) #negative part of integral
    POS = GaussComp.Int(f,zeta+phi,2000) #positive part of integral
    POL = np.exp(-zeta**2)*(-1j*np.pi-4*zeta*phi) #contribution of the pole
    ITOT = NEG + POS + POL 
    zprime = (1+zeta*ITOT/np.sqrt(np.pi))
    z_prime[0,i] = zeta
    z_prime[1,i] = zprime
    if i%100 == 0: print(-2*z_prime[1,i])
np.save("z_prime.npy",z_prime)
    
