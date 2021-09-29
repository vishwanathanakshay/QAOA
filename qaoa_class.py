#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.optimize import *
import random
import multiprocessing as mp
import time
from IPython.display import clear_output
import sys 
import csv
import os
import math 

class QAOA:
    
    def __init__(self,depth,H):             # Class initialization. Arguments are "depth", 
                                            # and a Diagonal Hamiltonian,"H".    
        
        self.H = H
        self.n = int(np.log2(int(len(self.H)))) # Calculates the number of qubits. 
        
        #______________________________________________________________________________________________________
        self.X = self.new_mixerX()              # Executes a sequence of array manipulations to encapsulate the 
                                                # effect of standard one body driver hamiltonian, \Sum \sigma_x,
                                                # acting on any state.
        #______________________________________________________________________________________________________
        
        
        self.min = min(self.H)                  # Calculates minimum of the Hamiltonain, Ground state energy.
        
        self.deg = len(self.H[self.H == self.min]) # Calculates the degeneracy of Ground states. 
        self.p = depth                             # Standard qaoa depth written as "p".
        
        self.heruistic_LW_seed1 = 45
        self.heruistic_LW_seed2 = 25
        
        #______________________________________________________________________________________________________   
                    
                    # The sequence of array manipulations that return action of the driver,
                    # in terms of permutation indices.
    
    def new_mixerX(self):
        def split(x,k):
            return x.reshape((2**k,-1))
        def sym_swap(x):
            return np.asarray([x[-1],x[-2],x[1],x[0]])
        
        n = self.n
        x_list = []
        t1 = np.asarray([np.arange(2**(n-1),2**n),np.arange(0,2**(n-1))])
        t1 = t1.flatten()
        x_list.append(t1.flatten())
        t2 = t1.reshape(4,-1)
        t3 = sym_swap(t2)
        t1 = t3.flatten()
        x_list.append(t1)
        
        
        k = 1
        while k < (n-1):
            t2 = split(t1,k)
            t2 = np.asarray(t2)
            t1=[]
            for y in t2:
                t3 = y.reshape((4,-1))
                t4 = sym_swap(t3)
                t1.append(t4.flatten())
            t1 = np.asarray(t1)
            t1 = t1.flatten()
            x_list.append(t1)
            k+=1        
        
        return x_list
    #__________________________________________________________________________________________________________   
        
        
    def U_gamma(self,angle,state):       # applies exp{-i\gamma H_z}, here as "U_gamma", on a "state".
        
        t = -1j*angle
        state = state*np.exp(t*self.H.reshape(2**self.n,1))
        
        return state
            
    
    
    
    def V_beta(self,angle,state):        # applies exp{-i\beta H_x}, here as "V_beta", on a "state".
        c = np.cos(angle)
        s = np.sin(angle)
        
        for i in range(self.n):
            t = self.X[i]
            st = state[t]
            state = c*state + (-1j*s*st)
            
        return state
  
    #__________________________________________________________________________________________________________
    
                        # This step creates the qaoa_ansatz w.r.t to "angles" that are passed. 
                        # "angles" are passed as [gamma_1,gamma_2,...,gamma_p,beta_1,beta2,....beta_p].
    
    def qaoa_ansatz(self, angles):
        
        state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        p = int(len(angles)/2)
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state 
    
    #__________________________________________________________________________________________________________
    
                        # This step applies qaoa_ansatz structure w.r.t to "angles" and state vector that 
                        # are passed. 
                        # "angles" are passed as [gamma_1,gamma_2,...,gamma_p,beta_1,beta2,....beta_p].
    def apply_ansatz(self, angles,state):
        p = int(len(angles)/2)
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state
    

        
    
    #__________________________________________________________________________________________________________
    
    
    def expectation(self,angles):   # Calculates expected value of the Hamiltonian w.r.t qaoa_ansatz state,
                                    # defined by the specific choice of "angles".
        
        state = self.qaoa_ansatz(angles)
        
        ex = np.vdot(state,state*(self.H).reshape((2**self.n,1)))
        
        return np.real(ex)
            
    
    
    
    def overlap(self,state):        # Calculates ground state overlap for any "state",
                                    # passed to it. Usually the final state  or "f_state" returned, 
                                    # after optimization.
        
        g_ener = min(self.H)
        olap = 0
        for i in range(len(self.H)):
            if self.H[i] == g_ener:
                olap+= np.absolute(state[i])**2
        
        return olap
    
   #__________________________________________________________________________________________________________ 
    
    
                    # Main execution of the algorithm. 
                    # 1) Create "initial_angles", this would be the guess or  
                    #    starting point for the optimizer.
                    # 2) Optimizer "L-BFGS-B" then takes "initial_angles" and calls "expectation".
                    # 3) "expectation" then returns a number and the optimizer tries to minimize this,
                    #     by doing finite differences. Thermination returns optimized angles, 
                    #     stored here as "res.x".
                    # 4) Treating the optimized angles as being global minima for "expectation",
                    #    we calculate and store (as class attributes) the qaoa energy, here as "q_energy",
                    #    energy error, here as "q_error",
                    #    ground state overlap, here as "olap"
                    #    and also the optimal state, here as "f_state" 
    
    def run_RI(self):
        t_start = time.time()
        initial_angles=[]
        bds= [(0.1,2*np.pi)]*self.p + [(0.1,1*np.pi)]*self.p
        for i in range(2*self.p):
            if i < self.p:
                initial_angles.append(random.uniform(0,2*np.pi))
            else:
                initial_angles.append(random.uniform(0,np.pi))
            
        res = minimize(self.expectation,initial_angles,method='L-BFGS-B', jac=None, bounds=bds, options={'maxfun': 150000})
        
        t_end = time.time()
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(res.nfev)
        self.opt_angles = res.x
        self.q_energy = self.expectation(res.x)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(res.x)
        self.olap = self.overlap(self.f_state)[0]
        
     #__________________________________________________________________________________________________________
    
    
                    # Implements a Heruistic layerwise optimization strategy to reduce the computational time 
                    # in finding good initial angles as a quasi-optimal seed for a round of global optimization
    
    
    def run_heruistic_LW(self):
        
        initial_guess = lambda x: ([random.uniform(0,2*np.pi) for _ in range(x) ] +                                    [random.uniform(0,np.pi) for _ in range(x)])
        bds = lambda x: [(0.1,2*np.pi)]*x + [(0.1,1*np.pi)]*x
        
        def combine(a,b):

            a = list(a)
            b = list(b)
            a1 = a[0:int(len(a)/2)]
            a2 = a[int(len(a)/2)::]
            b1 = b[0:int(len(b)/2)]
            b2 = b[int(len(b)/2)::]
            a = a1+b1
            b = a2+b2
            
            return a + b 
        
        
        
        temp = [] 
        t_start = time.time()
        
        for _ in range(self.heruistic_LW_seed1):
            initial_guess_p1 = initial_guess(1)
            res = minimize(self.expectation,initial_guess_p1,method='L-BFGS-B', jac=None, bounds=bds(1), options={'maxfun': 10000})
            temp.append([self.expectation(res.x),initial_guess_p1])
            
        temp = np.asarray(temp,dtype=object)
        idx = np.argmin(temp[:,0])
        opt_angles = temp[idx][1]
       
        
        t_state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        
        
        while len(opt_angles) < 2*self.p:
            ts1 = time.time()
            t_state = self.qaoa_ansatz(opt_angles)
            
            
            ex = lambda x : np.real(np.vdot(self.apply_ansatz(x,t_state),                                    self.apply_ansatz(x,t_state)*(self.H).reshape((2**self.n,1))))
            temp = [] 
            
            for _ in range(self.heruistic_LW_seed2):
                
                res = minimize(ex,initial_guess(1),method='L-BFGS-B', jac=None, bounds=bds(1), options={'maxfun': 10000})
                temp.append([res.fun, res.x])
            temp = np.asarray(temp,dtype=object)
            idx = np.argmin(temp[:,0])
            lw_angles = temp[idx][1]
            opt_angles = combine(opt_angles,lw_angles)
            

            
            res = minimize(self.expectation,opt_angles,method='L-BFGS-B', jac=None, bounds=bds(int(len(opt_angles)/2)), options={'maxfun': 15000})    
            opt_angles = res.x
        self.opt_angles = opt_angles    
       
        

            
        t_end = time.time()
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(res.nfev)
        self.q_energy = self.expectation(self.opt_angles)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)[0]
