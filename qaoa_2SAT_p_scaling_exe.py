
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

from qaoa_class import QAOA


# In[3]:


def file_dump(line,name):
    
    with open(name,'a') as f:
        w=csv.writer(f,delimiter=',')
        w.writerow(line)


# In[4]:


def depth_scaling(instance_num,H):
    
    global p_c_start
    p_c = p_c_start
    check = False
    epsilon = 0.3
    seeds = 10

    
    while not check and p_c <= p_c_stop:
        
        Q = QAOA(p_c,H)
        seed_result = []
        
        for s in range(1,seeds +1):
            Q.run_heruistic_LW()
            temp = [Q.q_energy,Q.min,Q.q_error,Q.olap,Q.deg,p_c,s,instance_num,Q.exe_time]
            file_dump(temp,f_name1)
            seed_result.append(temp)
            temp = list(Q.opt_angles)
            temp.append(s)
            temp.append(instance_num)
            temp.append(p_c)
            file_dump(temp,f_name2)
            
        seed_result = np.asarray(seed_result)
        idx = np.argmin(seed_result[:,2])
        best = seed_result[idx]
        
        if best[2]<=epsilon:
            check = True
        else:
            p_c+= 1
    
    return best
        


# In[5]:


def performance(instance_num,H,p):
    
    seeds = 10
    
    Q = QAOA(p,H)
    tr = np.sum(H)
    H_gap = np.unique(H)
    gap = H_gap[1] - H_gap[0]
    
    seed_result = []
        
    for s in range(1,seeds + 1):
        Q.run_heruistic_LW()
        temp = [Q.q_energy,Q.min,Q.q_error,Q.olap,Q.deg,p,s,instance_num,Q.exe_time]
        file_dump(temp,f_name1)
        seed_result.append(temp)
        temp = list(Q.opt_angles)
        temp.append(s)
        temp.append(instance_num)
        file_dump(temp,f_name2)
            
    seed_result = np.asarray(seed_result)
    idx = np.argmin(seed_result[:,2])
    best = seed_result[idx]
    
    return best


# In[6]:


def run_depth_scaling(n, m, start, stop , workers):
    
    global p_c_start,p_c_stop
    p_c_start = start
    p_c_stop = stop
    
    dataset = np.loadtxt(path + f'instance_bank_2sat_{n}N_{m}M_depth_critical.csv',delimiter= ',')
    
    pool = mp.Pool(processes = workers)
    res = []
    r = [pool.apply_async(depth_scaling, args = (*(x[-1],x[:-1]),),callback = res.append) for x in dataset]
    
    while len(res) != len(dataset):
        ts1=len(res)
        time.sleep(0.0000000001)
        ts2=len(res)
        if ts1!=ts2:
            print(res[-1])
    pool.terminate()


# In[7]:


def run_performance(n, m, workers, depth):
    p = depth
    dataset = np.loadtxt(path + f'instance_bank_2sat_{n}N_{m}M_{depth}P.csv',delimiter= ',')
        
    pool=mp.Pool(processes = workers)
    res = []
    r = [pool.apply_async(performance, args = (*(x[-1],x[:-1],p),),callback = res.append) for x in dataset]
    
    while len(res) != len(dataset):
        ts1=len(res)
        time.sleep(0.0000000001)
        ts2=len(res)
        if ts1!=ts2:
            print(res[-1])
    pool.terminate()    


# In[ ]:


n = int(sys.argv[1])
m = int(sys.argv[2])
workers = int(sys.argv[3])
start_p = int(sys.argv[4])
stop_p = int(sys.argv[5])

global f_name1, f_name2
    
f_name1 = f'data_2_sat_{n}N_{m}M_depth_critical.csv'
f_name2 = f'data_2_sat_{n}N_{m}M_depth_critical_angles.csv'

path = '/home/akshay/Documents/Huawei_codes/Instances/'

run_depth_scaling(n, m, start_p, stop_p, workers)

