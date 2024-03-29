# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:11:29 2021

@author: prachpal
"""
import numpy as np
import pandas as pd
import xpress as xp
from xpress import *
from collections import Counter
import random

# input data and indexes
rijkh = pd.read_csv("r_ijkh_decision_var.csv") 
master_data = pd.read_csv("r_ijkh_decision_var.csv")
# print(master_data)
site_list = master_data.loc[:, 'site'] #extracted all sites in the list
site_list = site_list.unique() # made a list of unique sites
OU_list = master_data.loc[:, 'OU']
OU_list = OU_list.unique()
Plangrp_list = master_data.loc[:, 'Plangrp']
Plangrp_list = Plangrp_list.unique()



master_data.set_index(['site', 'OU', 'Plangrp', 'shift'], inplace=True)
value_dict = master_data.to_dict() #converted dataframe to dictionary as in format of X variable

Time_Shift_data = pd.read_csv("u_ht_binary_var.csv")
# print(Time_Shift_data)
time_list = Time_Shift_data.loc[:, 'time']
time_list = time_list.unique()

shift_list = Time_Shift_data.loc[:, 'shift']
shift_list = shift_list.unique()


Time_Shift_data.set_index(['time', 'shift'], inplace=True)
Binary_dict = Time_Shift_data.to_dict()

Current_SU = pd.read_csv("Current_SU.csv")
# print(Current_SU)
CurrentSU = Current_SU[['site', 'SU']]
# print(CurrentSU)
CurrentSU = CurrentSU.set_index(['site'])
SU_dict = CurrentSU.to_dict()
# print(SU_dict)

Max_Seats = Current_SU[['site', 'MaxCapacity']]
Max_seats = Max_Seats.set_index(['site'])
Max_seats_dict = Max_seats.to_dict()
# print(Max_seats_dict)


"""
Trial version used initially however it throws due to some missing variables
flag_xpress_trial_version = 0 # Set 1 to run trial version; 0 to run commercial version
#   Make small problem for trial version of Fico Xpress
if flag_xpress_trial_version:
    site_list = site_list[0:2]
    OU_list = OU_list[0:3]
    Plangrp_list = Plangrp_list[0:3]
    time_list = time_list[0:3]
    shift_list = shift_list[0:3]
    print('Running Xpress trial version')
    print('sites: ', len(site_list))
    print('OU: ', len(OU_list))
    print('Plangrp_list: ', len(Plangrp_list))
    print('time', len(time_list))
    print('shift', len(shift_list))
else:
    print('\n\nRunning Xpress commercial version\n\n')
    
"""

M0 = 10

# decision variable

x = {(i, j, k, h): xp.var(vartype=xp.integer, name="x_{0}_{1}_{2}_{3}".format(i, j, k, h))
     for j in OU_list for k in Plangrp_list for i in site_list for h in shift_list}

y = {(i, t): xp.var(vartype=xp.continuous, name="y_{0}_{1}".format(i, t))
     for i in site_list for t in time_list}

Z = {i: xp.var(vartype=xp.continuous, name="Z_{0}".format(i))
     for i in site_list}

W = {i: xp.var(vartype=xp.continuous, name="W_{0}".format(i))
     for i in site_list}

# parameters

a = 0.02
b = 0.02

model = xp.problem()
model = xp.problem(name="SU_Optimization")

model.addVariable(x, y, Z, W)

# adding constraints 

myconstr1 = (2 * Z[i] <= W[i] for i in site_list)

myconstr2 = (y[i, t] <= Z[i] for i in site_list for t in time_list)

""" below development comments can be ignored """
####- we get no values for key , hence my constr 3 and is not working
## myconstr6 is also having an empty list
# t=0.5
# h='0.5||9.5'
# "("+str(time_list[0])+", '"+shift_list[0]+"')"
# "("+str(t)+", '"+h+"')"

# loops to check mapping of indexes and values against it exist and can be called in constraint

for h in shift_list:
    for t in time_list:
        if (Binary_dict["value"][(t,h)]) is not None:
            print((t, h), '\t', (Binary_dict["value"][(t,h)]))
        else:
            print('No values for key ', (t, h), '\t myconstr3 will not work')
            
# print(Binary_dict["value"]["("+str(t)+", "+h+")"])
# print(Binary_dict["value"][(23.5, '0||9')])
# print(Binary_dict["value"][(t, h)])

for i in site_list:
    for j in OU_list:
        for k in Plangrp_list:
            for h in shift_list:
                if (i, j, k, h) in value_dict["Value"]:
                    if (value_dict["Value"][(i, j, k, h)]) is not None:
                        print((i, j, k, h), '\t', (value_dict["Value"][(i, j, k, h)]))
                    else:
                        print((i, j, k, h), '\t')
                else:
                    pass
                    
# i=('CJB10-Coimbatore') 
# j=('IN') 
# k=('CS SUPPORT')
# h=('21||6')

# #this code is working now to implemented this in loop above                  
#print(value_dict["Value"][('CJB10-Coimbatore', 'IN', 'CS SUPPORT', '21||6')])
# print(value_dict["Value"][(i,j,k,h)]) #working 
                    


myconstr3 = [xp.Sum(x[i, j, k, h] * Binary_dict["value"][(t, h)] for j in OU_list for k in Plangrp_list for h in shift_list)
             == y[i, t] for i in site_list for t in time_list]

## - myconstr 4 and mycontr6 are not running properly - resolved - append did no work, had to specifically mention that this is xp.constraint

myconstr4 = []
for i in site_list:
    for j in OU_list:
        for k in Plangrp_list:
            for h in shift_list:
                if (i, j, k, h) in value_dict["Value"]:
                    if (value_dict["Value"][(i, j, k, h)]) is not None:
                        # value1=b * int(value_dict["Value"][(i, j, k, h)])
                        # value2=a * int(value_dict["Value"][(i, j, k, h)])
                        # myconstr4.append(random.uniform(value1,value2))
                        myconstr4.append(
                            xp.constraint((1-b) * int(value_dict["Value"][(i, j, k, h)]) <= x[i, j, k, h] <= (1-a) * int(value_dict["Value"][(i, j, k, h)])))
                else:
                    pass
                
myconstr5 = [W[i]==xp.Sum(x[i,j,k,h] for j in OU_list for k in Plangrp_list for h in shift_list) for i in site_list]

myconstr6 =[]
for i in site_list:
    if i in Max_seats_dict["MaxCapacity"]:
        if (Max_seats_dict["MaxCapacity"][(i)]) is not None:
            myconstr6.append(
                xp.constraint(Z[i]<=Max_seats_dict["MaxCapacity"][(i)]))
    else:
        pass
        
# print(Max_seats_dict["MaxCapacity"]['CJB10-Coimbatore'])
# print(Max_seats_dict)
# print(value_dict)        
# print(Binary_dict)
# i="CJB10-Coimbatore"
# print(Max_seats_dict["Value"](i))

        
myconstr7=[x[i,j,k,h]>=0 for i in site_list for j in OU_list for k in Plangrp_list for h in shift_list]
myconstr8=[y[i,t]>=0 for i in site_list for t in time_list]
myconstr9=[Z[i]>=0 for i in site_list]
myconstr10=[W[i]>=0 for i in site_list]


model.addConstraint(myconstr1, myconstr2, myconstr3, myconstr4,myconstr5,myconstr6,myconstr7,myconstr8,myconstr9,myconstr10)

model.setObjective(xp.Sum([Z[i] for i in site_list]), sense=xp.minimize)
model.write("example0", "lp")
model.solve()
print("objective value:", model.getObjVal())
print("solution:", model.getSolution())

'''
# Multicommodity flow example.
#
# (C) Fair Isaac Corp., 1983-2020

import xpress as xp
import numpy as np
import math

# Random network generation

n = 3 + math.ceil(10 * np.random.random())  # number of nodes

thres = 0.4     # density of network
thresdem = 0.8  # density of demand mesh

# generate random forward stars for each node

fwstars = {}

for i in range(n):
    fwstar = []
    for j in range(n):
        if j != i:
            if np.random.random() < thres:
                fwstar.append(j)
    fwstars[i] = fwstar

# backward stars are generated based on the forward stars

bwstars = {i: [] for i in range(n)}

for j in fwstars.keys():
    for i in fwstars[j]:
        bwstars[i].append(j)

# Create arc array

arcs = []
for i in range(n):
    for j in fwstars[i]:
        arcs.append((i, j))

# Create random demand between node pairs

dem = []

for i in range(n):
    for j in range(n):
        if i != j and np.random.random() < thresdem:
            dem.append((i, j, math.ceil(200*np.random.random())))

# U is the unit capacity of each edge
U = 1000
# edge cost
c = {(i, j): math.ceil(10 * np.random.random()) for (i, j) in arcs}

# flow variables
f = {(i, j, d): xp.var(name='f_{0}_{1}_{2}_{3}'.format(i, j, dem[d][0],
                                                       dem[d][1]))
     for (i, j) in arcs for d in range(len(dem))}

# capacity variables
x = {(i, j): xp.var(vartype=xp.integer, name='cap_{0}_{1}'.format(i, j))
     for (i, j) in arcs}

p = xp.problem()
p.addVariable(f, x)


def demand(i, d):
    if dem[d][0] == i:  # source
        return 1
    elif dem[d][1] == i:  # destination
        return -1
    else:
        return 0


# Flow conservation constraints: total flow balance at node i for each demand d
# must be 0 if i is an intermediate node, 1 if i is the source of demand d, and
# -1 if i is the destination.

flow = {(i, d):
        xp.constraint(constraint=xp.Sum(f[i, j, d]
                                        for j in range(n) if (i, j) in arcs) -
                      xp.Sum(f[j, i, d] for j in range(n) if (j, i) in arcs)
                      == demand(i, d),
                      name='cons_{0}_{1}_{2}'.format(i, dem[d][0], dem[d][1]))
        for d in range(len(dem)) for i in range(n)}

# Capacity constraints: weighted sum of flow variables must be contained in the
# total capacity installed on the arc (i, j)
capacity = {(i, j):
            xp.constraint(constraint=xp.Sum(dem[d][2] * f[i, j, d]
                                            for d in range(len(dem)))
                          <= U * x[i, j],
                          name='capacity_{0}_{1}'.format(i, j))
            for (i, j) in arcs}

p.addConstraint(flow, capacity)

p.setObjective(xp.Sum(c[i, j] * x[i, j] for (i, j) in arcs))

# Compact declaration:
#
# p = xp.problem(f, x, flow, capacity,
#                xp.Sum(c[i, j] * x[i, j] for (i, j) in arcs))

p.solve()

'''
