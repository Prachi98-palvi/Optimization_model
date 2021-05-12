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

# input data
rijkh = pd.read_csv("r_ijkh_decision_var.csv")
master_data = pd.read_csv("r_ijkh_decision_var.csv")
# print(master_data)
site_list = master_data.loc[:, 'site']
site_list = site_list.unique()
OU_list = master_data.loc[:, 'OU']
OU_list = OU_list.unique()
Plangrp_list = master_data.loc[:, 'Plangrp']
Plangrp_list = Plangrp_list.unique()



master_data.set_index(['site', 'OU', 'Plangrp', 'shift'], inplace=True)
value_dict = master_data.to_dict()

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

flag_xpress_trial_version = 1   # Set 1 to run trial version; 0 to run commercial version
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

####- we get no values for key , hence my constr 3 and is not working
## myconstr6 is also having an empty list
for h in shift_list:
    for t in time_list:
        if Binary_dict.get((t, h)) is not None:
            print((t, h), '\t', Binary_dict.get(h,t))
        else:
            print('No values for key ', (t, h), '\t myconstr3 will not work')

for i in site_list:
    for j in OU_list:
        for k in Plangrp_list:
            for h in shift_list:
                if value_dict.get((i, j, k, h)) is not None:
                    print((i, j, k, h), '\t', value_dict.get((i, j, k, h)))
                else:
                    print((i, j, k, h), '\t')


myconstr3 = [xp.Sum(x[i, j, k, h] * Binary_dict.get(h, t) for j in OU_list for k in Plangrp_list for h in shift_list)
             == y[i, t] for i in site_list for t in time_list]

myconstr4 = []
for i in site_list:
    for j in OU_list:
        for k in Plangrp_list:
            for h in shift_list:
                if value_dict.get((i, j, k, h)) is not None:
                    myconstr4.append(
                        b * int(value_dict.get([i, j, k, h])) <= x[i, j, k, h] <= a * int(value_dict.get([i, j, k, h])))

myconstr5 = [W[i]==xp.Sum(x[i,j,k,h] for j in OU_list for k in Plangrp_list for h in shift_list) for i in site_list]

myconstr6 =[]
for i in site_list:
    if Max_seats_dict.get((i)) is not None:
        myconstr6.append(
            Z[i]<=int(Max_seats_dict.get([i])))
        
myconstr7=[x[i,j,k,h]>=0]
myconstr8=[y[i,t]>=0]
myconstr9=[Z[i]>=0]
myconstr10=[W[i]>=0]


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
