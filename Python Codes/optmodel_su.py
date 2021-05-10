# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:11:29 2021

@author: prachpal
"""

import numpy as np
import pandas as pd
import xpress as xp
from xpress import *

# input data

rijkh=pd.read_csv("W:/My Documents/Project - Seat Utilization/SU project - documents/CSV_inputs/r_ijkh_decision_var.csv")
ri=rijkh.set_index(['site','OU','Plangrp','shift'],inplace=True)
r=rijkh.to_dict()


uht=pd.read_csv("W:/My Documents/Project - Seat Utilization/SU project - documents/CSV_inputs/u_ht_binary_var.csv")
uh=uht.set_index(['shift','time'],inplace=True)
u=uht.to_dict()
S=pd.read_csv("W:/My Documents/Project - Seat Utilization/SU project - documents/CSV_inputs/Current_SU.csv")

current_su=S[["site","SU"]]
max_seats=S[["site","MaxCapacity"]]
M0=10

#indexes


site=rijkh["site"].unique()
site=site.tolist()
#site
Time=uht["time"].unique()
time=Time.tolist()
Shift=uht["shift"].unique()
shift=Shift.tolist()
OU=rijkh["OU"].unique()
OU=OU.tolist()
Plangrp=rijkh["Plangrp"].unique()
Plangrp=Plangrp.tolist()

# decision variable 

x = {(i,j,k,h): xp.var(vartype=xp.integer,name="x_{0}_{1}_{2}_{3}".format(i,j,k,h))
                     for j in OU for k in Plangrp for i in site for h in Shift}


y ={(i,t):xp.var(vartype=xp.continuous,name="y_{0}_{1}".format(i,t))
                    for i in site for t in Time}

Z = {(i):xp.var(vartype=xp.continuous,name="Z_{0}".format(i))
                    for i in site}
W = {(i):xp.var(vartype=xp.continuous,name="W_{0}".format(i))
                    for i in site}


#parameters

a=2
b=2

model=xp.problem()
model=xp.problem(name="SU_Optimization")

model.addVariable(x,y,Z,W)

# adding constraints 

myconstr1=(y[i][t]<=Z[i] for i in site for t in Time)

myconstr2= (2*Z[i]<=W[i] for i in site)


myconstr4=[b*r<=x[i][j][k][h]<=a*r for i in site for j in OU for k in Plangrp for h in shift ]


myconstr3= [xp.Sum(x[i][j][k][h]*u for j in OU for k in Plangrp for h in Shift)<=y[i][t] for i in site for t in Time]

model.addConstraint(myconstr1,myconstr2)

model.setObjective(xp.Sum([Z[i] for i in site]), sense = xp.minimize)

model.solve()



