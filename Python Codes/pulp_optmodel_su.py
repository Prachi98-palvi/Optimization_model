# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:11:29 2021

@author: prachpal
"""
import time

import numpy as np
import pandas as pd
import pulp
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
value_dict = value_dict['Value']
print(value_dict.get(('A', 'O4', 'CS O8', '0.5||9.5')))
print(value_dict.keys())

#print(value_dict)

Time_Shift_data = pd.read_csv("u_ht_binary_var.csv")
# print(Time_Shift_data)
time_list = Time_Shift_data.loc[:, 'time']
time_list = time_list.unique()

shift_list = Time_Shift_data.loc[:, 'shift']
shift_list = shift_list.unique()


Time_Shift_data.set_index(['time', 'shift'], inplace=True)
Binary_dict = Time_Shift_data.to_dict()
Binary_dict = Binary_dict['value']
print(Binary_dict)

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
Max_seats_dict = Max_seats_dict['MaxCapacity']
# print(Max_seats_dict)
for j in OU_list:
    for k in Plangrp_list:
        for h in shift_list:
            flag = 0
            for i in site_list:
                if value_dict.get((i, j, k, h)) is not None:
                    print('Key found', (i, j, k, h), ': ', value_dict.get((i, j, k, h)))
                else:
                    flag += 1
                    print('********** Not found', (i, j, k, h))
            print('flag: ', flag)
            if flag == 0:
                print('Error', j, k, h)

# __MINIMIZATION_PROBLEM__
prob = pulp.LpProblem("Optimal Resource Allocation", pulp.LpMinimize)
M0 = 10

# decision variable
x = {}
y = {}
Z = {}
W = {}
for i in site_list:
    for j in OU_list:
        for k in Plangrp_list:
            for h in shift_list:
                x[i, j, k, h] = pulp.LpVariable("nCSA" + '_' + str(i) + '_' + str(j) + '_' + str(k) + '_' + str(h), lowBound=0, cat='Integer')

for i in site_list:
    for t in time_list:
        y[i, t] = pulp.LpVariable("nSeats" + '_' + str(i) + '_' + str(t), lowBound=0, cat='Continuous')

for i in site_list:
    Z[i] = pulp.LpVariable("max_seats" + '_' + str(i), lowBound=0, cat='Continuous')
    W[i] = pulp.LpVariable("total_seats" + '_' + str(i), lowBound=0, cat='Continuous')


# parameters

a = 0.02
b = 0.02

prob += (pulp.lpSum(Z[i] for i in site_list))

# adding constraints 
for i in site_list:
    prob += 2 * Z[i] <= W[i], 'Constraint1_' + str(i)
    for t in time_list:
        prob += y[i, t] <= Z[i], 'Constraint2_' + str(i) + '_' + str(t)
        x_sum = 0
        for j in OU_list:
            for k in Plangrp_list:
                for h in shift_list:
                    x_sum += x[i, j, k, h] * Binary_dict[(t, h)]

        prob += y[i, t] == x_sum, 'Constraint3_' + str(i) + '_' + str(t)

for j in OU_list:
    for k in Plangrp_list:
        for h in shift_list:
            r_sum = 0
            x_sum = 0
            for i in site_list:
                if value_dict.get((i, j, k, h)) is not None:
                    r_sum += int(value_dict[(i, j, k, h)])
                    x_sum += x[i, j, k, h]
            prob += (1 - b) * r_sum <= x_sum, 'Constraint4_1_' + str(i) + '_' + str(j) + '_' + str(k) + '_' + str(h)
            prob += (1 + a) * r_sum >= x_sum, 'Constraint4_2_' + str(i) + '_' + str(j) + '_' + str(k) + '_' + str(h)


for i in site_list:
    x_sum = 0
    for j in OU_list:
        for k in Plangrp_list:
            for h in shift_list:
                x_sum += x[i, j, k, h]
    prob += W[i] == x_sum, 'Constraint5_' + str(i)
    if i in Max_seats_dict.keys():
        if (Max_seats_dict[i]) is not None:
            prob += Z[i] <= Max_seats_dict[i], 'Constraint6_' + str(i)

#__SOLVE__
prob.writeLP('su.lp')
solver_start_time = time.time()
prob.solve(solver=pulp.COIN_CMD(threads=24, presolve=1, strong=1, keepFiles=0, msg=1))
solver_end_time = time.time()
print('Solution: ', pulp.LpStatus[prob.status])
print('CPU time taken by solver = ', solver_end_time - solver_start_time)

# __SOLUTION__
nWarehouse = 0
for i in site_list:
    if Z[i].value() >= 1:
        print('\n\nMaximum number of seats Utilized at Site ', i, ': ', Z[i].value())

'''
import pulp
import math
import os
import time
from filenames import filename_lpmodel

# __fac_loc_model__
# Arguments: parameters and decision variables
# Return Value: - (decision variables passed by address)
# Function: creates and solves the optimization model to find minimum number of warehouses, and transportation cost associated with using warehouses
def fac_loc_model(arg_trace, arg_Plants, arg_Customers, arg_Products, arg_QuarterlyDem, arg_AnnualProdCap, arg_IfNearCC,
                  arg_IfNearPC, arg_Produce, arg_PCdist, arg_CCdist, arg_transCost, arg_goodsPerTruck, is_W, supply, c_storage, p_storage):
    if arg_trace:
        print("\n\n", os.path.basename(__file__), ": In", fac_loc_model.__name__)
	
    # __MINIMIZATION_PROBLEM__
    prob = pulp.LpProblem("Glass Manufacturer Warehouse Location Problem", pulp.LpMinimize)

    # __VARIABLES__
    for c1 in arg_Customers:
        is_W[c1] = pulp.LpVariable("isWarehouse" + '_' + c1, cat='Binary')

    for p in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                supply[p, c, g] = pulp.LpVariable("supply" + '_' + p + '_' + c + '_' + g, lowBound=0, cat='Continuous')

    for c1 in arg_Customers:
        for c in arg_Customers:
            for g in arg_Products:
                c_storage[c1, c, g] = pulp.LpVariable("c_storage_" + c1 + '_' + c + '_' + g, lowBound=0,
                                                      cat='Continuous')

    for p1 in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                p_storage[p1, c, g] = pulp.LpVariable("p_storage" + '_' + p1 + '_' + c + '_' + g, lowBound=0,
                                                      cat='Continuous')

  # binary to denote if a producing plant p1 stores goods to supply to customer c
    is_p_storage = {}
    for p1 in arg_listPlants:
        for c in arg_listCustomers:
            is_p_storage[p1, c] = pulp.LpVariable("is_p_storage_"+ p1 + '_' + c, lowBound = 0, cat='Binary')

    # binary to denote if a customer location c1 stores goods to supply to customer c
    is_c_storage = {}
    for c1 in arg_listCustomers:
        for c in arg_listCustomers:
            is_c_storage[c1, c] = pulp.LpVariable("is_c_storage_"+ c1 + '_' + c , lowBound = 0, cat='Binary')

    
    # auxiliary variables
    aux_numTruck = (1/arg_goodsPerTruck)
    new_quarterly_transportationCost = pulp.LpVariable("new_quarterly_transportationCost", lowBound=0, cat='Continuous')
        
    # calculation of Big M
    M = 999999

    # __OBJECTIVE FUNCTION__

    # minimize the total number of warehouses considering transportation cost
    prob += (
        (99999999 * pulp.lpSum(is_W[c1]  for c1 in arg_Customers)) + new_quarterly_transportationCost
    )



    # __CONSTRAINTS__

    # Cons1: customer location c1 is a warehouse if it stores some products for some customers
    for c1 in arg_Customers:
        aux_sum_storage = 0
        for c in arg_Customers:
            for g in arg_Products:
                aux_sum_storage += c_storage[c1, c, g]
        prob += (
            is_W[c1] * M >= aux_sum_storage, 'C1_' + (str)(c1)
        )

    # Cons2: Total supply of a glass product g to all the customers by a plant p should not exceed its quarterly production capacity
    for p1 in arg_Plants:
        for g in arg_Products:
            aux_sum_supply = 0
            for c in arg_Customers:
                aux_sum_supply += supply[p1, c, g]
            prob += (
                aux_sum_supply <= arg_AnnualProdCap[p1, g] / 4, 'C2_' + (str)(p1) + '_' + (str)(g)
            )


    # Cons4: Ensure that a plant stores a good only if it produces it
    for p1 in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                prob += (
                    arg_Produce[p1, g] * M >= p_storage[p1, c, g],
                    'C4_' + (str)(p1) + '_' + (str)(c) + '_' + (str)(g)
                )

    # Cons5: Total glass product g supplied to a customer c should be equal to its quarterly demand
    for p in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                prob += (
                    supply[p, c, g] == arg_Produce[p, g] * arg_QuarterlyDem[c, g],
                    'C5_' + (str)(c) + '_' + (str)(g) + '_' + (str)(p)
                )

    # Cons6: total glass product g supplied by a plant p to a customer is either stored in some other customer's location (warehouse) or at the g's production plant p
    for p in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                aux_sum_c_storage = 0
                for c1 in arg_Customers:
                    if c == c1:
                        prob += (
                            c_storage[c1, c, g] == 0, 'C6_1_' + (str)(p) + '_' + (str)(c) + '_' + (str)(g)
                        )
                    else:
                        aux_sum_c_storage += c_storage[c1, c, g]
                prob += (
                    supply[p, c, g] == arg_Produce[p, g] * (aux_sum_c_storage + p_storage[p, c, g]),
                    'C6_' + (str)(p) + '_' + (str)(c) + '_' + (str)(g)
                )

    # Cons7: Total goods for a customer stored in warehouses or plants in less than 500 miles should be greater than 80% of the total demands of the customer
    for c in arg_Customers:
        aux_sum_demand = 0
        aux_sum_c_storage = 0
        aux_sum_p_storage = 0
        for g in arg_Products:
            for p1 in arg_Plants:
                aux_sum_p_storage += (arg_IfNearPC[p1, c] * p_storage[p1, c, g])
            for c1 in arg_Customers:
                aux_sum_c_storage += (arg_IfNearCC[c1, c] * c_storage[c1, c, g])
            aux_sum_demand += arg_QuarterlyDem[c, g]
        prob += (
            aux_sum_c_storage + aux_sum_p_storage >= 0.8 * aux_sum_demand, 'C7_' + (str)(c)
        )

    # Con8: Transportation cost
    aux_new_cost = 0
    for c1 in arg_Customers:
        for c in arg_Customers:
            for g in arg_Products:
                aux_plant_distance = 0
                for p1 in arg_Plants:
                    aux_plant_distance += arg_Produce[p1, g] * arg_PCdist[p1, c1]
                aux_new_cost += c_storage[c1, c, g] * aux_numTruck * (arg_CCdist[c1, c] + aux_plant_distance) * arg_transCost

    for p1 in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                aux_new_cost += p_storage[p1, c, g] * aux_numTruck * arg_PCdist[p1, c] * arg_transCost

    prob += (
        new_quarterly_transportationCost == aux_new_cost, 'C8'
    )

	#__SOLVE__
    # prob.writeLP('glass_manufacturer.lp')
    #prob.writeLP(filename_lpmodel)
    solver_start_time = time.time()
    prob.solve(solver=pulp.COIN_CMD(threads=24, presolve=1, strong=1, keepFiles=0, msg=1))
    solver_end_time = time.time()
    print('Solution: ', pulp.LpStatus[prob.status])
    print('CPU time taken by solver = ', solver_end_time - solver_start_time)

    # __SOLUTION__
    print('Warehouse locations: ', end='')
    nWarehouse = 0
    for c in arg_Customers:
        if (is_W[c].value()):
            print(c, end='\t')
            nWarehouse += 1
    print('\n\nTotal number of warehouse: ', nWarehouse)


    # Old transportation cost
    aux_old_cost = 0
    aux_old_num_trucks = 0
    for p1 in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                aux_old_cost += supply[p1,c,g].value() * aux_numTruck * arg_PCdist[p1,c] * arg_transCost
    print('\nOld quarterly transportation cost = ', round(aux_old_cost/1000000,1), 'million USD')

    #print('\nNew quarterly transportation cost (without ceil) = ', new_quarterly_transportationCost.value())

    # New transportation cost
    aux_new_cost = 0
    for c1 in arg_Customers:
        for c in arg_Customers:
            for g in arg_Products:
                aux_plant_distance = 0
                for p1 in arg_Plants:
                    aux_plant_distance += arg_Produce[p1, g] * arg_PCdist[p1, c1]
                aux_new_cost += c_storage[c1, c, g].value() * aux_numTruck * (arg_CCdist[c1, c] + aux_plant_distance) * arg_transCost

    for p1 in arg_Plants:
        for c in arg_Customers:
            for g in arg_Products:
                aux_new_cost += p_storage[p1, c, g].value() * aux_numTruck * arg_PCdist[p1, c] * arg_transCost
    print('\nNew quarterly transportation cost = ', round(aux_new_cost/1000000,1), 'million USD')
    
    
    
'''


