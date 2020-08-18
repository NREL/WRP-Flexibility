# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:39:42 2018

@author: ksedzro
"""

#import sys
#sys.path.insert(0, 'C:/Users/ksedzro/Documents/Python Scripts/ForGrid/FORGrid1_2/Code')

from pyomo.environ import *
import pandas as pd
import numpy as np
#from pyomo.environ import *
from pyomo.opt import SolverFactory
import time
import os

from parameters import get_global_parameters

from utils import get_model_input_data, build_scuc_model, update_parameters, \
plot_generation_mix, aggregate_wind_forecast,assign_unit_commitment_status,compute_da_LMPs, get_fuel_type_list

mip_solver = SolverFactory('xpress', is_mip=True)
lp_solver = SolverFactory('xpress', is_mip=False)


def da_input_data(wind_penetration,day_idx, data_path, FlexibleRampFactor, load_scaling_factor,\
                  start=1, input_mode='static', mode='day-ahead'):
    """
    This function extracts and outputs all relevant input data for day-ahead market simulation
    Syntaxe: 
        load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor,\
        RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
        wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
        blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df,\
        hourly_load_dict, total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
        RampDnRequirement_dict, bus_slot_load_dict, horizon =\
        da_input_data(day_idx, data_path, data_path, day_idx,start,\
        FlexibleRampFactor, load_scaling_factor, start=1, mode='day-ahead', input_mode='static') 
    """
    

    wind_scaling_factor, load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                         gen_df, genth_df, bus_df, branch_df, ptdf_dict, \
                         wind_generator_names, margcost_df, blockmargcost_df,\
                         blockmargcost_dict, blockoutputlimit_dict, genforren_dict,\
                         load_s_df, hourly_load_df, hourly_load_dict, \
                         total_hourly_load_dict, slot_load_dict,\
                         RampUpRequirement_dict, RampDnRequirement_dict = \
                         get_model_input_data(start,day_idx, data_path, wind_penetration)
                         
    bus_slot_load_dict, slot_load_dict, hourly_load_dict, total_hourly_load_dict,\
    hourly_load_df, genforren_dict, horizon, RampUpRequirement_dict, RampDnRequirement_dict = \
            update_parameters(data_path,day_idx,start, FlexibleRampFactor,\
                              load_scaling_factor, wind_scaling_factor, input_mode, 'day-ahead')
            
    return wind_scaling_factor, load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                         gen_df, genth_df, bus_df, branch_df, ptdf_dict, \
                         wind_generator_names, margcost_df, blockmargcost_df,\
                         blockmargcost_dict, blockoutputlimit_dict, genforren_dict,\
                         load_s_df, hourly_load_df, hourly_load_dict,\
                         total_hourly_load_dict, slot_load_dict,\
                         RampUpRequirement_dict, RampDnRequirement_dict, bus_slot_load_dict, horizon
    


def get_da_mip_solution(result_path,load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor,\
        RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
        wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
        blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df,\
        hourly_load_dict, total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
        RampDnRequirement_dict, bus_slot_load_dict, horizon, wrp_status):
    
    """
    This function builds and solve the day-ahead SCUC model and keeps track of 
    the wind and load forecast data used
    Syntaxe:
        da_mip_instance, da_Generation_by_fueltype, da_hourly_objective,\
        total_DAMIP_time, da_forecasts =\
        get_da_mip_solution(load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor,\
        RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
        wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
        blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df,\
        hourly_load_dict, total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
        RampDnRequirement_dict, bus_slot_load_dict, horizon, wrp_status)
    """
    print('Building DA Model ...')
    model_time_init=time.time()
    model = build_scuc_model(start, valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                         gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                         margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
                         genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict, \
                         slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict, wrp_status['da'])
    model_time_end=time.time()

    da_load = pd.DataFrame.from_dict(slot_load_dict,'index')
    da_windfor = pd.DataFrame.from_dict(aggregate_wind_forecast(genforren_dict,wind_generator_names),'index')
    da_forecasts=pd.concat([da_load,da_windfor], axis=1)
    da_forecasts.columns = ['Load_Forecast','Wind_Power_Forecast']
    da_forecasts.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_da_forecasts.csv'))
    


    print('Done building DA model, model time =', model_time_end-model_time_init) 

    print('Creating DA instance')

    instance_time_init = time.time()
    da_instance = model.create_instance()
    instance_time_end = time.time()

    print('Done building DA instance, instance time =', instance_time_end-instance_time_init)
    
    print('Solving DA instance')

    t0 = time.time()
    results = mip_solver.solve(da_instance)

    print('Got a DA solution')

    t1 = time.time()
    total_DAMIP_time = t1-t0
    print("Total solution time:", total_DAMIP_time)
    print("Overall total time :", t1-model_time_init)
    results.write(num=1)
    TotalRampCost = sum(da_instance.RampingCost[t].value for t in da_instance.TimePeriods)
    print('Objective: ',da_instance.TotalFixedCost.value+da_instance.TotalProductionCost.value + TotalRampCost)
    
    da_hourly_objective=dict()
    for t in da_instance.TimePeriods:
        da_hourly_objective[t]={'fixed_cost': sum(da_instance.StartupCost[g, 1].value + da_instance.ShutdownCost[g, 1].value\
                           for g in da_instance.ThermalGenerators), 'ramp_cost':da_instance.RampingCost[t].value,'prod_cost': sum(da_instance.ProductionCost[g,t].value for g in da_instance.ThermalGenerators)}
    
    FuelType = get_fuel_type_list(list(gen_df.index))
    da_Generation_by_fueltype = pd.DataFrame(0,index=range(1,25), columns=FuelType)
    for t in range(1,25):
        for i in FuelType:
                    iset = [x for x in da_instance.AllGenerators if x.startswith(i)]  
                    #t=start
                    da_Generation_by_fueltype.loc[t,i] = sum(da_instance.PowerGenerated[g,t].value for g in iset)
    da_mip_instance = da_instance

    return da_mip_instance, da_Generation_by_fueltype, da_hourly_objective, total_DAMIP_time, da_forecasts



#plot_generation_mix(da_Generation_by_fueltype)


def get_da_lmp(da_mip_instance, start, valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
               gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
               margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
               genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict,\
               slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict, wrp_status, linear_status=1):
    """
    This function builds and solve an LP version of the day-ahead SCUC model using the commitment statuses from 
    MIP solutions for the purpose of extracting the relevant duals and computes the LMPs.
    Output variables are the LP model instance, LMPs dataframe, Shadow price dataframe,
    and Congestion price dataframe.
    Syntaxe:
        da_lp_instance, da_LMP_df, da_shadow_prices_df, da_congestion_prices_df =\
        get_da_lmp(da_mip_instance, start, valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
               gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
               margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
               genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict, \
               slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict, wrp_status, linear_status=1)
    """

    """Build linear model and instance with commitment solutions from previous MILP model instance"""
    print('Building the DA LP model')
    da_lp_model_time_init = time.time()
    lp_model = build_scuc_model(start, valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                         gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                         margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
                         genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict, \
                         slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict, wrp_status['da'], linear_status=1)
    da_lp_model_time = time.time() - da_lp_model_time_init

    print('Done with DA LP model, model time: ', da_lp_model_time)

    print('Building DA LP instance ... ')
    da_lp_instance_time_init = time.time()
    da_lp_instance = lp_model.create_instance()
    da_lp_instance_time = time.time()-da_lp_instance_time_init
    print('Done with DA LP instance, instance time: ', da_lp_instance_time)
    """ Assign Commitment """
    print('Assigning commitment parameters ...')

    commitment_time_init = time.time()
    da_lp_instance = assign_unit_commitment_status(da_mip_instance, da_lp_instance)
    del da_mip_instance
    print('Commitment time: ', time.time()-commitment_time_init)
    
    print('Solving DA LP instance ...')
    da_lp_solve_time_init = time.time()        
    lp_solver.solve(da_lp_instance)
    print('Done solving DA LP instance! Time: ', time.time()-da_lp_solve_time_init)
    
    print('Computing DA LMPs')
    da_LMPs=dict()
    da_shadow_prices=dict()
    da_congestion_prices = dict()
    da_LMPs, da_shadow_prices, da_congestion_prices = compute_da_LMPs(da_lp_instance, ptdf_dict, da_shadow_prices, da_congestion_prices, da_LMPs)
    da_LMP_df = pd.DataFrame.from_dict(da_LMPs,'index')
    da_LMP_df.columns =['Bus', 'Hour', 'LMP']
    
    da_shadow_prices_df = pd.DataFrame.from_dict(da_shadow_prices,'index')
    da_shadow_prices_df.columns =['ShadowPrice']
    
    da_congestion_prices_df = pd.DataFrame.from_dict(da_congestion_prices,'index')
    da_congestion_prices_df.columns =['CongestionPrice']

    del da_lp_instance
    
    return da_LMP_df, da_shadow_prices_df, da_congestion_prices_df


    def run_day_ahead(data_path, result_path, wind_penetration, load_scaling_factor, day_idx, start=1, mode='day-ahead', input_mode='static'):
        """
        syntax:
            da_mip_instance, da_Generation_by_fueltype, da_hourly_objective,\
             total_DAMIP_time, da_forecasts,da_lp_instance, da_LMP_df,\
              da_shadow_prices_df, da_congestion_prices_df = run_day-ahead()

        """

        # get day-ahead input data
        wind_scaling_factor, load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor,\
        RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
        wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
        blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df,\
        hourly_load_dict, total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
        RampDnRequirement_dict, bus_slot_load_dict, horizon =\
        da_input_data(wind_penetration,day_idx, data_path, FlexibleRampFactor, load_scaling_factor,\
         start, input_mode, 'day-ahead')

        

        # build and solve day-ahead model
        da_mip_instance, da_Generation_by_fueltype, da_hourly_objective,\
        total_DAMIP_time, da_forecasts =\
        get_da_mip_solution(result_path,load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor,\
        RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
        wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
        blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df,\
        hourly_load_dict, total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
        RampDnRequirement_dict, bus_slot_load_dict, horizon, wrp_status)

        # compute day-ahead LMPs
        da_lp_instance, da_LMP_df, da_shadow_prices_df, da_congestion_prices_df =\
        get_da_lmp(da_mip_instance, start, valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
               gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
               margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
               genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict, \
               slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict, wrp_status, linear_status=1)

        return da_mip_instance, da_Generation_by_fueltype, da_hourly_objective, total_DAMIP_time, da_forecasts,\
        da_lp_instance, da_LMP_df, da_shadow_prices_df, da_congestion_prices_df
