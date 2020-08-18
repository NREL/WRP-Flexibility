# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:10:52 2018

@author: ksedzro
"""


from pyomo.environ import *
import pandas as pd
import numpy as np
#from pyomo.environ import *
from pyomo.opt import SolverFactory
import time
from parameters import get_global_parameters

from utils import update_parameters

mip_solver = SolverFactory('xpress', is_mip=True)
lp_solver = SolverFactory('xpress', is_mip=False)

mode='real-time'

previous_dispatch=dict()


def build_sced_model(start,slot,valid_id, gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                     margcost_df, blockmargcost_df, FlexibleRampFactor, blockmargcost_dict, blockoutputlimit_dict,\
                      genforren_sced, load_s_df, hourly_load_df, hourly_load_dict,\
                      total_hourly_load_dict, slot_load_sced, bus_slot_load_sced):


    ########################################################################################################
    # MODIFIED
    # a basic (thermal) unit commitment model, drawn from:                                                 #
    # A Computationally Efficient Mixed-Integer Linear Formulation for the Thermal Unit Commitment Problem #
    # Miguel Carrion and Jose M. Arroyo                                                                    #
    # IEEE Transactions on Power Systems, Volume 21, Number 3, August 2006. 
    # Model with bus-wise curtailment and reserve/ramp shortages                                           #
    """ Sample syntax:
    model = build_sced_model(start,valid_id, gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                     margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
                      genforren_sced, load_s_df, hourly_load_df, hourly_load_dict,\
                      total_hourly_load_dict, slot_load_sced, bus_slot_load_sced)
    """
    ########################################################################################################
    
    
    model = AbstractModel()
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    
    
    
    
    #=======================================================#
    # INPUT DATA
    # Get input from the helper function get_model_input_data
    # as for example:
    # FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
    # gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
    # margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
    # genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, slot_load_dict
    # =  get_model_input_data(day_idx,\
    #  data_path = 'C:/Users/ksedzro/Documents/Python Scripts/5-bus system/')                                           #
    #=======================================================#
    
    
    #****************************************************************************************************************************************************#
    # MODEL COMPONENTS
    
    """SETS & PARAMETERS"""
    ##########################################################
    # string indentifiers for the sets of different types of generators. #
    ##########################################################
    model.AllGenerators = Set(initialize=gen_df.index)
    model.ThermalGenerators = Set(initialize=gen_df[gen_df['GEN_TYPE']=='Thermal'].index)
    model.NonThermalGenerators = Set(initialize=gen_df[gen_df['GEN_TYPE']!='Thermal'].index)
    model.RenewableGenerators = Set(initialize=gen_df[gen_df['GEN_TYPE']=='Renewable'].index)
    model.HydroGenerators = Set(initialize=gen_df[gen_df['GEN_TYPE']=='Hydro'].index)
    model.WindGenerators = Set(initialize=wind_generator_names)
    model.NonFlexGen = Set(initialize=gen_df[gen_df['FLEX_TYPE']=='NonFlexible'].index)
    
    ##########################################################
    # Set of Generator Blocks Set.                               #
    ##########################################################
    model.Blocks = Set(initialize = blockmargcost_df.columns)
    #model.GenNumBlocks = Param(model.ThermalGenerators, initialize=margcost_df['nblock'].to_dict())
    model.BlockSize = Param(model.ThermalGenerators, model.Blocks, initialize=blockoutputlimit_dict)
    ##########################################################
    # string indentifiers for the set of thermal generators buses. #
    ##########################################################
    
    model.GenBuses = Param(model.AllGenerators, initialize=gen_df['GEN_BUS'].to_dict())
    
    ##########################################################
    # string indentifiers for the set of load buses. #
    ##########################################################
    
    model.LoadBuses = Set(initialize=load_s_df.columns.difference(['Hour']))
    
    ##########################################################
    # string indentifiers for the set of branches. #
    ##########################################################
    
    model.Branches = Set(initialize=branch_df.index)
    model.EnforcedBranches = Set(initialize=valid_id)
    
    model.Buses = Set(initialize=bus_df.index)
    
    
    
    #################################################################
    # Line capacity limits: units are MW. #
    #################################################################
    
    model.LineLimits = Param(model.Branches, within=NonNegativeReals, initialize=branch_df['RATE_A'].to_dict())
    
    
    #################################################################
    # PTDF. #
    #################################################################
    
    #model.PTDF = Param(model.Buses, model.Branches, within=Reals, initialize=ptdf_dict)
    
    
    
    #################################################################
    # the global system demand, for each time period. units are MW. #
    #################################################################
    
    
    
    model.SlotDemand = Param(within=NonNegativeReals, initialize=slot_load_sced, mutable=True)
    
    model.EnergyBid = Param(model.AllGenerators, within=NonNegativeReals, initialize=0, mutable=True)
    
    model.ReserveUp = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=0, mutable=True)
    
    model.ReserveDn = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=0, mutable=True)
    
    
    model.PreviousDispatch = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=0, mutable=True)
    
    ##############################################################################################
    # the bus-by-bus demand and value of loss load, for each time period. units are MW and $/MW. #
    ##############################################################################################
    
    model.BusDemand = Param(model.LoadBuses, within=NonNegativeReals, initialize=bus_slot_load_sced, mutable=True)
    
    model.BusVOLL = Param(model.LoadBuses, within=NonNegativeReals, initialize=bus_df[bus_df['PD']>0]['VOLL'].to_dict())
    
    # Power forecasts
    
    model.PowerForecast = Param(model.NonThermalGenerators, within=NonNegativeReals,initialize=genforren_sced, mutable=True)
    
    
    
    model.MinimumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['PMIN'].to_dict())
    
    def maximum_power_output_validator(m, v, g):
       return v >= value(m.MinimumPowerOutput[g])
    
    model.MaximumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, validate=maximum_power_output_validator, initialize=genth_df['PMAX'].to_dict())
    
    #################################################
    # generator ramp up/down rates. units are MW/h. #
    #################################################
    
    # limits for normal time periods
    model.UpperDispatchLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=0, mutable=True)
    #model.NominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['RAMP_10'].to_dict())
    #model.NominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['RAMP_10'].to_dict())
    model.LowerDispatchLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=0, mutable=True)
    
    model.MaximumRamp = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['RAMP_10'].to_dict())
    
    #############################################
    # unit on state at t=0 (initial condition). #
    #############################################
    
    # if positive, the number of hours prior to (and including) t=0 that the unit has been on.
    # if negative, the number of hours prior to (and including) t=0 that the unit has been off.
    # the value cannot be 0, by definition.
    
   
    model.BlockMarginalCost = Param(model.ThermalGenerators, model.Blocks, within=NonNegativeReals, initialize=blockmargcost_dict)
    
   
    #*********************************************************************************************************************************************************#
    """VARIABLES"""
    #==============================================================================
    #  VARIABLE DEFINITION
    #==============================================================================
    
    # indicator variables for each generator, at each time period.
    model.UnitOn = Param(model.ThermalGenerators, within=Binary,initialize=0, mutable=True)
    
    model.Start = Param(within=NonNegativeReals, initialize=start, mutable=True)
    model.Slot = Param(within=NonNegativeReals, initialize=slot, mutable=True)
    
    # amount of power produced by each generator, at each time period.
    model.PowerGenerated = Var(model.AllGenerators,  within=NonNegativeReals, initialize=0.0)
    # amount of power produced by each generator, in each block, at each time period.
    model.BlockPowerGenerated = Var(model.ThermalGenerators, model.Blocks, within=NonNegativeReals, initialize=0.0)
    
    # maximum power output for each generator, at each time period.
    #model.MaximumPowerAvailable = Var(model.AllGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    
    #model.MaxWindAvailable = Var(model.WindGenerators, model.TimePeriods,model.TimeSlots, within=NonNegativeReals, initialize=0.0)
    #model.WindRpCurtailment = Var(model.WindGenerators, model.TimePeriods,model.TimeSlots, within=NonNegativeReals, initialize=0.0)
    
    ###################
    # cost components #
    ###################
    
    # production cost associated with each generator, for each time period.
    model.ProductionCost = Var(model.ThermalGenerators, within=NonNegativeReals, initialize=0.0)
    
    # startup and shutdown costs for each generator, each time period.
    #model.StartupCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    #model.ShutdownCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    
    # cost over all generators, for all time periods.
    model.TotalProductionCost = Var(within=NonNegativeReals, initialize=0.0)
    
    model.BusCurtailment = Var(model.LoadBuses, initialize=0.0, within=NonNegativeReals)
    
    model.Curtailment = Var(initialize=0.0, within=NonNegativeReals)   
    
    
    model.TotalCurtailmentCost = Var(initialize=0.0, within=NonNegativeReals)
    

    
    #*****************************************************************************************************************************************************#
    """CONSTRAINTS"""
    #==============================================================================
    # CONSTRAINTS
    #==============================================================================
    
    
    ############################################
    # supply-demand constraints                #
    ############################################
    # meet the demand at each time period.
    # encodes Constraint 2 in Carrion and Arroyo.
    
    def enforce_bus_curtailment_limits_rule(m,b):
        return m.BusCurtailment[b]<= m.BusDemand[b]
    model.EnforceBusCurtailmentLimits = Constraint(model.LoadBuses, rule=enforce_bus_curtailment_limits_rule) 
    
    
    def definition_hourly_curtailment_rule(m):
       return m.Curtailment == sum(m.BusCurtailment[b] for b in m.LoadBuses)
    
    model.DefineHourlyCurtailment = Constraint(rule=definition_hourly_curtailment_rule) 
    
    def production_equals_demand_rule(m):
       return sum(m.PowerGenerated[g] for g in m.AllGenerators)  == m.SlotDemand - m.Curtailment
   
    def production_equals_demand_rule_b(m):
       return sum(m.PowerGenerated[g] for g in m.AllGenerators)  >= m.SlotDemand
    
    model.ProductionEqualsDemand = Constraint(rule=production_equals_demand_rule)
    
    
   
    ############################################
    # generation limit and ramping constraints #
    ############################################
    
    # enforce the generator power output limits on a per-period basis.
    # the maximum power available at any given time period is dynamic,
    # bounded from above by the maximum generator output.
    
    # the following three constraints encode Constraints 16 and 17 defined in Carrion and Arroyo.
    
    # NOTE: The expression below is what we really want - however, due to a pyomo bug, we have to split it into two constraints:
    # m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]
    # When fixed, merge back parts "a" and "b", leaving two constraints.
    
    def enforce_generator_output_limits_rule_part_a(m,g):
       return m.MinimumPowerOutput[g] * m.UnitOn[g] <= m.PowerGenerated[g]
    
    model.EnforceGeneratorOutputLimitsPartA = Constraint(model.ThermalGenerators, rule=enforce_generator_output_limits_rule_part_a)
    
    def enforce_generator_ramp_limits_rule_part_a(m, g):
    
       if value(m.Start)==1 and value(m.Slot)==1:
           return Constraint.Skip
           
       else:
           return m.PowerGenerated[g] - m.PreviousDispatch[g]<= m.MaximumRamp[g]/6
    
    model.EnforceGeneratorRampLimitsPartB = Constraint(model.ThermalGenerators, rule=enforce_generator_ramp_limits_rule_part_a)
    
    def enforce_generator_ramp_limits_rule_part_b(m, g):
       if value(m.Start)==1 and value(m.Slot)==1:
           return Constraint.Skip
       else:
           return -m.PowerGenerated[g] + m.PreviousDispatch[g]<= m.MaximumRamp[g]/6       
    
    model.EnforceGeneratorRampLimitsPartB = Constraint(model.ThermalGenerators, rule=enforce_generator_ramp_limits_rule_part_b)
    
    def enforce_generator_output_limits_rule_part_b(m,g):
       return m.PowerGenerated[g] <= m.MaximumPowerOutput[g] * m.UnitOn[g]
    
    model.EnforceGeneratorOutputLimitsPartB = Constraint(model.ThermalGenerators, rule=enforce_generator_output_limits_rule_part_b)
    
    def enforce_generator_output_limits_rule_part_c(m,g):
       return m.PowerGenerated[g] >= m.LowerDispatchLimit[g] * m.UnitOn[g]
    
    #model.EnforceGeneratorOutputLimitsPartC = Constraint(model.ThermalGenerators, rule=enforce_generator_output_limits_rule_part_c)
    
    def enforce_generator_output_limits_rule_part_d(m,g):
       return m.PowerGenerated[g] <= m.UpperDispatchLimit[g] * m.UnitOn[g]
    
    #model.EnforceGeneratorOutputLimitsPartD = Constraint(model.ThermalGenerators, rule=enforce_generator_output_limits_rule_part_d)
    
    
    def enforce_generator_block_output_rule(m,g):
       return m.PowerGenerated[g] == sum(m.BlockPowerGenerated[g,k] for k in m.Blocks) + m.UnitOn[g]*margcost_df.loc[g,'Pmax0']
    
    model.EnforceGeneratorBlockOutput = Constraint(model.ThermalGenerators, rule=enforce_generator_block_output_rule)
    
    def enforce_generator_block_output_limit_rule(m, g, k):
       return m.BlockPowerGenerated[g,k] <= m.BlockSize[g,k]
    
    model.EnforceGeneratorBlockOutputLimit = Constraint(model.ThermalGenerators, model.Blocks, rule=enforce_generator_block_output_limit_rule)
    
    
    def enforce_renewable_generator_output_limits_rule(m, g):
       return  m.PowerGenerated[g]<= m.PowerForecast[g]
    
    model.EnforceRenewableOutputLimits = Constraint(model.NonThermalGenerators, rule=enforce_renewable_generator_output_limits_rule)
    
    
    #############################################
    # constraints for computing cost components #
    #############################################
    
    def production_cost_function(m, g):
        return m.ProductionCost[g] == sum(value(m.BlockMarginalCost[g,k])*(m.BlockPowerGenerated[g,k]) for k in m.Blocks) + m.UnitOn[g]*margcost_df.loc[g,'nlcost']
    model.ComputeProductionCost = Constraint(model.ThermalGenerators, rule=production_cost_function)
    #---------------------------------------
    
    # compute the per-generator, per-time period production costs. this is a "simple" piecewise linear construct.
    # the first argument to piecewise is the index set. the second and third arguments are respectively the input and output variables. 
    """
    model.ComputeProductionCosts = Piecewise(model.ThermalGenerators * model.TimePeriods, model.ProductionCost, model.PowerGenerated, pw_pts=model.PowerGenerationPiecewisePoints, f_rule=production_cost_function, pw_constr_type='LB')
    """
    # compute the total production costs, across all generators and time periods.
    def compute_total_production_cost_rule(m):
       return m.TotalProductionCost == sum(m.ProductionCost[g] for g in m.ThermalGenerators)
    
    model.ComputeTotalProductionCost = Constraint(rule=compute_total_production_cost_rule)
    
    
    def compute_total_curtailment_cost_rule(m):
       return m.TotalCurtailmentCost == sum(100000* m.BusCurtailment[b]  for b in m.LoadBuses)
    
    model.ComputeTotalCurtailmentCost = Constraint(rule=compute_total_curtailment_cost_rule)
    
    #*****
    
    #############################################
    # constraints for line capacity limits #
    #############################################
    
    print('Building network constraints ...')
    
    def line_flow_rule(m, l):
       # This is an expression of the power flow on bus b in time t, defined here
       # to save time.
       return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g] for g in m.AllGenerators) -\
	      sum(ptdf_dict[l][b]*(m.BusDemand[b] - m.BusCurtailment[b]) for b in m.LoadBuses)
    
    model.LineFlow = Expression(model.EnforcedBranches, rule=line_flow_rule)
	
    def enforce_line_capacity_limits_rule_a(m, l):
       return m.LineFlow[l] <= m.LineLimits[l]

    def enforce_line_capacity_limits_rule_b(m, l):
       return m.LineFlow[l] >= -m.LineLimits[l]
    
    #def enforce_line_capacity_limits_rule_a(m, l):
    #    return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g] for g in m.AllGenerators) - \
    #           sum(ptdf_dict[l][b]*(m.BusDemand[b] - m.BusCurtailment[b]) for b in m.LoadBuses) <= m.LineLimits[l]
    
    model.EnforceLineCapacityLimitsA = Constraint(model.EnforcedBranches, rule=enforce_line_capacity_limits_rule_a)   
        
    #def enforce_line_capacity_limits_rule_b(m, l):
    #    return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g] for g in m.AllGenerators) - \
    #           sum(ptdf_dict[l][b]*(m.BusDemand[b] - m.BusCurtailment[b]) for b in m.LoadBuses) >= -m.LineLimits[l]
    #           
    model.EnforceLineCapacityLimitsB = Constraint(model.EnforcedBranches, rule=enforce_line_capacity_limits_rule_b)
    
    
    
    ##--------------------------------------------------------------------------------------------------
   
    def enforce_renewable_generator_output_limits_c(m, g):
       return m.PowerGenerated[g] <= m.PowerForecast[g]
    
    model.EnforceRenewableUpperLimits0 = Constraint(model.NonThermalGenerators, rule=enforce_renewable_generator_output_limits_c)  
    
    #---------------------------------------------------------------
    
   
    #-------------------------------------------------------------
    # Objectives
    #
    
    def total_cost_objective_rule(m):
       return m.TotalProductionCost + m.TotalCurtailmentCost
    
    
    model.TotalCostObjective = Objective(rule=total_cost_objective_rule, sense=minimize)
    
    return model

#====================================================================================================================
    
def reset_sced_parameters(previous_dispatch, ha_instance, sced_instance, bus_slot_load_dict, slot_load_dict, genforren_dict, start, slot, shift=0):
     PowerForecast = extract_dictionary_for_sced(genforren_dict, 1, 2, 0, slot, 1, shift)
     #print('Gen forecasts',PowerForecast)
     sced_instance.SlotDemand = slot_load_dict[1+shift,slot]
     BusDemand = extract_dictionary_for_sced(bus_slot_load_dict, 1,2, 0, slot, 1, shift)
     sced_instance.Start=start
     sced_instance.Slot=slot
     
     for g in sced_instance.ThermalGenerators:
         sced_instance.UnitOn[g] = round(value(ha_instance.UnitOn[g,1+shift]))
         sced_instance.ReserveDn[g] = max(0, value(ha_instance.RegulatingReserveDnAvailable[g,1+shift]))
         
         sced_instance.ReserveUp[g] = max(0, value(ha_instance.SpinningReserveUpAvailable[g,1+shift]) +\
                                      value(ha_instance.RegulatingReserveUpAvailable[g,1+shift]))
         sced_instance.EnergyBid[g] = max(0,value(ha_instance.PowerGenerated[g,1+shift]))      
         sced_instance.UpperDispatchLimit[g] = max(0,value(sced_instance.EnergyBid[g]) + value(sced_instance.ReserveUp[g]))
         sced_instance.LowerDispatchLimit[g] = max(0,value(sced_instance.EnergyBid[g]) - value(sced_instance.ReserveDn[g]))
         
         if start==1 and slot==1:
             pass
         else:
             sced_instance.PreviousDispatch[g] = max(0,previous_dispatch[g])
     #print (value(sced_instance.PreviousDispatch[g]))            
         
         
     for g in sced_instance.RenewableGenerators:           
         sced_instance.PowerForecast[g] = PowerForecast[g]
         sced_instance.EnergyBid[g] = max(0,value(ha_instance.PowerGenerated[g,1+shift])) 
         
     for b in sced_instance.LoadBuses:
         sced_instance.BusDemand[b] = BusDemand[b]
         
     return sced_instance    
                                      
                                      
    
        
    
def extract_dictionary_for_sced(original_dict, time_index_order, timeslot_index_order, key_index_order, current_slot, start, shift=0):
    """
    DicSubset = extract_dictionary_for_sced(original_dict, time_index_order, timeslot_index_order, key_index_order, current_slot, start, shift=0)
    """
    DicSubset = dict()
    for key in original_dict.keys():
        if key[time_index_order]==start+shift and key[timeslot_index_order]==current_slot:
            DicSubset[key[0]]=original_dict[key]
    #print(DicSubset)
    return DicSubset
                                     
def run_sced(day_idx,data_path,previous_dispatch,wind_scaling_factor, input_mode, sced_model,ha_instance, valid_id, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
             wind_generator_names, margcost_df, blockmargcost_df, FlexibleRampFactor,\
             blockmargcost_dict, blockoutputlimit_dict, load_s_df,slot_load_dict,\
             hourly_load_df, hourly_load_dict,total_hourly_load_dict,\
             bus_slot_load_dict, genforren_dict, start, slot, shift=0):
    """
    sced_instance = run_sced(sced_model, ha_instance, valid_id, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
             wind_generator_names, margcost_df, blockmargcost_df, FlexibleRampFactor,\
             blockmargcost_dict, blockoutputlimit_dict, load_s_df,slot_load_dict,\
             hourly_load_df, hourly_load_dict,total_hourly_load_dict,\
             bus_slot_load_dict, genforren_dict, start, slot, shift=0)
    """
    load_scaling_factor=1
    bus_slot_load_dict, slot_load_dict, hourly_load_dict, total_hourly_load_dict,\
    hourly_load_df, genforren_dict, horizon, RampUpRequirement_dict, RampDnRequirement_dict = \
        update_parameters(data_path,day_idx,start+shift, FlexibleRampFactor, load_scaling_factor, wind_scaling_factor, input_mode, 'real-time')
        
    #slot_load_sced = slot_load_dict[1+shift,slot]
    
    #bus_slot_load_sced = extract_dictionary_for_sced(bus_slot_load_dict, 1,2, 0, slot, start, shift)
    
    #genforren_sced = extract_dictionary_for_sced(genforren_dict, 1, 2, 0, slot, start, shift)
    
    
    sced_instance = sced_model.create_instance()
    #sced_instance = remove_param_constraints(sced_instance)
    sced_instance = reset_sced_parameters(previous_dispatch, ha_instance, sced_instance, bus_slot_load_dict, slot_load_dict, genforren_dict, start, slot, shift=0)
    #print(genforren_dict)
    sced_results = lp_solver.solve(sced_instance)
    #sced_instance = remove_param_constraints(sced_instance)
    for g in sced_instance.ThermalGenerators:
        previous_dispatch[g] = sced_instance.PowerGenerated[g].value
    #print(previous_dispatch)
    
    return sced_instance,sced_results,previous_dispatch
    
def initiate_sced_model(day_idx, data_path,start, valid_id, gen_df, genth_df, bus_df,\
                                      branch_df, ptdf_dict, wind_generator_names,\
                                      margcost_df, blockmargcost_df, FlexibleRampFactor,load_scaling_factor, wind_scaling_factor,\
                                      blockmargcost_dict, blockoutputlimit_dict,\
                                       load_s_df, hourly_load_df,hourly_load_dict, input_mode):
    shift=0
    slot=1
    bus_slot_load_dict, slot_load_dict, hourly_load_dict, total_hourly_load_dict,\
    hourly_load_df, genforren_dict, horizon, RampUpRequirement_dict, RampDnRequirement_dict = \
        update_parameters(data_path,day_idx,start+shift, FlexibleRampFactor, load_scaling_factor, wind_scaling_factor, input_mode, 'real-time')
    slot_load_sced = slot_load_dict[1+shift,slot]
    
    bus_slot_load_sced = extract_dictionary_for_sced(bus_slot_load_dict, 1,2, 0, slot, start, shift)
    
    genforren_sced = extract_dictionary_for_sced(genforren_dict, 1, 2, 0, slot, start, shift)
    #print('sced genforren',genforren_sced)
    
    sced_model = build_sced_model(start, slot, valid_id, gen_df, genth_df, bus_df,\
                                      branch_df, ptdf_dict, wind_generator_names,\
                                      margcost_df, blockmargcost_df, FlexibleRampFactor,\
                                      blockmargcost_dict, blockoutputlimit_dict,\
                                      genforren_sced, load_s_df, hourly_load_df,\
                                      hourly_load_dict,total_hourly_load_dict,\
                                      slot_load_sced, bus_slot_load_sced)
    return sced_model
