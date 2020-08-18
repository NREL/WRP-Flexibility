# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:08:06 2018

@author: ksedzro
"""


from pyomo.environ import *
import pandas as pd
import numpy as np
#from pyomo.environ import *
from pyomo.opt import SolverFactory
import time

from parameters import get_global_parameters

from utils import get_model_input_data, build_scuc_model, \
update_parameters, compute_LMPs, write_out_lmps_hourly, get_fuel_type_list


from real_time import  initiate_sced_model, run_sced


mip_solver = SolverFactory('xpress', is_mip=True)
lp_solver = SolverFactory('xpress', is_mip=False)





def run_hour_ahead_sequence(day_idx,data_path,result_path, previous_dispatch,input_mode, start,valid_id, FlexibleRampFactor, ReserveFactor,\
                            RegulatingReserveFactor, load_scaling_factor, wind_scaling_factor,\
                            gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
                            wind_generator_names, margcost_df, blockmargcost_df,\
                            blockmargcost_dict, blockoutputlimit_dict, genforren_dict,\
                            load_s_df, hourly_load_df, hourly_load_dict,\
                            total_hourly_load_dict, slot_load_dict,\
                            RampUpRequirement_dict, RampDnRequirement_dict, wrp_status):
    
    
    #start=1
    #slot =1
    print('Updating parameters for hour-ahead model ...')
    update_time_init = time.time()
    bus_slot_load_dict, slot_load_dict, hourly_load_dict, total_hourly_load_dict,\
    hourly_load_df, genforren_dict, horizon, RampUpRequirement_dict, RampDnRequirement_dict = \
        update_parameters(data_path,day_idx,start, FlexibleRampFactor,\
                          load_scaling_factor, wind_scaling_factor, input_mode, mode='hour-ahead')
    print('Update done! Time: ', time.time()-update_time_init)


    print('Building hour-ahead model ...')
    ha_model_time_init = time.time()
    model2 = build_scuc_model(start,valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                             gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                             margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
                              genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict,\
                              slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict, wrp_status['ha'])
    print('Done building hour-ahead model! Time:', time.time()-ha_model_time_init)
    
    print('Initiating SCED model ...')
    sced_model_time_init = time.time()
    sced_model = initiate_sced_model(day_idx,data_path,start,valid_id, gen_df, genth_df, bus_df,\
                                      branch_df, ptdf_dict, wind_generator_names,\
                                      margcost_df, blockmargcost_df, FlexibleRampFactor,load_scaling_factor, wind_scaling_factor,\
                                      blockmargcost_dict, blockoutputlimit_dict,\
                                       load_s_df, hourly_load_df,hourly_load_dict,input_mode)
    print('Done initiating SCED model! Time: ', time.time()-sced_model_time_init)
    
    
    sced_instance = sced_model.create_instance()
    obj_dict = dict()
    ha_obj = dict()
    rt_obj = dict()
    Hourly_FixedCost = dict()
    Hourly_ProductionCost = dict()
    Hourly_RampingCost = dict()
    Demand = []
    SlotDemand = []
    WindPowerGenerated = []
    SlotWindPowerGenerated = []
    WindPowerForecasted = []
    TotalFlexRampRequired = []
    TotalFlexRampDnRequired = []
    WindTotalFlexRamp = []
    WindTotalFlexRampDn = []
    TotalFlexRampProvided = []
    WindTotalCurtailments = []
    WindCurtailments = []
    WindFlexRamp = []
    WindFlexRampDn = []
    rt_demand = dict()
    rt_load_curtailment = dict()
    FuelType = get_fuel_type_list(list(gen_df.index)) 
    Generation_by_fueltype = pd.DataFrame(0,index=range(1,25), columns=FuelType) 
    ha_previous_dispatch = dict()
    shadow_prices = dict()
    congestion_prices = dict()
    LMPs = dict()
    Dispatch = dict()
    
    for start in range(1,23):
        print('********************************* Hour ',start,'**********************************')
        print('Updating input data for hour-ahead instance ...')
        ha_update_new_time_init = time.time()
        bus_slot_load_dict, slot_load_dict, hourly_load_dict, total_hourly_load_dict,\
        hourly_load_df, genforren_dict, horizon, RampUpRequirement_dict, RampDnRequirement_dict = \
        update_parameters(data_path,day_idx,start, FlexibleRampFactor, load_scaling_factor, wind_scaling_factor,\
                          input_mode, mode='hour-ahead')    
        print('Done updating input data for hour-ahead instance! Time: ', time.time()-ha_update_new_time_init)
        
        print('Creating hour-ahead instance ...')
        ha_instance_time_init = time.time()
        ha_instance = model2.create_instance()
        print('Done creating hour-ahead instance! Time: ', time.time()-ha_instance_time_init)
        
        print('Resetting hour-ahead instance ...')
        ha_instance_reset_time_init = time.time()
        ha_instance = reset_instance(start, ha_previous_dispatch, ha_instance,\
                                     slot_load_dict, hourly_load_dict, total_hourly_load_dict,\
                                     hourly_load_df, genforren_dict,RampUpRequirement_dict,\
                                     RampDnRequirement_dict)
        print('Done with hour-ahead instance reset! Time: ', time.time()-ha_instance_reset_time_init)
        
        
        print('Solving hour-ahead instance ...')
        t0 = time.time()
        results = mip_solver.solve(ha_instance)
        t1 = time.time()
        total_time = t1-t0
        print('Done solving hour-ahead instance! Time:', total_time)
        
        print('*** CONDITIONING THE HOUR-AHEAD RESULTS ***')
        misc_time_init = time.time()
        results.write(num=1)
        TotalRampCost = sum(ha_instance.RampingCost[t].value for t in ha_instance.TimePeriods)
        print('Objective: ',ha_instance.TotalProductionCost.value)
        ha_obj[start]={'fixed_cost': sum(ha_instance.StartupCost[g, 1].value + ha_instance.ShutdownCost[g, 1].value\
                                        for g in ha_instance.ThermalGenerators),\
              'ramp_cost': ha_instance.RampingCost[1].value,\
              'prod_cost':sum(ha_instance.ProductionCost[g, 1].value\
                             for g in ha_instance.ThermalGenerators)}
        #print(value(ha_instance.TimeStart))
        print('')
        #TotWindGen = [sum(ha_instance.PowerGenerated[g,t].value\
        #                  for g in ha_instance.WindGenerators) for t in ha_instance.TimePeriods]
        #print('Total wind generation over this horizon: ',TotWindGen)
        #print('Total thermal generation: ', [sum(ha_instance.PowerGenerated[g,t].value\
        #                                         for g in ha_instance.ThermalGenerators) for t in ha_instance.TimePeriods])
        #print('')
        #print('Thermal generation costs: ', [ha_instance.ProductionCost[g,t].value\
        #                                     for g in ha_instance.ThermalGenerators for t in ha_instance.TimePeriods])
        #print('')
        #print('Ramping cost: ', [ha_instance.RampingCost[t].value for t in ha_instance.TimePeriods])
        print('Total ramp cost: ', TotalRampCost)
        obj_dict[start] = ha_instance.TotalFixedCost.value+ha_instance.TotalProductionCost.value + TotalRampCost
        Hourly_ProductionCost[start] = sum(ha_instance.ProductionCost[g, 1].value\
                             for g in ha_instance.ThermalGenerators)
        Hourly_RampingCost[start] = ha_instance.RampingCost[1].value
        Hourly_FixedCost[start] = sum(ha_instance.StartupCost[g, 1].value + ha_instance.ShutdownCost[g, 1].value\
                                        for g in ha_instance.ThermalGenerators)
        
        for g in ha_instance.ThermalGenerators:
            ha_previous_dispatch[g] = ha_instance.PowerGenerated[g,1].value
        #print('PREV: ',ha_previous_dispatch)
        Demand.append(value(ha_instance.Demand[1]))
        SlotDemand.append([value(ha_instance.SlotDemand[1,s]) for s in ha_instance.TimeSlots] )
  
        WindPowerGenerated.append(sum(ha_instance.PowerGenerated[g,1].value for g in ha_instance.WindGenerators))
        SlotWindPowerGenerated.append([sum(ha_instance.PowerGenerated[g,1].value\
                                           for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
    
        WindPowerForecasted.append([sum(ha_instance.PowerForecast[g,1,s].value\
                                        for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
    
        WindTotalFlexRamp.append([sum(ha_instance.FlexibleRampUpAvailable[g,1,s].value - ha_instance.WindRpCurtailment[g,1,s].value\
                                      for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
    
        WindTotalFlexRampDn.append([sum(ha_instance.FlexibleRampDnAvailable[g,1,s].value\
                                        for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
        
        
        WindFlexRamp.append([[ha_instance.FlexibleRampUpAvailable[g,1,s].value - ha_instance.WindRpCurtailment[g,1,s].value\
                              for s in ha_instance.TimeSlots] for g in ha_instance.WindGenerators ])
    
        WindFlexRampDn.append([[ha_instance.FlexibleRampDnAvailable[g,1,s].value\
                                for s in ha_instance.TimeSlots] for g in ha_instance.WindGenerators ])
        
        WindTotalCurtailments.append([sum(ha_instance.WindRpCurtailment[g,1,s].value\
                                          for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
    
        WindCurtailments.append([[ha_instance.WindRpCurtailment[g,1,s].value\
                                  for s in ha_instance.TimeSlots] for g in ha_instance.WindGenerators ])
  
        
        TotalFlexRampProvided.append([sum(ha_instance.FlexibleRampUpAvailable[g,1,s].value\
                                          for g in ha_instance.ThermalGenerators|ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
    
        TotalFlexRampRequired.append([ha_instance.FlexibleRampUpRequirement[1,s].value for s in ha_instance.TimeSlots])
        TotalFlexRampDnRequired.append([ha_instance.FlexibleRampDnRequirement[1,s].value for s in ha_instance.TimeSlots])
        #print('actual bid for wind0:', ha_instance.PowerGenerated['wind0',1].value)
        for i in FuelType:
            iset = [x for x in ha_instance.AllGenerators if x.startswith(i)]  
            t=start
            Generation_by_fueltype.loc[t,i] = sum(ha_instance.PowerGenerated[g,1].value for g in iset)
        """New!!!"""
        print('*** End of Conditioning! Time: ', time.time()-misc_time_init,' ***')

        print('*** Starting Real-time Security Constrained Economic Dispatch ***')
        overall_sced_time_init = time.time()
        for slot in range(1,7):
            print('Running SCED for slot ', slot, ' ...')
            sced_slot_time_init = time.time()
            sced_instance,sced_results,previous_dispatch =\
            run_sced(day_idx,data_path,previous_dispatch,wind_scaling_factor, input_mode, sced_model,ha_instance, valid_id, gen_df, genth_df, bus_df,\
            branch_df, ptdf_dict, wind_generator_names, margcost_df, blockmargcost_df,\
            FlexibleRampFactor,blockmargcost_dict, blockoutputlimit_dict,\
            load_s_df,slot_load_dict, hourly_load_df, hourly_load_dict,\
            total_hourly_load_dict, bus_slot_load_dict, genforren_dict, start, slot, shift=0)
            
            LMPs, shadow_prices, congestion_prices =\
            compute_LMPs(sced_instance, ptdf_dict, shadow_prices, congestion_prices,\
                         LMPs, start,slot,shift=0)
            
            rt_demand[start,slot] = [start, slot, value(sced_instance.SlotDemand), sum(value(sced_instance.PowerForecast[g])\
                     for g in sced_instance.WindGenerators)]
            rt_load_curtailment[start, slot] = [start, slot, sum(sced_instance.BusCurtailment[b].value\
                                for b in sced_instance.LoadBuses)]
            
            rt_obj[start,slot]={'curtailment_cost': sced_instance.TotalCurtailmentCost.value,\
                  'prod_cost':sced_instance.TotalProductionCost.value}
            
            for g in sced_instance.AllGenerators:
                Dispatch[g,start,slot] = [ g, start, slot, sced_instance.PowerGenerated[g].value]
            #print('passed bid',value(sced_instance.EnergyBid['wind0']))
            print('Done with SCED and LMP computation for slot ', slot, '! Time: ', time.time()-sced_slot_time_init)

        # Write out LMPs for this hour and clear the LMP dictionary
        write_out_lmps_hourly (wrp_status,day_idx,result_path,LMPs, start)
        LMPs=dict()

        if start==22:
            for i in range(1,3):
                print('********************************* Hour ',start+i,' **********************************')
                print('*** CONDITIONING THE HOUR-AHEAD RESULTS ***')
                misc_time_init = time.time()
                
                
                ha_obj[start+i]={'fixed_cost': sum(ha_instance.StartupCost[g, 1+i].value + ha_instance.ShutdownCost[g, 1].value\
                      for g in ha_instance.ThermalGenerators),\
                'ramp_cost': ha_instance.RampingCost[1+i].value,\
                'prod_cost':sum(ha_instance.ProductionCost[g, 1+i].value for g in ha_instance.ThermalGenerators)}
                
                Hourly_ProductionCost[start+i] = sum(ha_instance.ProductionCost[g, 1+i].value\
                                     for g in ha_instance.ThermalGenerators)
                Hourly_RampingCost[start+i] = ha_instance.RampingCost[1+i].value
                Hourly_FixedCost[start+i] = sum(ha_instance.StartupCost[g, 1+i].value + ha_instance.ShutdownCost[g, 1+1].value\
                                for g in ha_instance.ThermalGenerators)
                
                Demand.append(value(ha_instance.Demand[1+i]))
                SlotDemand.append([value(ha_instance.SlotDemand[1+i,s])\
                                   for s in ha_instance.TimeSlots] )
                WindPowerGenerated.append(sum(ha_instance.PowerGenerated[g,1+i].value\
                                              for g in ha_instance.WindGenerators))
                SlotWindPowerGenerated.append([sum(ha_instance.PowerGenerated[g,1+1].value\
                                                   for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
        
                WindPowerForecasted.append([sum(ha_instance.PowerForecast[g,1+i,s].value\
                                                for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
                WindTotalFlexRamp.append([sum(ha_instance.FlexibleRampUpAvailable[g,1+i,s].value - ha_instance.WindRpCurtailment[g,1+i,s].value\
                                              for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
                WindTotalFlexRampDn.append([sum(ha_instance.FlexibleRampDnAvailable[g,1+i,s].value\
                                                for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
                WindTotalCurtailments.append([sum(ha_instance.WindRpCurtailment[g,1+i,s].value\
                                                  for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
                
                WindCurtailments.append([[ha_instance.WindRpCurtailment[g,1+i,s].value\
                                          for s in ha_instance.TimeSlots] for g in ha_instance.WindGenerators ])
                WindFlexRamp.append([[ha_instance.FlexibleRampUpAvailable[g,1+i,s].value - ha_instance.WindRpCurtailment[g,1+i,s].value\
                                      for s in ha_instance.TimeSlots] for g in ha_instance.WindGenerators ])
                WindFlexRampDn.append([[ha_instance.FlexibleRampDnAvailable[g,1+i,s].value\
                                        for s in ha_instance.TimeSlots] for g in ha_instance.WindGenerators ])
                
                TotalFlexRampProvided.append([sum(ha_instance.FlexibleRampUpAvailable[g,1+i,s].value\
                                                  for g in ha_instance.WindGenerators) for s in ha_instance.TimeSlots])
                TotalFlexRampRequired.append([ha_instance.FlexibleRampUpRequirement[1+i,s].value\
                                              for s in ha_instance.TimeSlots])
                TotalFlexRampDnRequired.append([ha_instance.FlexibleRampDnRequirement[1+i,s].value\
                                                for s in ha_instance.TimeSlots])
                for j in FuelType:
                    iset = [x for x in ha_instance.AllGenerators if x.startswith(j)] 
                    Generation_by_fueltype.loc[start+i,j] = sum(ha_instance.PowerGenerated[g,1+i].value for g in iset)
                
                for g in ha_instance.ThermalGenerators:
                    ha_previous_dispatch[g] = ha_instance.PowerGenerated[g,1+i].value
                print('*** End of Conditioning! Time: ', time.time()-misc_time_init,' ***')
                
                """New!!!"""
                for slot in range(1,7):
                    print('Running SCED for slot ', slot, ' ...')
                    sced_slot_time_init = time.time()
                    sced_instance,sced_results,previous_dispatch =\
                    run_sced(day_idx,data_path,previous_dispatch,wind_scaling_factor, input_mode,sced_model,ha_instance, valid_id, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
                    wind_generator_names, margcost_df, blockmargcost_df, FlexibleRampFactor,\
                    blockmargcost_dict, blockoutputlimit_dict, load_s_df,slot_load_dict,\
                    hourly_load_df, hourly_load_dict,total_hourly_load_dict,\
                    bus_slot_load_dict, genforren_dict, start, slot, i)
                    
                    LMPs, shadow_prices, congestion_prices =\
                    compute_LMPs(sced_instance, ptdf_dict, shadow_prices,\
                                 congestion_prices, LMPs, start,slot,i)
                    
                    rt_demand[start+i,slot] = [start+i, slot, value(sced_instance.SlotDemand),\
                             sum(value(sced_instance.PowerForecast[g])\
                                 for g in sced_instance.WindGenerators)]
                             
                    rt_load_curtailment[start+i, slot] = [start+i, slot, sum(sced_instance.BusCurtailment[b].value\
                                        for b in sced_instance.LoadBuses)]
                    
                    rt_obj[start+i,slot]={'curtailment_cost': sced_instance.TotalCurtailmentCost.value,\
                          'prod_cost':sced_instance.TotalProductionCost.value}
                
    
                    for g in sced_instance.AllGenerators:
                        Dispatch[g,start+i,slot] = [g, start+i, slot, sced_instance.PowerGenerated[g].value]
                    print('Done with SCED and LMP computation for slot ', slot, '! Time: ', time.time()-sced_slot_time_init)

                write_out_lmps_hourly (wrp_status,day_idx,result_path,LMPs, start, i)
                LMPs=dict()

        print('SCED done for the hour! Time: ', time.time()-overall_sced_time_init)
        
    print('Additional conditioning ...')
    additional_misc_time_init = time.time()                
    WindFlexRamp_arr = np.array(WindFlexRamp).swapaxes(1,2)
    WindFlexRampDn_arr = np.array(WindFlexRampDn).swapaxes(1,2)
    
    #WindFlexRamp_arr = WindFlexRamp.swapaxes(1,2)
    nt,ns,nw = np.shape(WindFlexRamp_arr)
    WindFlexRamp = np.array([WindFlexRamp_arr[t,s,:] for t in range(nt) for s in range(ns)])
    WindFlexRampDn = np.array([WindFlexRampDn_arr[t,s,:] for t in range(nt) for s in range(ns)])
    
    WindCurtailments_arr = np.array(WindCurtailments).swapaxes(1,2)
    nt,ns,nw = np.shape(WindCurtailments_arr)
    WindCurtailments = np.array([WindCurtailments_arr[t,s,:] for t in range(nt) for s in range(ns)])
    print('End of conditioning! Time: ', time.time()-additional_misc_time_init)
    
    return sced_instance, ha_instance, obj_dict, ha_obj, rt_obj, Demand, SlotDemand, WindPowerForecasted,\
           WindPowerGenerated,SlotWindPowerGenerated,WindTotalFlexRamp,WindTotalFlexRampDn,\
           TotalFlexRampRequired, TotalFlexRampDnRequired,TotalFlexRampProvided,\
           Generation_by_fueltype, WindFlexRamp, WindFlexRampDn,Hourly_ProductionCost,\
           Hourly_RampingCost, Hourly_FixedCost, WindTotalCurtailments, WindCurtailments,\
           LMPs, shadow_prices, congestion_prices, Dispatch, rt_demand, rt_load_curtailment


def reset_instance(start, ha_previous_dispatch,ha_instance, slot_load_dict, hourly_load_dict,\
 total_hourly_load_dict, hourly_load_df, genforren_dict,RampUpRequirement_dict,RampDnRequirement_dict):
    #print(value(instance_temp.TimeEnd))
    ha_instance.Start = start
    for t in ha_instance.TimePeriods:
        #print(t)
        ha_instance.Demand[t] = total_hourly_load_dict[t]
        
        for g in ha_instance.ThermalGenerators:
            
            if value(ha_instance.Start)>1 and t==1:
                #print("CHECK",ha_instance.Start, t)
                ha_instance.PreviousDispatch[g] = max(0,ha_previous_dispatch[g])

            else:
                pass
            
        
        for s in ha_instance.TimeSlots:
            ha_instance.SlotDemand[t,s] = slot_load_dict[t,s]
            ha_instance.FlexibleRampUpRequirement[t,s] = RampUpRequirement_dict[t,s]
            ha_instance.FlexibleRampDnRequirement[t,s] = RampDnRequirement_dict[t,s]
            
            for g in ha_instance.RenewableGenerators:
                ha_instance.PowerForecast[g,t,s] = genforren_dict[g,t,s]
            
        for lbus in ha_instance.LoadBuses:
            ha_instance.BusDemand[lbus,t] = hourly_load_dict[lbus,t]
    return ha_instance
            
