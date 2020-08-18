# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:47:12 2018

@author: ksedzro
"""
#import sys
#sys.path.insert(0, 'C:/Users/ksedzro/Documents/Python Scripts/ForGrid/FORGrid1_2/Code')

from parameters import get_global_parameters

from pyomo.environ import *
import pandas as pd
import numpy as np
import os
#from pyomo.environ import *
from pyomo.opt import SolverFactory
import time



previous_dispatch=dict()

from utils import get_model_input_data, build_scuc_model, update_parameters, \
plot_generation_mix, aggregate_wind_forecast,assign_unit_commitment_status,\
store_results,compute_da_LMPs, get_fuel_type_list, get_fuel_type

from hourly_sequence import run_hour_ahead_sequence
    
from day_ahead import da_input_data, get_da_mip_solution, get_da_lmp


def main():
  day_idx, data_path, result_path, wrp_status, input_mode, casename, wind_penetration = get_global_parameters()

  ############# DAY-AHEAD ##########################
  da_time_init = time.time()
  
  
  # da_mip_instance, da_Generation_by_fueltype, da_hourly_objective,\
  #              total_DAMIP_time, da_forecasts,da_lp_instance, da_LMP_df,\
  #               da_shadow_prices_df, da_congestion_prices_df = run_day_ahead()
  
  # get day-ahead input data
  FlexibleRampFactor=0.1
  load_scaling_factor=1
  #mode = 'day-ahead'
  
  wind_scaling_factor,load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor,\
  RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
  wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
  blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df,\
  hourly_load_dict, total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
  RampDnRequirement_dict, bus_slot_load_dict, horizon =\
  da_input_data(wind_penetration,day_idx, data_path, FlexibleRampFactor, load_scaling_factor,\
      1, input_mode,'day-ahead')

  
  
  # build and solve day-ahead model
  start=1
  da_mip_instance, da_Generation_by_fueltype, da_hourly_objective,\
  total_DAMIP_time, da_forecasts =\
  get_da_mip_solution(result_path,load_scaling_factor,start,valid_id, FlexibleRampFactor, ReserveFactor,\
  RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df, ptdf_dict,\
  wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
  blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df,\
  hourly_load_dict, total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
  RampDnRequirement_dict, bus_slot_load_dict, horizon, wrp_status)
  
  total_da_time = time.time() - da_time_init
  print('Total DA MIP time: ', total_da_time)
  
  # compute day-ahead LMPs
  da_LMP_time_init = time.time()
  
  da_LMP_df, da_shadow_prices_df, da_congestion_prices_df =\
  get_da_lmp(da_mip_instance, start, valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
         gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
         margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
         genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict, \
         slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict, wrp_status, linear_status=1)
  
  total_da_LMP_time = time.time() - da_LMP_time_init
  print('Total DA LP/LMP time: ', total_da_LMP_time)
  
  ############### REAL-TIME ######################
  #sigma = 0.10
  #load_scaling_factor=2
  #print('START',start)
  hart_time_init = time.time()
  
  sced_instance, ha_instance, obj_dict, ha_obj, rt_obj , Demand, SlotDemand, WindPowerForecasted,WindPowerGenerated,SlotWindPowerGenerated,WindTotalFlexRamp,WindTotalFlexRampDn,\
              TotalFlexRampRequired, TotalFlexRampDnRequired,TotalFlexRampProvided,Generation_by_fueltype, WindFlexRamp, WindFlexRampDn,Hourly_ProductionCost,\
              Hourly_RampingCost, Hourly_FixedCost, WindTotalCurtailments, WindCurtailments, LMPs, shadow_prices, congestion_prices, Dispatch, rt_demand, rt_load_curtailment = \
   run_hour_ahead_sequence(day_idx,data_path,result_path,previous_dispatch,input_mode, start,valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor, load_scaling_factor, wind_scaling_factor, gen_df, genth_df,\
                           bus_df, branch_df, ptdf_dict, wind_generator_names, margcost_df, blockmargcost_df, blockmargcost_dict,\
                           blockoutputlimit_dict, genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict,\
                           slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict,wrp_status)
  
  total_hart_time = time.time() - hart_time_init
  print('Total HA/RT time: ', total_hart_time)
  
   
  ha,has = store_results(Demand, SlotDemand, WindPowerForecasted,WindPowerGenerated,\
                         SlotWindPowerGenerated,WindTotalFlexRamp,WindTotalFlexRampDn,\
                         TotalFlexRampRequired,TotalFlexRampDnRequired,TotalFlexRampProvided,WindTotalCurtailments)
  
  ha.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'_hour_ahead_day'+str(day_idx)+'.csv'))
  has.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'_hour_ahead_slot_day'+str(day_idx)+'.csv'))
  
  rt_load = pd.DataFrame.from_dict(rt_demand,'index')
  rt_load.columns = ['Hour','Slot','Demand','WindForecast']
  
  
  plot_generation_mix(Generation_by_fueltype)
  
  #df=pd.DataFrame.from_dict(LMPs,orient='index')
  #df.columns=['Bus','Hour','Slot','LMP']
  #df.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_sced_LMP_slot_day'+str(day_idx)+'.csv'))
  
  df2=pd.DataFrame.from_dict(Dispatch,orient='index')
  df2.columns=['Generator','Hour','Slot','Dispatch']
  df2.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(wind_penetration*100)+'_sced_dispatch_slot_day'+str(day_idx)+'.csv'))
  
  Fuel_type=get_fuel_type_list(list(df2['Generator']))
  
  
  df3=df2.copy()
  df3['Fueltype']=''
  df3['idx'] = 0
  len(df3.columns)
  #for i in range(len(df3.index)):
  #    #print(df3.iloc[i, 0][:-1])
  #    df3.iloc[i,4]=get_fuel_type_list([df3.iloc[i, 0]])[0]
  #    df3.iloc[i,5]=6*(df3.iloc[i, 1]-1)+df3.iloc[i, 2]
  
  df3['Fueltype'] = df3['Generator'].apply(get_fuel_type)
  df3['idx']=6*(df3['Hour']-1)+df3['Slot']
  
  df4=df3[['Fueltype','Hour','Slot','Dispatch']]
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(6,4))
  df5=df4.groupby(['Fueltype','Hour','Slot'])['Dispatch'].sum().unstack().apply(lambda df: df.reset_index(drop=True))
  
  
  rt_disp = pd.DataFrame([0]*(24*6))
  j=0
  for i in Fuel_type:
      rt_disp[i] = np.array(df5.loc[24*j:23+24*j,:]).reshape(24*6)
      j+=1
  rt_disp = rt_disp[Fuel_type]
  """
  
  #dfng=df5.loc[:23,:]
  #dfng_ar =np.array(dfng).reshape(24*6)
  #
  #dfwind=df5.loc[24:47,:]
  #dfwind_ar =np.array(dfwind).reshape(24*6)
  #rt_disp=pd.DataFrame(dfng_ar, columns=['ng'])
  #rt_disp['wind'] = dfwind_ar
  """
  
  
  # use unstack()
  #data.groupby(['date','type']).count()['amount'].unstack().plot(ax=ax)
  x=range(1,145)
  plt.figure(1)
  for i in (rt_disp.columns):
      plt.plot(x,rt_disp[i], label=i.capitalize(), linewidth=2)
  plt.xlabel('Time (in 10 min increment)')
  plt.ylabel('Generation output (MW)') 
  
  #plt.ylabel('Generation output (MW)')
  plt.title('Real-time System generation shares')
  plt.legend(loc = 'lower center')
  plt.show()
  
  da_gen = da_Generation_by_fueltype
  da_g = pd.DataFrame([0]*(6*24), columns=['ng'])
  
  ha_gen = Generation_by_fueltype
  ha_g = pd.DataFrame([0]*(6*24), columns=['ng'])
  
  
  for i in Fuel_type:  
      da_gen_i = np.array([[da_gen.iloc[j,list(da_gen.columns).index(i)]]*6 for j in range(24)])
      da_g[i] = np.array(da_gen_i).reshape(6*24)
      
      ha_gen_i = np.array([[ha_gen.iloc[j,list(ha_gen.columns).index(i)]]*6 for j in range(24)])
      ha_g[i] = np.array(ha_gen_i).reshape(6*24)
      
  #da_gen = da_Generation_by_fueltype[['ng','wind']]
  #ha_gen = Generation_by_fueltype[['ng','wind']]
  #
  #da_ar_ng = np.array([[da_gen.iloc[i,0]]*6 for i in range(24)])
  #da_ar_w = np.array([[da_gen.iloc[i,1]]*6 for i in range(24)])
  #da_g = pd.DataFrame(np.array(da_ar_ng).reshape(6*24), columns=['ng'])
  #da_g['wind'] = np.array(da_ar_w).reshape(6*24)
  #
  #ha_ar_ng = np.array([[ha_gen.iloc[i,0]]*6 for i in range(24)])
  #ha_ar_w = np.array([[ha_gen.iloc[i,1]]*6 for i in range(24)])
  #ha_g = pd.DataFrame(np.array(ha_ar_ng).reshape(6*24), columns=['ng'])
  #ha_g['wind'] = np.array(ha_ar_w).reshape(6*24)
  
  rt_disp.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'_rt_generation.csv'))
  da_g.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'_da_generation.csv'))
  ha_g.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'_ha_generation.csv'))
  
  plt.figure(figsize=(10,6))
  
      
  for i in (da_g.columns):
      plt.plot(x,da_g[i], label='DA_'+i.capitalize(), linewidth=2)
      
  for i in (ha_g.columns):
      plt.plot(x,ha_g[i], label='HA_'+i.capitalize(), linewidth=2)
      
  for i in (rt_disp.columns):
      plt.plot(x,rt_disp[i], label='RT_'+i.capitalize(), linewidth=2)
  
  
  
  plt.xlabel('Time (in 10 min increment)')
  plt.ylabel('Generation output (MW)')
  
  #plt.ylabel('Generation output (MW)')
  plt.title('System generation shares in day-ahead (DA), hour ahead (HA) and real-time (RT)')
  plt.legend(loc = 'best')
  
  plt.savefig(os.path.join(result_path,'genmix_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'.pdf'))
  plt.show()
      
  plt.figure(figsize=(10,6))
  for i in (da_forecasts.columns):
      plt.plot(x,da_forecasts[i], '--', label='DA_'+i.capitalize(), linewidth=2)
      
  plt.plot(x, has['Demand'], '-', label='HA_DemandForecast', linewidth=2.5)
  plt.plot(x, has['WindPowerForecasted'], '-', label='HA_WindPowerForecasted', linewidth=2.5)
  
  da_forecast = da_forecasts[['Load_Forecast','Wind_Power_Forecast']]
  da_forecast.columns = ['Demand', 'WindPower']
  
  ha_forecast=  has[['Demand', 'WindPowerForecasted']]
  ha_forecast.columns = ['Demand', 'WindPower']
  
  rt_forecast = rt_load[['Demand','WindForecast']]
  rt_forecast.columns = ['Demand', 'WindPower']
  
  for i in (rt_load.columns.difference(['Hour','Slot'])):
      plt.plot(x,rt_load[i], label='RT_'+i.capitalize(), linewidth=1.5, marker='.')
  
  
  
  plt.xlabel('Time (in 10 min increment)')
  plt.ylabel('Wind and demand forecast (MW)')
  
  #plt.ylabel('Generation output (MW)')
  plt.title('Wind and demand forecast in day-ahead (DA), hour ahead (HA) and real-time (RT)')
  plt.legend(loc = 'lower center')
  
  plt.savefig(os.path.join(result_path,'forecasts_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'.pdf'))
  plt.show()    
  
  
  #print(rt_load['Demand'],(rt_disp['ng']+rt_disp['wind']))
  print('#===========================================================================#')
  print('#                           Computation Time Summary                        #')
  print('#===========================================================================#')
  print('#  Case Name: *', casename, '*')
  print('#  Total DA MIP time: ', total_da_time, 'sec.')
  print('#  Total DA LP/LMP time: ', total_da_LMP_time, 'sec.')
  print('#  Total HA/RT time: ', total_hart_time, 'sec.')
  print('#===========================================================================#')
  
  """
  plt.figure(figsize=(10,6))
  plt.plot(x,da_forecasts['Load_Forecast'], '--', label='da_Load_Forecast', linewidth=2)
      
  plt.plot(x, has['Demand'], '-', label='ha_DemandForecast', linewidth=2.5)
  #plt.plot(x, has['WindPowerForecasted'], '-', label='ha_WindPowerForecasted', linewidth=2.5)
  
  plt.plot(x,rt_load['Demand'], label='rt_Demand', linewidth=1.5, marker='.')
  
  
  
  plt.xlabel('Time (in 10 min increment)')
  plt.ylabel('Demand forecast (MW)')
  
  #plt.ylabel('Generation output (MW)')
  plt.title('Demand forecast in day-ahead (da), hour ahead (ha) and real-time (rt)')
  plt.legend(loc = 'lower center')
  """
  n=100*len(rt_disp.columns)+10
  
  plt.figure(figsize=(10,3*len(rt_disp.columns)))
  for i in (rt_disp.columns):
      #if any([max(da_g[i]),max(ha_g[i]),max(rt_disp[i])])>0:
          
      plt.subplot(n+list(rt_disp.columns).index(i)+1)
      plt.plot(x,da_g[i], '--', label='DA_'+i.capitalize(), linewidth=2.5)
      plt.plot(x,ha_g[i], label='HA_'+i.capitalize(), linewidth=1.5, marker='.')
      plt.plot(x,rt_disp[i], label='RT_'+i.capitalize(), linewidth=2)
      plt.grid()
      plt.ylabel(i.capitalize()+' generation (MW)')
      plt.legend(loc = 0)
      #plt.text(2, 0.65, i)
  plt.xlabel('Time (in 10 min increment)')
  
  plt.savefig(os.path.join(result_path,'genmix_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'_subs.pdf'))
  plt.show()
  #plt.ylabel('Generation output (MW)')
  
  
   
  n=100*len(da_forecasts.columns)+10
  plt.figure(figsize=(10,3*len(da_forecasts.columns)))
  for i in (da_forecast.columns):
      plt.subplot(n+list(da_forecast.columns).index(i)+1)
      plt.plot(x,da_forecast[i], '--', label='DA_'+i.capitalize(), linewidth=2)
      plt.plot(x,ha_forecast[i], label='HA_'+i.capitalize(), linewidth=1.5, marker='.')
      plt.plot(x,rt_forecast[i], label='RT_'+i.capitalize(), linewidth=2)
      plt.ylabel(i.capitalize()+' forecast (MW)')
      plt.grid()
      plt.legend(loc = 0)
      #plt.text(2, 0.65, i)
  plt.xlabel('Time (in 10 min increment)')
  plt.savefig(os.path.join(result_path,'forecasts_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'_subs.pdf'))
  
  plt.show()
      
  pd.DataFrame.from_dict(da_hourly_objective,'index').to_csv(os.path.join(result_path,'DA_objective_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'.csv'))
  pd.DataFrame.from_dict(ha_obj,'index').to_csv(os.path.join(result_path,'HA_objective_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'.csv'))
  pd.DataFrame.from_dict(rt_obj,'index').to_csv(os.path.join(result_path,'RT_objective_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_wp_'+str(round(wind_penetration*100))+'.csv'))    


if __name__ == '__main__':

  main()
