# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:32:00 2018

@author: ksedzro
"""


import re
import json
import pandas as pd
import numpy as np
import operator
import os
import time
import numpy.core.defchararray as npd


from pyomo.environ import *
from pyomo.opt import SolverFactory

from parameters import get_global_parameters


mip_solver = SolverFactory('xpress', is_mip=True)
lp_solver = SolverFactory('xpress', is_mip=False)

def update_parameters(datapath,day_idx,start, FlexibleRampFactor, load_scaling_factor, wind_scaling_factor, input_mode='dynamic', mode='hour-ahead'):
    """
    slot_load_dict, hourly_load_dict, hourly_load_df, genforen_dict = \
    update_parameters(datapath,sigma,start,mode='hour-ahead')
    """
    
    name_suffix = mode+str(start)
    
    if mode=='hour-ahead':
        horizon = min(3,24-start+1) # 3-hour horizon if start<=22
        sigma = 0.15
    elif mode=='real-time':
        horizon = min(1,24-start+1) # 3-hour horizon if start<=22
        sigma = 0.05
    elif mode=='day-ahead':
        horizon = 24 # 24-hour horizon (day-ahead)
        sigma = 0.30
        start = 1
    
    if input_mode=='dynamic':
        
        load_target = pd.read_csv(os.path.join(datapath,'target_load.csv'))
        load_target = load_target[load_target['Day'] == day_idx].copy()
        load_target = load_target[load_target.columns.difference(['Day'])].copy()
        
        #sample_format = pd.read_csv(datapath +'windforecast.csv',index_col=0)
        sample_format = pd.read_csv(os.path.join(datapath ,'renforecast.csv'),index_col=0)
        sample_format = sample_format[sample_format['Day'] == day_idx].copy()
        sample_format = sample_format[sample_format.columns.difference(['Day'])].copy()
        
        """scale wind forecast if needed"""
        """
        Scaling Wind Generation
        """
        """
        wind_penetration_wanted = 0.10 # 
        wind_penetration_current = sum(gen_df.loc[x ,'PMAX'] for x in gen_df.index if x.startswith('wind'))/ sum(gen_df['PMAX'])# 
        wind_scaling_facor = wind_penetration_wanted * (1/wind_penetration_current -1)/(1-wind_penetration_wanted)   
        
        # Scale Capacity
        for x in sample_format.index:
            if x.startswith('wind'):
                gen_df.loc[x ,'PMAX'] = wind_scaling_facor*gen_df.loc[x ,'PMAX']
        
        # Scale forecast
        for x in sample_format.columns:
            if x.startswith('wind'):
                sample_format.loc[:,x] = wind_scaling_facor*sample_format.loc[:,x]
        """

            
        genforen = sample_format.loc[operator.and_(sample_format['Hour']>=start , sample_format['Hour']<=start+horizon-1)].copy()
         
        
        slot_load = sample_format.loc[operator.and_(sample_format['Hour']>=start , sample_format['Hour']<=start+horizon-1)][['Hour','Slot']].copy()
        
        #sl = sample_format[[all([a,b]) for a,b in zip(sample_format['Hour']>=start , sample_format['Hour']<=start+horizon-1)]].copy()
        
        hourly_load = load_target.loc[operator.and_(load_target['Hour']>=start , load_target['Hour']<=start+horizon-1)].copy()
        
        
        h_cols = hourly_load.columns
        #s_cols = slot_load.columns
        hourly_load[h_cols.difference(['Hour'])] = hourly_load[h_cols.difference(['Hour'])]*load_scaling_factor
        #slot_load[s_cols.difference(['Hour','Slot'])] = slot_load[s_cols.difference(['Hour','Slot'])]*load_scaling_factor
        
        
        
        #slot_load[["Hour","Slot"]] = sample_format[["Hour","Slot"]]
        #hourly_load["Hour"] = range(1,horizon+1)
        for col in hourly_load.columns.difference(['Hour']):
            for h in range(start,start+horizon):
                #print(h)
                #print("Hi", load_target.loc[load_target['Hour']==h,col]*(sigma*np.random.randn(6) +1))
                sv = np.array(load_target.loc[load_target['Hour']==h,col])*(sigma*np.random.randn(6) +1)
                sv = np.clip(sv,0,None) # no element of sv should be less than 0
                slot_load.loc[slot_load['Hour']==h ,col] = sv
                hourly_load.loc[hourly_load['Hour']==h,col] = np.mean(sv)
        
        
        slot_load['LOAD'] = slot_load[slot_load.columns.difference(['Hour','Slot','LOAD'])].sum(axis=1)
        hourly_load['LOAD'] = hourly_load[hourly_load.columns.difference(['Hour','LOAD'])].sum(axis=1)
        #print (hourly_load)
        
        for col in genforen.columns.difference(['Hour','Slot']):
            for h in range(start,start+horizon):
                if max(genforen.loc[genforen['Hour']==h,col])>0:
                    sv = np.array(genforen.loc[genforen['Hour']==h,col])*(sigma*np.random.randn(6) +1)
                    sv = np.clip(sv,0,None) # no element of sv should be less than 0
                    genforen.loc[genforen['Hour']==h ,col] = sv
                else:
                    genforen.loc[genforen['Hour']==h ,col] = 0
                
        """Converting dataframe hour values to 1,2,3 for the sake of successful instanciation of SCUC model"""
        if start>=1:
            
            genforen = ext2int(genforen,start)
            slot_load = ext2int(slot_load,start)
            hourly_load = ext2int(hourly_load,start)
            #hourly_load = ext2int(hourly_load,start)
            
        
        hourly_load.to_csv(os.path.join(datapath,name_suffix+'_hourly_load_df.csv'))
        slot_load.to_csv(os.path.join(datapath,name_suffix+'_slot_load_df.csv'))
        genforen.to_csv(os.path.join(datapath,name_suffix+'_genforen_df.csv'))
        
    elif input_mode=='static':
        hourly_load = pd.read_csv(os.path.join(datapath,name_suffix+'_hourly_load_df.csv'), index_col=0)
        slot_load = pd.read_csv(os.path.join(datapath,name_suffix+'_slot_load_df.csv'), index_col=0)
        genforen = pd.read_csv(os.path.join(datapath,name_suffix+'_genforen_df.csv'), index_col=0)
        
        
    
    
    # Scale forecast
    for x in genforen.columns:
        if x.startswith('wind'):
            genforen.loc[:,x] = wind_scaling_factor*genforen.loc[:,x]
           
        
    bus_slot_load_dict = from_df_to_dict(slot_load[slot_load.columns.difference(['LOAD'])],['Hour','Slot'])  
    #print (bus_slot_load_dict)      
    genforen_dict = from_df_to_dict(genforen,['Hour','Slot'])
    slot_load_dict = from_1coldf_to_dict(slot_load, ['LOAD'], ['Hour','Slot'])
    hourly_load_dict = from_df_to_dict(hourly_load.loc[:,hourly_load.columns.difference(['LOAD'])],['Hour'])
    total_hourly_load_dict = from_1coldf_to_dict(hourly_load, ['LOAD'], ['Hour'])
    hourly_load_df = hourly_load
    hourly_load_df['Idx']=hourly_load_df['Hour']
    hourly_load_df.set_index('Idx', inplace=True)
    
    RampUpRequirement_dict = compute_ramp_up_requirement(slot_load_dict,FlexibleRampFactor)
    RampDnRequirement_dict = compute_ramp_down_requirement(slot_load_dict,FlexibleRampFactor)
    
         
    
    return bus_slot_load_dict, slot_load_dict, hourly_load_dict, total_hourly_load_dict,\
 hourly_load_df, genforen_dict, horizon, RampUpRequirement_dict, RampDnRequirement_dict


def from_df_to_dict(df,dict_keys):
    """
    This function returns a 2- or 3-key dictionary from a given dataframe df
    df: dataframe
    dict_keys: column names that will serve as keys for the dictionary entries
    except the value columns extracted in col. Note that each element of dict_keys
    should also be a column name in df.
    sample dict_keys: dict_keys = ['Hour','Slot']
    sample syntax: my_dictionary = from_df_to_dict(my_dataframe,['Hour','Slot'])
    
    """
    df2dict = dict()
    
#    k = int("{0:b}".format(int(np.array(df[dict_keys[0]])[0]+1)))%10 # binary conversion and last digit extraction
#    if len(dict_keys)==2:
#        j = int("{0:b}".format(int(np.array(df[dict_keys[1]])[0]+1)))%10 # binary conversion and last digit extraction
#        #print(j)
    
    columns = df.columns.difference(dict_keys)
    for i, t in df.iterrows():
        for col in columns:
            if len(dict_keys)==2:
                df2dict[(col, int(t[dict_keys[0]]), int(t[dict_keys[1]]))] = t[col]
            elif len(dict_keys)==1:
                df2dict[(col, int(t[dict_keys[0]]))] = t[col]
            else:
                raise ValueError("This function only deals with dictionaries with 2 or 3 keys")
    return df2dict


#start=1
def from_1coldf_to_dict(df, col, dict_keys):
    """
    This function returns a 2- or 3-key dictionary from a given dataframe df
    df: dataframe
    dict_keys: column names that will serve as keys for the dictionary entries
    except the value columns extracted in col. Note that each element of dict_keys
    should also be a column name in df.
    sample dict_keys: dict_keys = ['Hour','Slot']
    sample syntax: my_dictionary = from_1coldf_to_dict(mydf, col, ['Hour','Slot'])
    
    """
    dfn = df[dict_keys+col].copy()
    #print(dfn)
    
    df2dict = dict()
    
#    k = int("{0:b}".format(int(np.array(df[dict_keys[0]])[0]+1)))%10 # binary conversion and last digit extraction
#    if len(dict_keys)==2:
#        j = int("{0:b}".format(int(np.array(df[dict_keys[1]])[0]+1)))%10 # binary conversion and last digit extraction
#        #print(j)
    
    for i, t in dfn.iterrows():
       
        if len(dict_keys)==2:
            df2dict[(int(t[dict_keys[0]]), int(t[dict_keys[1]]))] = t[col[0]]
        elif len(dict_keys)==1:
            df2dict[(int(t[dict_keys[0]]))] = t[col[0]]
        else:
            raise ValueError("This function only deals with dictionaries with 2 or 3 keys")
    return df2dict
                    
#slot_load_dict, hourly_load_dict, total_hourly_load_dict, hourly_load_df, genforen_dict, horizon = \
#    update_parameters(datapath,sigma,day_idx,start,mode='hour-ahead')




def compute_ramp_up_requirement(slot_load_dict,FlexibleRampFactor):
    RampUpReq = dict()
    tlength = int(len(slot_load_dict)/6)
    for t in range(1,tlength+1):
        for s in range(1,7):
            if  [t,s] == [tlength,6] :
                RampUpReq[t,s] = max(0.0,(FlexibleRampFactor * slot_load_dict[t,s]))
                
            elif t<tlength and s == 6:    
                RampUpReq[t,s] = max(0.0,((1+FlexibleRampFactor) *  slot_load_dict[t+1,1] -  slot_load_dict[t,s]))
            
            elif (t<=tlength) and (s < 6):    
                RampUpReq[t,s] = max(0.0,((1+FlexibleRampFactor) *  slot_load_dict[t,s+1] -  slot_load_dict[t,s]))
            else:
                pass
    return RampUpReq
                
def compute_ramp_down_requirement(slot_load_dict,FlexibleRampFactor):
    RampDnReq = dict()
    tlength = int(len(slot_load_dict)/6)
    for t in range(1,tlength+1):
        for s in range(1,7):
            if  [t,s] == [tlength,6] :
                RampDnReq[t,s] = max(0.0, (FlexibleRampFactor *  slot_load_dict[t,s]))
                
            elif t<tlength and s == 6:    
                RampDnReq[t,s] = max(0.0, (slot_load_dict[t,s] - (1-FlexibleRampFactor) *  slot_load_dict[t+1,1]))
            
            elif (t<=tlength) and (s < 6):    
                RampDnReq[t,s] = max(0.0, ( slot_load_dict[t,s] - (1-FlexibleRampFactor) *  slot_load_dict[t,s+1]) ) 
            else:
                pass
    return RampDnReq
    
    
    
#day_idx = 2
    
def get_model_input_data(start, day_idx , data_path, wind_penetration):
    
    """sample syntax:
    wind_scaling_factor, load_scaling_factor, start, valid_id, FlexibleRampFactor,\
    ReserveFactor, RegulatingReserveFactor, gen_df, genth_df, bus_df, branch_df,\
    ptdf_dict, wind_generator_names,margcost_df, blockmargcost_df, blockmargcost_dict,\
    blockoutputlimit_dict,genforren_dict, load_s_df, hourly_load_df, hourly_load_dict,\
    total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict\
    = get_model_input_data(start, day_idx , data_path, wind_penetration)
    """
    
    print('loading data ...')


    bus_df = pd.read_csv(os.path.join(data_path,'bus.csv'),index_col=['BUS_ID'])
    branch_df = pd.read_csv(os.path.join(data_path,'branch.csv'),index_col=['BR_ID'])
    
    gen_df = pd.read_csv(os.path.join(data_path,'generator.csv'),index_col=0) # Generation units with their parameters
    
    
    
    #genfor_df = pd.read_csv(data_path+'windforecast.csv',index_col=0) # Forecast data
    genfor_df = pd.read_csv(os.path.join(data_path,'renforecast.csv'),index_col=0) # Forecast data
    print(genfor_df.head())
    #print(type(genfor_df['Day']))
    
    #resultdataset = npd.equal(dataset1, dataset2)
    #import pdb;pdb.set_trace()
    genfor_df = genfor_df[genfor_df['Day'] == day_idx].copy()
    genfor_df = genfor_df[genfor_df.columns.difference(['Day'])].copy()
    
    
    hourly_load_df = pd.read_csv(os.path.join(data_path,'hourly_load.csv'),index_col=0) # Hourly load
    hourly_load_df = hourly_load_df[hourly_load_df['Day'] == day_idx].copy()
    hourly_load_df = hourly_load_df[hourly_load_df.columns.difference(['Day'])].copy()
    
    slot_load_df = pd.read_csv(os.path.join(data_path,'slot_load.csv'),index_col=0) # 10-min load
    slot_load_df = slot_load_df[slot_load_df['Day'] == day_idx].copy()
    slot_load_df = slot_load_df[slot_load_df.columns.difference(['Day'])].copy()
    
    load_scaling_factor = 1.00
    h_cols = hourly_load_df.columns
    s_cols = slot_load_df.columns
    hourly_load_df[h_cols.difference(['Hour'])] = hourly_load_df[h_cols.difference(['Hour'])]*load_scaling_factor
    slot_load_df[s_cols.difference(['Hour','Slot'])] = slot_load_df[s_cols.difference(['Hour','Slot'])]*load_scaling_factor
    
    
    
    """
    Scaling Wind Generation
    """
    """
    wind_penetration_wanted = 0.10 # 
    wind_penetration_current = sum(gen_df.loc[x ,'PMAX'] for x in gen_df.index if x.startswith('wind'))/ sum(gen_df['PMAX'])# 
    wind_scaling_facor = wind_penetration_wanted * (1/wind_penetration_current -1)/(1-wind_penetration_wanted)   
    
    # Scale Capacity
    for x in gen_df.index:
        if x.startswith('wind'):
            gen_df.loc[x ,'PMAX'] = wind_scaling_facor*gen_df.loc[x ,'PMAX']
    
    # Scale forecast
    for x in genfor_df.columns:
        if x.startswith('wind'):
            genfor_df.loc[:,x] = wind_scaling_facor*genfor_df.loc[:,x]
    """
    genforren_df=pd.DataFrame()
    #genforren_df=genfor_df.loc[:,gen_df[gen_df['GEN_TYPE']!='Thermal'].index]
    genforren_df=genfor_df
    genforren_df.fillna(0, inplace=True)
    
    

    
    """
    select the branches where to enforce network constraint based on voltage level
    """
    kV_level = 230
    bus_kVlevel_set = list(bus_df[bus_df['BASEKV']>=kV_level].index)
    branch_kVlevel_set = [i for i in branch_df.index if branch_df.loc[i,'F_BUS'] in bus_kVlevel_set and branch_df.loc[i,'T_BUS'] in bus_kVlevel_set]
    valid_id = branch_kVlevel_set
    ptdf_df = pd.read_csv(os.path.join(data_path,'ptdf.csv'),index_col=0)
    
    # One can select the branches where to enforce network constraint based on thermal limits as below 
    #valid_id = branch_df[branch_df.loc[:,'RATE_A']>=1500].index
    
    ptdf_df = ptdf_df.loc[valid_id,:].copy()
    
    gen_df['STARTUP_RAMP']  = gen_df[['STARTUP_RAMP','PMIN']].max(axis=1)
    gen_df['SHUTDOWN_RAMP'] = gen_df[['SHUTDOWN_RAMP','PMIN']].max(axis=1)
    
    wind_generator_names  =  [x for x in gen_df.index if x.startswith('wind')]
    
    genth_df = pd.DataFrame(gen_df[gen_df['GEN_TYPE']=='Thermal'])
    
    print('Finished loading data')
    
    if not os.path.exists(os.path.join(data_path,'marginalcost.csv')):
        
        # This section creates some auxiliary CSV files needed in the model
        
        nn=6
        margcost_df = pd.DataFrame([],index=genth_df.index, columns=[str(i) for i in range(1,nn)])
        
        
        margcost_df['Pmax0']= genth_df['COST_0']
        margcost_df['nlcost']= genth_df['COST_1']
        gtherm=genth_df[['COST_'+str(i) for i in range(2*nn)]]
        gtherm['nblock'] = list([(np.count_nonzero(np.array(gtherm.loc[i,'COST_1':]))-1)/2 for i in gtherm.index])
        gtherm['one'] = 1
        gtherm['zero'] = 0
        for i in range(1,nn):
            gtherm.loc[:,'denom']= gtherm.loc[:,'COST_'+str(2*i)] - gtherm.loc[:,'COST_'+str(2*i-2)]
            gtherm.loc[:,'num']= gtherm.loc[:,'COST_'+str(2*i+1)] - gtherm.loc[:,'COST_'+str(2*i-1)]
            for j in gtherm.index:
                if (gtherm.loc[j,'COST_'+str(2*i+1)] - gtherm.loc[j,'COST_'+str(2*i-1)])>0:
                    margcost_df.loc[j,str(i)]= gtherm.loc[j,'num']/ gtherm.loc[j,'denom']
                    margcost_df.loc[j,'Pmax'+str(i)]= gtherm.loc[j,'COST_'+str(2*i)]
                else:
                    margcost_df.loc[j,str(i)]= 0
                    margcost_df.loc[j,'Pmax'+str(i)]= 0    
        margcost_df.fillna(0, inplace=True)
        margcost_df['nblock'] = gtherm['nblock']
        margcost_df.clip_lower(0,inplace=True)
        #uuu=[np.count_nonzero(np.array(margcost_df.loc[i,:])) for i in margcost_df.index]
        #---------------------------
        #gtherm.head()
        #margcost_df.tail()
        
           
        blockmargcost_df = margcost_df [[str(i) for i in range(1,nn)]].copy()
        
                
        blockmaxoutput_df = margcost_df[['Pmax'+str(i) for i in range(nn)]].copy()
        for i in blockmaxoutput_df.index:
            blockmaxoutput_df.loc[i,'Pmax'+str(gtherm.loc[i,'nblock'])] = genth_df.loc[i,'PMAX']
        
        
        for i in range(1,nn):
            blockmaxoutput_df[str(i)] = margcost_df['Pmax'+str(i)] - margcost_df['Pmax'+str(i-1)]
        
        
        blockmaxoutput_df.clip_lower(0,inplace=True) 
        blockoutputlimit_df = blockmaxoutput_df[[str(i) for i in range(1,nn)]].copy()
        blockoutputlimit_df.clip_lower(0,inplace=True) 
        
        
        margcost_df.to_csv(os.path.join(data_path,'marginalcost.csv'))
        blockmargcost_df.to_csv(os.path.join(data_path,'blockmarginalcost.csv'))
        blockoutputlimit_df.to_csv(os.path.join(data_path,'blockoutputlimit.csv'))
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    else:
        margcost_df= pd.read_csv(os.path.join(data_path,'marginalcost.csv'), index_col=0)
        blockmargcost_df = pd.read_csv(os.path.join(data_path,'blockmarginalcost.csv'), index_col=0)
        blockoutputlimit_df = pd.read_csv(os.path.join(data_path,'blockoutputlimit.csv'), index_col=0)
    
    print('Creating dictionaries ...')
    
    hourly_load_dict = dict()
    load_s_df = hourly_load_df[hourly_load_df.columns.difference(['LOAD'])].copy()
    columns = load_s_df.columns.difference(["Hour"])
    
    """get the valid increment that ensures all time and slot indices start from 1"""
    k= int("{0:b}".format(int(np.array(hourly_load_df['Hour'])[0]+1)))%10 # binary conversion and last digit extraction
    
    for i, t in load_s_df.iterrows():
        for col in columns:
            hourly_load_dict[(col, int(t["Hour"]+k))] = t[col] # +1 to reindex hour from 1
    
    
    
    slot_load_dict = dict() # indexed slot_load_dict[bus_id,hour,slot]
    load_sl_df = slot_load_df[['Hour','Slot','LOAD']].copy()
    columns = load_sl_df.columns.difference(["Hour","Slot"])
    
    j = int("{0:b}".format(int(np.array(slot_load_df['Slot'])[0]+1)))%10 # binary conversion and last digit extraction
    k = int("{0:b}".format(int(np.array(slot_load_df['Hour'])[0]+1)))%10 # binary conversion and last digit extraction
    
    for i, t in load_sl_df.iterrows():
        for col in columns:
            slot_load_dict[(int(t["Hour"]+k), int(t["Slot"]+j))] = t[col]
    
    
    print('Started with the ptdf dictionary')
            
            
    ptdf_dict = ptdf_df.to_dict('index') # should be indexed ptdf_dict[l][b], l for line and b for bus
    print('Done with the ptdf dictionary')
     
    j = int("{0:b}".format(int(np.array(genforren_df['Slot'])[0]+1)))%10 # binary conversion and last digit extraction
    k = int("{0:b}".format(int(np.array(genforren_df['Hour'])[0]+1)))%10 # binary conversion and last digit extraction
       
    genforren_dict = dict()
    columns = genforren_df.columns.difference(["Hour","Slot"])
    for i, t in genforren_df.iterrows():
        for col in columns:
            genforren_dict[(col, int(t["Hour"]+k), int(t["Slot"]+j))] = t[col]
    
    print('Done with the forecast dictionary')
    
    blockmargcost_dict = dict()    
    columns = blockmargcost_df.columns
    for i, t in blockmargcost_df.iterrows():
        for col in columns:
            #print (i,t)
            blockmargcost_dict[(i,col)] = t[col]        
    
    total_hourly_load_dict = from_1coldf_to_dict(hourly_load_df, ['LOAD'], ['Hour'])
    
    print('Done with the block marginal cost dictionary')
    
    blockoutputlimit_dict = dict()
    columns = blockoutputlimit_df.columns
    for i, t in blockoutputlimit_df.iterrows():
        for col in columns:
            #print (i,t)
            blockoutputlimit_dict[(i,col)] = t[col]
    print('Done with all dictionaries')
    #================================================
    
    
    #
    # Reserve Parameters
    FlexibleRampFactor = 0.1
    ReserveFactor = 0.1
    RegulatingReserveFactor = 0.1
    
    RampUpRequirement_dict = compute_ramp_up_requirement(slot_load_dict,FlexibleRampFactor)
    RampDnRequirement_dict = compute_ramp_down_requirement(slot_load_dict,FlexibleRampFactor)
    
    wind_scaling_factor = compute_wind_scaling_factor(gen_df, wind_penetration)
    
    return wind_scaling_factor, load_scaling_factor, start, valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                     gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                     margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
                      genforren_dict, load_s_df, hourly_load_df, hourly_load_dict,\
                      total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict, RampDnRequirement_dict
                      


from pyomo.environ import *
from pyomo.opt import SolverFactory

def build_scuc_model(start,valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                     gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                     margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
                      genforren_dict, load_s_df, hourly_load_df, hourly_load_dict,\
                      total_hourly_load_dict, slot_load_dict, RampUpRequirement_dict,\
                      RampDnRequirement_dict, wrp=1, linear_status=0):
    
    
    ########################################################################################################
    # MODIFIED
    # a basic (thermal) unit commitment model, drawn from:                                                 #
    # A Computationally Efficient Mixed-Integer Linear Formulation for the Thermal Unit Commitment Problem #
    # Miguel Carrion and Jose M. Arroyo                                                                    #
    # IEEE Transactions on Power Systems, Volume 21, Number 3, August 2006. 
    # Model with bus-wise curtailment and reserve/ramp shortages                                           #
    """ Sample syntax:
    model = build_scuc_model(start,valid_id, FlexibleRampFactor, ReserveFactor, RegulatingReserveFactor,\
                     gen_df, genth_df, bus_df, branch_df, ptdf_dict, wind_generator_names,\
                     margcost_df, blockmargcost_df, blockmargcost_dict, blockoutputlimit_dict,\
                      genforren_dict, load_s_df, hourly_load_df, hourly_load_dict, total_hourly_load_dict, slot_load_dict)
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
    
    
    ###################################################
    # the number of time periods under consideration, #
    # in addition to the corresponding set.           #
    ###################################################
    model.Start = Param(within=PositiveIntegers, initialize=start, mutable = True)
    
    model.TimeStart = Param(within=PositiveIntegers, initialize=1, mutable = False)
    
    model.NumTimePeriods = Param(within=PositiveIntegers, initialize=len(hourly_load_df.index), mutable = True)
    
    
    model.TimeEnd = Param(within=PositiveIntegers, initialize=model.TimeStart + model.NumTimePeriods - 1, mutable = True)
    
    
    model.TimePeriods = RangeSet(model.TimeStart, model.TimeEnd)
    
    """
    Time slots are subdivisions within time periods. Considering 10 min time slots, 
    we have 6 in one 60-min time period
    """
    
    model.NumTimeSlots = Param(within=PositiveIntegers, initialize=6, default=6)
    
    model.TimeSlots = RangeSet(1, model.NumTimeSlots)
    #print(value(model.NumTimeSlots),value(model.TimeSlots))
    #################################################################
    # the global system demand, for each time period. units are MW. #
    #################################################################
    
    model.Demand = Param(model.TimePeriods, within=NonNegativeReals, initialize=total_hourly_load_dict, mutable=True)
    
    model.SlotDemand = Param(model.TimePeriods, model.TimeSlots, within=NonNegativeReals, initialize=slot_load_dict, mutable=True)
    
    ##############################################################################################
    # the bus-by-bus demand and value of loss load, for each time period. units are MW and $/MW. #
    ##############################################################################################
    
    model.BusDemand = Param(model.LoadBuses, model.TimePeriods, within=NonNegativeReals, initialize=hourly_load_dict, mutable=True)
    
    model.BusVOLL = Param(model.LoadBuses, within=NonNegativeReals, initialize=bus_df[bus_df['PD']>0]['VOLL'].to_dict())
    
    # Power forecasts
    
    model.PowerForecast = Param(model.NonThermalGenerators, model.TimePeriods, model.TimeSlots, within=NonNegativeReals, initialize=genforren_dict, mutable=True)
    
    ##################################################################
    # the global system reserve, for each time period. units are MW. #
    ##################################################################
    
    model.ReserveRequirements = Param(model.TimePeriods, initialize=0.0, within=NonNegativeReals, default=0.0)
    
    
    # previous dispatch
    model.PreviousDispatch = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=0, mutable=True)
    
    ####################################################################################
    # minimum and maximum generation levels, for each thermal generator. units are MW. #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################
    
    model.MinimumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['PMIN'].to_dict())
    
    def maximum_power_output_validator(m, v, g):
       return v >= value(m.MinimumPowerOutput[g])
    
    model.MaximumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, validate=maximum_power_output_validator, initialize=genth_df['PMAX'].to_dict())
    
    #################################################
    # generator ramp up/down rates. units are MW/h. #
    #################################################
    
    # limits for normal time periods
    model.NominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['RAMP_10'].to_dict())
    model.NominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['RAMP_10'].to_dict())
    
    # limits for time periods in which generators are brought on or off-line. 
    # must be no less than the generator minimum output. 
    def at_least_generator_minimum_output_validator(m, v, g):
       return v >= m.MinimumPowerOutput[g]
    
    model.StartupRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=at_least_generator_minimum_output_validator, initialize=genth_df['STARTUP_RAMP'].to_dict())
    model.ShutdownRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=at_least_generator_minimum_output_validator, initialize=genth_df['SHUTDOWN_RAMP'].to_dict())
    
    ##########################################################################################################
    # the minimum number of time periods that a generator must be on-line (off-line) once brought up (down). #
    ##########################################################################################################
    
    model.MinimumUpTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=genth_df['MINIMUM_UP_TIME'].to_dict(), mutable=True)
    model.MinimumDownTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=genth_df['MINIMUM_DOWN_TIME'].to_dict(), mutable=True)
    
    ##############################################
    # Flexible Ramping  parameters                         # 
    ##############################################
    print('Building ramp requirement ...')
    def _flexible_ramp_up_requirement_rule(m, t, s):
        if  [t,s] == [len(m.TimePeriods),value(m.NumTimeSlots)] :
            return max(0.0,(m.FlexibleRampFactor * m.SlotDemand[t,s]))
        elif t<len(m.TimePeriods) and s == value(m.NumTimeSlots):    
            return max(0.0,((1+m.FlexibleRampFactor) *  m.SlotDemand[t+1,1] -  m.SlotDemand[t,s]))
        elif (t<len(m.TimePeriods)) and (s < value(m.NumTimeSlots)):    
            return max(0.0,((1+m.FlexibleRampFactor) *  m.SlotDemand[t,s+1] -  m.SlotDemand[t,s]))
        
    
    def _flexible_ramp_down_requirement_rule(m, t, s):
        if  [t,s] == [len(m.TimePeriods),value(m.NumTimeSlots)] :
            return max(0.0, (m.FlexibleRampFactor *  m.SlotDemand[t,s]))
        elif t<len(m.TimePeriods) and s == value(m.NumTimeSlots):    
            return max(0.0, (m.SlotDemand[t,s] - (1-m.FlexibleRampFactor) *  m.SlotDemand[t+1,1]) )
        elif (t<=len(m.TimePeriods)) and (s < value(m.NumTimeSlots)):    
            return max(0.0, ( m.SlotDemand[t,s] - (1-m.FlexibleRampFactor) *  m.SlotDemand[t,s+1]) ) 
        else:
            pass
    #def initialize_flexible_ramp(model, flexible_ramp_factor=0.0, flexible_ramp_Up_requirement=_flexible_ramp_up_requirement_rule, flexible_ramp_Dn_requirement=_flexible_ramp_down_requirement_rule):
    
    model.FlexibleRampFactor = Param(within=Reals, initialize=FlexibleRampFactor, default=0.0, mutable=True)
    model.FlexibleRampUpRequirement = Param(model.TimePeriods, model.TimeSlots, initialize= RampUpRequirement_dict, within=Reals, default=0.0, mutable=True)    
    model.FlexibleRampDnRequirement = Param(model.TimePeriods, model.TimeSlots, initialize=RampDnRequirement_dict, within=Reals, default=0.0, mutable=True)       
    
    print('Done with ramp requirements' )
    
    #############################################
    # unit on state at t=0 (initial condition). #
    #############################################
    
    # if positive, the number of hours prior to (and including) t=0 that the unit has been on.
    # if negative, the number of hours prior to (and including) t=0 that the unit has been off.
    # the value cannot be 0, by definition.
    
    def t0_state_nonzero_validator(m, v, g):
        return v != 0
    
    model.UnitOnT0State = Param(model.ThermalGenerators, within=Integers, validate=t0_state_nonzero_validator, initialize=genth_df['GEN_STATUS'].to_dict())
    
    def t0_unit_on_rule(m, g):
        return value(m.UnitOnT0State[g]) >= 1
    
    model.UnitOnT0 = Param(model.ThermalGenerators, within=Binary, initialize=t0_unit_on_rule)
    
    #######################################################################################
    # the number of time periods that a generator must initally on-line (off-line) due to #
    # its minimum up time (down time) constraint.                                         #
    #######################################################################################
    
    def initial_time_periods_online_rule(m, g):
       if not value(m.UnitOnT0[g]):
          return 0
       else:
          return min(value(m.NumTimePeriods), \
                     max(0, \
                         value(m.MinimumUpTime[g]) - value(m.UnitOnT0State[g])))
    
    model.InitialTimePeriodsOnLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_online_rule)
    
    def initial_time_periods_offline_rule(m, g):
       if value(m.UnitOnT0[g]):
          return 0
       else:
          return min(value(m.NumTimePeriods), \
                     max(0, \
                         value(m.MinimumDownTime[g]) + value(m.UnitOnT0State[g]))) # m.UnitOnT0State is negative if unit is off
    
    model.InitialTimePeriodsOffLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_offline_rule)
    
    ####################################################################
    # generator power output at t=0 (initial condition). units are MW. #
    ####################################################################
    
    model.PowerGeneratedT0 = Param(model.AllGenerators, within=NonNegativeReals, initialize=gen_df['PMIN'].to_dict())
    
    ##################################################################################################################
    # production cost coefficients (for the quadratic) a0=constant, a1=linear coefficient, a2=quadratic coefficient. #
    ##################################################################################################################
    
    #model.ProductionCostA0 = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=gen_df['COST_0'].to_dict()) # units are $/hr (or whatever the time unit is).
    #model.ProductionCostA1 = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=margcost_df['1'].to_dict()) # units are $/MWhr.
    #model.ProductionCostA2 = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=gen_df['COST_2'].to_dict()) # units are $/(MWhr^2).
    model.BlockMarginalCost = Param(model.ThermalGenerators, model.Blocks, within=NonNegativeReals, initialize=blockmargcost_dict)
    
    
    ##################################################################################
    # shutdown and startup cost for each generator. in the literature, these are often set to 0. #
    ##################################################################################
    
    model.ShutdownCostCoefficient = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['STARTUP'].to_dict()) # units are $.
    
    model.StartupCostCoefficient = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=genth_df['SHUTDOWN'].to_dict()) # units are $.
    
    #
    ################################################################################
    # Spinning and Regulating Reserves
    ###############################################################################
    
    def _reserve_requirement_rule(m, t):
        return m.ReserveFactor * sum(value(m.BusDemand[b,t]) for b in m.LoadBuses)
    
    #def initialize_global_reserves(model, reserve_factor=0.0, reserve_requirement=_reserve_requirement_rule):
    
    model.ReserveFactor = Param(within=Reals, initialize=ReserveFactor, default=0.0, mutable=True)
    model.SpinningReserveRequirement = Param(model.TimePeriods, initialize=_reserve_requirement_rule, within=NonNegativeReals, default=0.0, mutable=True)
    
    def _regulating_requirement_rule(m, t):
        return m.RegulatingReserveFactor * sum(value(m.BusDemand[b,t]) for b in m.LoadBuses)
    
    #def initialize_regulating_reserves_requirement(model, regulating_reserve_factor=0.0, regulating_reserve_requirement=_regulating_requirement_rule):
    
    model.RegulatingReserveFactor = Param(within=Reals, initialize=RegulatingReserveFactor, default=0.0, mutable=True)
    model.RegulatingReserveRequirement = Param(model.TimePeriods, initialize=_regulating_requirement_rule, within=NonNegativeReals, default=0.0, mutable=True)
           
    # Ramp cost
    model.RampCost = Param(model.AllGenerators, initialize=gen_df['RAMP_COST'].to_dict(), within=NonNegativeReals)
    
    
    #*********************************************************************************************************************************************************#
    """VARIABLES"""
    #==============================================================================
    #  VARIABLE DEFINITION
    #==============================================================================
    #def initialize_flexible_ramp_reserves(model):
    model.FlexibleRampUpAvailable = Var(model.ThermalGenerators | model.WindGenerators, model.TimePeriods, model.TimeSlots, initialize=0.0, within=NonNegativeReals)
    model.FlexibleRampDnAvailable = Var(model.ThermalGenerators | model.WindGenerators, model.TimePeriods, model.TimeSlots, initialize=0.0, within=NonNegativeReals)
        
    #def initialize_regulating_reserves(model):
    model.RegulatingReserveUpAvailable = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)    
    model.RegulatingReserveDnAvailable = Var(model.ThermalGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)    
    
    #def initialize_spinning_reserves(model):
    model.SpinningReserveUpAvailable = Var(model.AllGenerators, model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    
    
    
    # indicator variables for each generator, at each time period.
    if linear_status == 1:
        model.UnitOn = Param(model.ThermalGenerators, model.TimePeriods, within=Binary,initialize=0, mutable=True)

    else:
        model.UnitOn = Var(model.ThermalGenerators, model.TimePeriods, within=Binary,initialize=0)
    
    # amount of power produced by each generator, at each time period.
    model.PowerGenerated = Var(model.AllGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    # amount of power produced by each generator, in each block, at each time period.
    model.BlockPowerGenerated = Var(model.ThermalGenerators, model.Blocks, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    
    # maximum power output for each generator, at each time period.
    model.MaximumPowerAvailable = Var(model.AllGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    
    model.MaxWindAvailable = Var(model.WindGenerators, model.TimePeriods,model.TimeSlots, within=NonNegativeReals, initialize=0.0)
    model.WindRpCurtailment = Var(model.WindGenerators, model.TimePeriods,model.TimeSlots, within=NonNegativeReals, initialize=0.0)
    
    ###################
    # cost components #
    ###################
    
    # production cost associated with each generator, for each time period.
    model.ProductionCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    
    # startup and shutdown costs for each generator, each time period.
    model.StartupCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    model.ShutdownCost = Var(model.ThermalGenerators, model.TimePeriods, within=NonNegativeReals, initialize=0.0)
    
    # cost over all generators, for all time periods.
    model.TotalProductionCost = Var(within=NonNegativeReals, initialize=0.0)
    
    # all other overhead / fixed costs, e.g., associated with startup and shutdown.
    model.TotalFixedCost = Var(within=NonNegativeReals, initialize=0.0)
    
    model.BusCurtailment = Var(model.LoadBuses,model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    
    model.Curtailment = Var(model.TimePeriods, initialize=0.0, within=NonNegativeReals)   
    
    model.TotalCurtailment = Var(initialize=0.0, within=NonNegativeReals) 
    
    model.TotalCurtailmentCost = Var(initialize=0.0, within=NonNegativeReals)
    
    model.SpinningReserveShortage = Var(model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    
    model.RegulatingReserveShortage = Var(model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    
    model.FlexibleRampUpShortage = Var(model.TimePeriods, model.TimeSlots, initialize=0.0, within=NonNegativeReals)
    
    model.FlexibleRampDnShortage = Var(model.TimePeriods, model.TimeSlots, initialize=0.0, within=NonNegativeReals)
    
    model.RampingCost = Var(model.TimePeriods, initialize=0.0, within=NonNegativeReals)
    
    #*****************************************************************************************************************************************************#
    """CONSTRAINTS"""
    #==============================================================================
    # CONSTRAINTS
    #==============================================================================
    
    ###################################################
    # Linear unit commitment parameter constraints    #
    ##################################################
    
        
    
    
    ############################################
    # supply-demand constraints                #
    ############################################
    # meet the demand at each time period.
    # encodes Constraint 2 in Carrion and Arroyo.
    
    def enforce_bus_curtailment_limits_rule(m,b,t):
        return m.BusCurtailment[b, t]<= m.BusDemand[b,t]
    model.EnforceBusCurtailmentLimits = Constraint(model.LoadBuses, model.TimePeriods, rule=enforce_bus_curtailment_limits_rule) 
    
    
    def definition_hourly_curtailment_rule(m, t):
       return m.Curtailment[t] == sum(m.BusCurtailment[b, t] for b in m.LoadBuses)
    
    model.DefineHourlyCurtailment = Constraint(model.TimePeriods, rule=definition_hourly_curtailment_rule) 
    
    def production_equals_demand_rule(m, t):
       return sum(m.PowerGenerated[g, t] for g in m.AllGenerators)  == m.Demand[t] - m.Curtailment[t]
    
    model.ProductionEqualsDemand = Constraint(model.TimePeriods, rule=production_equals_demand_rule)
    
    
    def definition_total_curtailment_rule(m):
       return m.TotalCurtailment == sum(m.Curtailment[t] for t in m.TimePeriods)
     
    model.DefineTotalCurtailment = Constraint(rule=definition_total_curtailment_rule)
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
    
    def enforce_generator_output_limits_rule_part_a(m, g, t):
       return m.MinimumPowerOutput[g] * m.UnitOn[g, t] <= m.PowerGenerated[g,t]
    
    model.EnforceGeneratorOutputLimitsPartA = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_a)
    
    def enforce_generator_output_limits_rule_part_b(m, g, t):
       return m.PowerGenerated[g,t] <= m.MaximumPowerAvailable[g, t]
    
    model.EnforceGeneratorOutputLimitsPartB = Constraint(model.AllGenerators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_b)
    
    def enforce_generator_output_limits_rule_part_c(m, g, t):
       return m.MaximumPowerAvailable[g, t] <= m.MaximumPowerOutput[g] * m.UnitOn[g, t]
    
    model.EnforceGeneratorOutputLimitsPartC = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_c)

    def enforce_generator_output_limits_rule_part_d(m, g, t):
       return m.MaximumPowerAvailable[g, t] <= sum(m.PowerForecast[g,t,s] for s in m.TimeSlots)/value(m.NumTimeSlots)
    
    model.EnforceGeneratorOutputLimitsPartD = Constraint(model.NonThermalGenerators, model.TimePeriods, rule=enforce_generator_output_limits_rule_part_d)
    
    
    def enforce_generator_block_output_rule(m, g, t):
       return m.PowerGenerated[g, t] == sum(m.BlockPowerGenerated[g,k,t] for k in m.Blocks) + m.UnitOn[g,t]*margcost_df.loc[g,'Pmax0']
    
    model.EnforceGeneratorBlockOutput = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_generator_block_output_rule)
    
    def enforce_generator_block_output_limit_rule(m, g, k, t):
       return m.BlockPowerGenerated[g,k,t] <= m.BlockSize[g,k]
    
    model.EnforceGeneratorBlockOutputLimit = Constraint(model.ThermalGenerators, model.Blocks, model.TimePeriods, rule=enforce_generator_block_output_limit_rule)
    
    """
    def enforce_renewable_generator_output_limits_rule(m, g, t):
       return  m.PowerGenerated[g, t]<= m.PowerForecast[g,t]
    
    model.EnforceRenewableOutputLimits = Constraint(model.NonThermalGenerators, model.TimePeriods, rule=enforce_renewable_generator_output_limits_rule)
    """
    # impose upper bounds on the maximum power available for each generator in each time period, 
    # based on standard and start-up ramp limits.
    
    # the following constraint encodes Constraint 18 defined in Carrion and Arroyo.
    
    def enforce_max_available_ramp_up_rates_rule(m, g, t):
       # 4 cases, split by (t-1, t) unit status (RHS is defined as the delta from m.PowerGenerated[g, t-1])
       # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound due to unit being off) 
       # (0, 1) - unit switching on:  RHS = startup ramp limit 
       # (1, 0) - unit switching off: RHS = standard ramp limit minus startup ramp limit plus maximum power generated (degenerate upper bound due to unit off)
       # (1, 1) - unit staying on:    RHS = standard ramp limit
       if t == value(m.TimeStart):
          return m.MaximumPowerAvailable[g, t] - m.RegulatingReserveUpAvailable[g,t] - m.SpinningReserveUpAvailable[g,t] <= \
                      m.PowerGeneratedT0[g] +  m.NominalRampUpLimit[g] * m.UnitOnT0[g] + \
                                                  m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOnT0[g]) + \
                                                  m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
       else:
          return m.MaximumPowerAvailable[g, t] -m.RegulatingReserveUpAvailable[g,t] - m.SpinningReserveUpAvailable[g,t] <= \
                      m.PowerGenerated[g, t-1] + m.NominalRampUpLimit[g] * m.UnitOn[g, t-1] + \
                                                  m.StartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t-1]) + \
                                                  m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
    
    model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)
    
    # the following constraint encodes Constraint 19 defined in Carrion and Arroyo.
    
    def enforce_max_available_ramp_down_rates_rule(m, g, t):
       # 4 cases, split by (t, t+1) unit status
       # (0, 0) - unit staying off:   RHS = 0 (degenerate upper bound)
       # (0, 1) - unit switching on:  RHS = maximum generator output minus shutdown ramp limit (degenerate upper bound) - this is the strangest case.
       # (1, 0) - unit switching off: RHS = shutdown ramp limit
       # (1, 1) - unit staying on:    RHS = maximum generator output (degenerate upper bound)
       if t == value(m.TimeEnd):
          return Constraint.Skip
       else:
          return m.MaximumPowerAvailable[g, t] <= \
                 m.MaximumPowerOutput[g] * m.UnitOn[g, t+1] + \
                 m.ShutdownRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t+1])
    
    model.EnforceMaxAvailableRampDownRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_down_rates_rule)
    
    # the following constraint encodes Constraint 20 defined in Carrion and Arroyo.
    
    def enforce_ramp_down_limits_rule(m, g, t):
       # 4 cases, split by (t-1, t) unit status: 
       # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound)
       # (0, 1) - unit switching on:  RHS = standard ramp-down limit minus shutdown ramp limit plus maximum generator output - this is the strangest case.
       # (1, 0) - unit switching off: RHS = shutdown ramp limit 
       # (1, 1) - unit staying on:    RHS = standard ramp-down limit 
       if t == value(m.TimeStart):
          return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] - m.RegulatingReserveDnAvailable[g,t] <= \
                 m.NominalRampDownLimit[g] * m.UnitOn[g, t] + \
                 m.ShutdownRampLimit[g] * (m.UnitOnT0[g] - m.UnitOn[g, t]) + \
                 m.MaximumPowerOutput[g] * (1 - m.UnitOnT0[g])                
       else:
          return m.PowerGenerated[g, t-1] - m.PowerGenerated[g, t] - m.RegulatingReserveDnAvailable[g,t] <= \
                 m.NominalRampDownLimit[g] * m.UnitOn[g, t] + \
                 m.ShutdownRampLimit[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t]) + \
                 m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t-1])             
    
    model.EnforceNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)
    
    #############################################
    # constraints for computing cost components #
    #############################################
    # New ADDITION

    def production_cost_function(m, g, t):
        return m.ProductionCost[g,t] == sum(value(m.BlockMarginalCost[g,k])*(m.BlockPowerGenerated[g,k,t]) for k in m.Blocks) + m.UnitOn[g,t]*margcost_df.loc[g,'nlcost']
    model.ComputeProductionCost = Constraint(model.ThermalGenerators, model.TimePeriods, rule=production_cost_function)
    #---------------------------------------
    
    # compute the per-generator, per-time period production costs. this is a "simple" piecewise linear construct.
    # the first argument to piecewise is the index set. the second and third arguments are respectively the input and output variables. 
    """
    model.ComputeProductionCosts = Piecewise(model.ThermalGenerators * model.TimePeriods, model.ProductionCost, model.PowerGenerated, pw_pts=model.PowerGenerationPiecewisePoints, f_rule=production_cost_function, pw_constr_type='LB')
    """
    # compute the total production costs, across all generators and time periods.
    def compute_total_production_cost_rule(m):
       return m.TotalProductionCost == sum(m.ProductionCost[g, t] for g in m.ThermalGenerators for t in m.TimePeriods)
    
    model.ComputeTotalProductionCost = Constraint(rule=compute_total_production_cost_rule)
    
    # compute the per-generator, per-time period shutdown costs.
    def compute_shutdown_costs_rule(m, g, t):
       if t ==value(m.TimeStart):
          return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (m.UnitOnT0[g] - m.UnitOn[g, t])
       else:
          return m.ShutdownCost[g, t] >= m.ShutdownCostCoefficient[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t])
    
    model.ComputeShutdownCosts = Constraint(model.ThermalGenerators, model.TimePeriods, rule=compute_shutdown_costs_rule)
    
    
    
    def compute_startup_costs_rule(m, g, t):
       if t == value(m.TimeStart):
          return m.StartupCost[g, t] >= m.StartupCostCoefficient[g] * (-m.UnitOnT0[g] + m.UnitOn[g, t])
       else:
          return m.StartupCost[g, t] >= m.StartupCostCoefficient[g] * (-m.UnitOn[g, t-1] + m.UnitOn[g, t])
    
    model.ComputeStartupCosts = Constraint(model.ThermalGenerators, model.TimePeriods, rule=compute_startup_costs_rule)
    
    # compute the total startup and shutdown costs, across all generators and time periods.
    def compute_total_fixed_cost_rule(m):
       return m.TotalFixedCost == sum(m.StartupCost[g, t] + m.ShutdownCost[g, t]  for g in m.ThermalGenerators for t in m.TimePeriods)
    
    model.ComputeTotalFixedCost = Constraint(rule=compute_total_fixed_cost_rule)
    
    def compute_total_curtailment_cost_rule(m):
       return m.TotalCurtailmentCost == sum(100000* m.BusCurtailment[b,t]  for b in m.LoadBuses for t in m.TimePeriods)
    
    model.ComputeTotalCurtailmentCost = Constraint(rule=compute_total_curtailment_cost_rule)
    
    #*****
    
    #############################################
    # constraints for line capacity limits #
    #############################################
    
    print('Building network constraints ...')

    def line_flow_rule(m, l, t):
       # This is an expression of the power flow on bus b in time t, defined here
       # to save time.
       return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g,t] for g in m.AllGenerators) -\
	      sum(ptdf_dict[l][b]*(m.BusDemand[b,t] - m.BusCurtailment[b,t]) for b in m.LoadBuses)
    
    model.LineFlow = Expression(model.EnforcedBranches, model.TimePeriods, rule=line_flow_rule)
	
    def enforce_line_capacity_limits_rule_a(m, l, t):
       return m.LineFlow[l, t] <= m.LineLimits[l]

    def enforce_line_capacity_limits_rule_b(m, l, t):
       return m.LineFlow[l, t] >= -m.LineLimits[l]
    
    #def enforce_line_capacity_limits_rule_a(m, l, t):
    #    return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g,t] for g in m.AllGenerators) - \
    #           sum(ptdf_dict[l][b]*(m.BusDemand[b,t] - m.BusCurtailment[b,t]) for b in m.LoadBuses) <= m.LineLimits[l]
    
    model.EnforceLineCapacityLimitsA = Constraint(model.EnforcedBranches, model.TimePeriods, rule=enforce_line_capacity_limits_rule_a)   
        
    #def enforce_line_capacity_limits_rule_b(m, l, t):
    #    return sum(ptdf_dict[l][m.GenBuses[g]]*m.PowerGenerated[g,t] for g in m.AllGenerators) - \
    #           sum(ptdf_dict[l][b]*(m.BusDemand[b,t] - m.BusCurtailment[b,t]) for b in m.LoadBuses) >= -m.LineLimits[l]
    #           
    model.EnforceLineCapacityLimitsB = Constraint(model.EnforcedBranches, model.TimePeriods, rule=enforce_line_capacity_limits_rule_b)
    
    print('Done with network constraints')
    #######################
    # up-time constraints #
    #######################
    
    # constraint due to initial conditions.
    def enforce_up_time_constraints_initial(m, g):
       if value(m.InitialTimePeriodsOnLine[g]) is 0:
          return Constraint.Skip
       else:
          return sum((1 - m.UnitOn[g, t]) for g in m.ThermalGenerators for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOnLine[g])) == 0.0
    
    #model.EnforceUpTimeConstraintsInitial = Constraint(model.ThermalGenerators, rule=enforce_up_time_constraints_initial)
    
    # constraint for each time period after that not involving the initial condition.
    def enforce_up_time_constraints_subsequent(m, g, t):
       if t <= value(m.InitialTimePeriodsOnLine[g]):
          # handled by the EnforceUpTimeConstraintInitial constraint.
          return Constraint.Skip
       elif t <= (value(m.NumTimePeriods) - value(m.MinimumUpTime[g]) + 1):
          # the right-hand side terms below are only positive if the unit was off in the previous time period but on in this one =>
          # the value is the minimum number of subsequent consecutive time periods that the unit is required to be on.
          if t == value(m.TimeStart):
             return sum(m.UnitOn[g, n] for n in m.TimePeriods if n >= t and n <= (t + value(m.MinimumUpTime[g]) - 1)) >= \
                    (m.MinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOnT0[g]))
          else:
             return sum(m.UnitOn[g, n] for n in m.TimePeriods if n >= t and n <= (t + value(m.MinimumUpTime[g]) - 1)) >= \
                    (m.MinimumUpTime[g] * (m.UnitOn[g, t] - m.UnitOn[g, t-1]))
       else:
          # handle the final (MinimumUpTime[g] - 1) time periods - if a unit is started up in 
          # this interval, it must remain on-line until the end of the time span.
          if t == value(m.TimeStart): # can happen when small time horizons are specified
             return sum((m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOnT0[g])) for n in m.TimePeriods if n >= t) >= 0.0
          else:
             return sum((m.UnitOn[g, n] - (m.UnitOn[g, t] - m.UnitOn[g, t-1])) for n in m.TimePeriods if n >= t) >= 0.0
    
    model.EnforceUpTimeConstraintsSubsequent = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_up_time_constraints_subsequent)
    
    #########################
    # down-time constraints #
    #########################
    
    # constraint due to initial conditions.
    def enforce_down_time_constraints_initial(m, g):
       if value(m.InitialTimePeriodsOffLine[g]) is 0: 
          return Constraint.Skip
       return sum(m.UnitOn[g, t] for g in m.ThermalGenerators for t in m.TimePeriods if t <= value(m.InitialTimePeriodsOffLine[g])) == 0.0
    
    model.EnforceDownTimeConstraintsInitial = Constraint(model.ThermalGenerators, rule=enforce_down_time_constraints_initial)
    
    # constraint for each time period after that not involving the initial condition.
    def enforce_down_time_constraints_subsequent(m, g, t):
       if t <= value(m.InitialTimePeriodsOffLine[g]):
          # handled by the EnforceDownTimeConstraintInitial constraint.
          return Constraint.Skip
       elif t <= (value(m.NumTimePeriods) - value(m.MinimumDownTime[g]) + 1):
          # the right-hand side terms below are only positive if the unit was off in the previous time period but on in this one =>
          # the value is the minimum number of subsequent consecutive time periods that the unit is required to be on.
          if t == value(m.TimeStart):
             return sum((1 - m.UnitOn[g, n]) for n in m.TimePeriods if n >= t and n <= (t + value(m.MinimumDownTime[g]) - 1)) >= \
                    (m.MinimumDownTime[g] * (m.UnitOnT0[g] - m.UnitOn[g, t]))
          else:
             return sum((1 - m.UnitOn[g, n] for n in m.TimePeriods if n >= t and n <= (t + value(m.MinimumDownTime[g]) - 1))) >= \
                    (m.MinimumDownTime[g] * (m.UnitOn[g, t-1] - m.UnitOn[g, t]))
       else:
          # handle the final (MinimumDownTime[g] - 1) time periods - if a unit is shut down in
          # this interval, it must remain off-line until the end of the time span.
          if t == value(m.TimeStart): # can happen when small time horizons are specified
             return sum(((1 - m.UnitOn[g, n]) - (m.UnitOnT0[g] - m.UnitOn[g, t])) for n in m.TimePeriods if n >= t) >= 0.0
          else:
             return sum(((1 - m.UnitOn[g, n]) - (m.UnitOn[g, t-1] - m.UnitOn[g, t])) for n in m.TimePeriods if n >= t) >= 0.0
    
    model.EnforceDownTimeConstraintsSubsequent = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_down_time_constraints_subsequent)
    
    ##--------------------------------------------------------------------------------------------------
    print('Building windoutput constraints ...')
    def enforce_wind_generator_output_limits_b(m, g, t,s):
       return m.PowerGenerated[g,t]    - m.FlexibleRampDnAvailable[g,t,s]     >= 0 
    #***
    def enforce_wind_generator_output_limits_c(m, g, t, s):
        return m.PowerGenerated[g,t] <= sum(m.PowerForecast[g,t,s] for s in m.TimeSlots)/value(m.NumTimeSlots)
    
    """m.MaxWindAvailable[g, t] = Max{m.PowerForecast[g,t],m.PowerForecast[g,t+1]}
    See (1)enforce_max_wind_generator_output_limits_a and (2)enforce_max_wind_generator_output_limits_b
     Notice that a penalty is associated with m.MaxWindAvailable[g,t,s] in the objective"""
    def enforce_max_wind_generator_output_limits_a(m, g, t, s):
        return m.MaxWindAvailable[g,t,s] >= m.PowerForecast[g,t,s] 
    
    def enforce_max_wind_generator_output_limits_b(m, g, t, s):
        if [t,s] == [value(m.TimeEnd),len(m.TimeSlots)]:
            return Constraint.Skip
        elif t<value(m.TimeEnd) and s==len(m.TimeSlots):
            return m.MaxWindAvailable[g,t,s] >= m.PowerForecast[g,t+1,1]
        elif t<=value(m.TimeEnd) and s < len(m.TimeSlots): 
            return m.MaxWindAvailable[g,t,s] >= m.PowerForecast[g,t,s+1] 
    
    
    def enforce_wind_generator_output_limits_a(m, g, t, s):
        
       return m.PowerGenerated[g,t]    + m.FlexibleRampUpAvailable[g,t,s]  <= m.MaxWindAvailable[g,t,s]
    
    
    def enforce_wind_curtailment_a(m, g, t, s):
       if value(s<len(m.TimeSlots)): 
           return m.WindRpCurtailment[g, t, s]  >= m.MaxWindAvailable[g,t,s] - m.PowerForecast[g,t,s+1]
       elif t<value(m.TimeEnd) and s==len(m.TimeSlots):
           return m.WindRpCurtailment[g, t, s]  >= m.MaxWindAvailable[g,t,s] - m.PowerForecast[g,t+1,1]
       else:
           return Constraint.Skip
              
    model.EnforceWindRpCurtailmentConstraints_a = Constraint(model.WindGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_wind_curtailment_a)
           
    def enforce_wind_curtailment_b(m, g, t, s):
        
        return m.FlexibleRampUpAvailable[g,t,s] >= m.WindRpCurtailment[g, t, s]  
    
    model.EnforceWindRpCurtailmentConstraints_b = Constraint(model.WindGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_wind_curtailment_b)
         
    #def enforce_wind_generator_output_limits_a(m, g, t):
    #    if t < (len(m.TimePeriods)):
    #        return m.PowerGenerated[g, t] + m.FlexibleRampUpAvailable[g,t] <= m.MaximumPowerAvailable[g, t+1]
    #    else:
    #        return m.PowerGenerated[g, t] + m.FlexibleRampUpAvailable[g,t] <= m.MaximumPowerAvailable[g, t]
    
    # Add ramping constraints for the 1st hour based on the previous hour's dispatch
    def constrain_generators_first_dispatch_rule_part_a(m, g, t):
    
       if value(m.Start)>1 and value(t)==1:
           return  m.PowerGenerated[g,t] - m.PreviousDispatch[g]<= m.MaximumRamp[g]
           
       else:
           return Constraint.Skip
    
    model.ConstrainGeneratorsFirstDispatchA = Constraint(model.ThermalGenerators, model.TimePeriods, rule=constrain_generators_first_dispatch_rule_part_a)
    
    def constrain_generators_first_dispatch_rule_part_b(m, g, t):
       if value(m.Start)>1 and value(t)==1:
           return  -m.PowerGenerated[g,t] + m.PreviousDispatch[g]<= m.MaximumRamp[g]
           
       else:
           return Constraint.Skip       
    
    model.ConstrainGeneratorsFirstDispatchB = Constraint(model.ThermalGenerators, model.TimePeriods, rule=constrain_generators_first_dispatch_rule_part_b)
    
    
    
    
    
    #add flexible ramp constraint        
    def enforce_flexible_ramp_up_requirement_rule(m, t, s):
        if wrp==0:
            return sum(m.FlexibleRampUpAvailable[g,t,s] for g in (m.ThermalGenerators)) + m.FlexibleRampUpShortage[t,s] == m.FlexibleRampUpRequirement[t,s] #- m.FlexibleRampUpShortage[t,s] 
        else:
            return sum(m.FlexibleRampUpAvailable[g,t,s] for g in (m.ThermalGenerators|m.WindGenerators)) - sum(m.WindRpCurtailment[g,t,s] for g in m.WindGenerators) + m.FlexibleRampUpShortage[t,s] == m.FlexibleRampUpRequirement[t,s] #- m.FlexibleRampUpShortage[t,s] 
      
    def enforce_flexible_ramp_down_requirement_rule(m, t, s):
        if wrp==0:
            return sum(m.FlexibleRampDnAvailable[g,t,s] for g in m.ThermalGenerators) + m.FlexibleRampDnShortage[t,s] >= m.FlexibleRampDnRequirement[t,s] #- m.FlexibleRampDnShortage[t,s]   
        else:
            return sum(m.FlexibleRampDnAvailable[g,t,s] for g in (m.ThermalGenerators  |m.WindGenerators)) + m.FlexibleRampDnShortage[t,s] >= m.FlexibleRampDnRequirement[t,s] #- m.FlexibleRampDnShortage[t,s]   
    
    """
    For conventional generation : enforce_flexible_ramp_down_limits_rule and enforce_flexible_ramp_up_limits_rule
    """
    def enforce_flexible_ramp_down_limits_rule(m, g, t, s):
        if t == m.TimeEnd: #len(m.TimePeriods):
            return m.PowerGenerated[g,t] - m.FlexibleRampDnAvailable[g,t,s] >= m.MinimumPowerOutput[g] * m.UnitOn[g,t]
        elif t<value(m.TimeEnd) and s==len(m.TimeSlots):
            return  m.PowerGenerated[g,t] - m.FlexibleRampDnAvailable[g,t,s] >= m.MinimumPowerOutput[g] * m.UnitOn[g,t]
        elif t<value(m.TimeEnd) and s < len(m.TimeSlots):
            return  m.PowerGenerated[g,t] - m.FlexibleRampDnAvailable[g,t,s] >= m.MinimumPowerOutput[g] * m.UnitOn[g,t+1]
    
    
    def enforce_flexible_ramp_up_limits_rule(m, g, t, s):
        if t == m.TimeEnd:
            return m.PowerGenerated[g,t] + m.FlexibleRampUpAvailable[g,t,s] <= m.MaximumPowerAvailable[g,t]
        elif t<value(m.TimeEnd) and s==len(m.TimeSlots):
            return m.PowerGenerated[g,t] + m.FlexibleRampUpAvailable[g,t,s] <= m.MaximumPowerAvailable[g,t+1]
        elif t<value(m.TimeEnd) and s < len(m.TimeSlots):
            return m.PowerGenerated[g,t] + m.FlexibleRampUpAvailable[g,t,s] <= m.MaximumPowerAvailable[g,t]
        
    #def constraint_for_Flexible_Ramping(model):
    model.EnforceFlexibleRampUpRates = Constraint(model.TimePeriods, model.TimeSlots, rule=enforce_flexible_ramp_up_requirement_rule)
    model.EnforceFlexibleRampDownRates = Constraint(model.TimePeriods, model.TimeSlots, rule=enforce_flexible_ramp_down_requirement_rule)
    model.EnforceFlexibleRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_flexible_ramp_down_limits_rule)
    model.EnforceFlexibleRampUpLimits = Constraint(model.ThermalGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_flexible_ramp_up_limits_rule)    
    model.EnforceWindFlexibleRampUpLimits = Constraint(model.WindGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_wind_generator_output_limits_a)    
    model.EnforceWindFlexibleRampDnLimits = Constraint(model.WindGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_wind_generator_output_limits_b)   
    
    """Model Justification (P_wt + FRU_gt <=   max{P_maxavail_wt, P_maxavail_wt+1}) """
    model.EnforceWindUpperLimits1 = Constraint(model.WindGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_max_wind_generator_output_limits_a)    
    model.EnforceWindUpperLimits2 = Constraint(model.WindGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_max_wind_generator_output_limits_b)  
    model.EnforceWindUpperLimits0 = Constraint(model.WindGenerators, model.TimePeriods, model.TimeSlots, rule=enforce_wind_generator_output_limits_c)  
    
    #---------------------------------------------------------------
    
    # ensure there is sufficient maximal power output available to meet both the 
    # demand and the spinning reserve requirements in each time period.
    # encodes Constraint 3 in Carrion and Arroyo.
    def enforce_reserve_requirements_rule(m, t):
       return sum(m.MaximumPowerAvailable[g,t] for g in m.AllGenerators) >= m.Demand[t] - m.Curtailment[t] + m.RegulatingReserveRequirement[t] + m.SpinningReserveRequirement[t]
    #def enforce_reserve_requirements_rule(m, t):
    #   return sum(m.MaximumPowerAvailable[g, t] for g in m.AllGenerators) >= m.Demand[t] - m.Curtailment[t] + m.RegulatingReserveRequirement[t]-m.RegulatingReserveShortage[t] + m.SpinningReserveRequirement[t]-m.SpinningReserveShortage[t]
    
    model.EnforceReserveRequirements = Constraint(model.TimePeriods, rule=enforce_reserve_requirements_rule)
    
    
    
    ###
    def calculate_spinning_reserve_up_available_per_generator(m, g, t):
        return m.SpinningReserveUpAvailable[g, t]  <= m.MaximumPowerAvailable[g,t] - m.PowerGenerated[g,t]
    
    def enforce_spinning_reserve_requirement_rule(m,  t):
        return sum(m.SpinningReserveUpAvailable[g,t] for g in m.ThermalGenerators) >= m.SpinningReserveRequirement[t] #- m.SpinningReserveShortage[t]
    
    def enforce_SpinningReserve_up_reserve_limit(m, g, t):
         return m.SpinningReserveUpAvailable[g,t]  <= m.UnitOn[g, t]*m.NominalRampUpLimit[g]
    
    def enforce_regulating_up_reserve_requirement_rule(m, t):
         return sum(m.RegulatingReserveUpAvailable[g,t] for g in m.ThermalGenerators) >= m.RegulatingReserveRequirement[t] #- m.RegulatingReserveShortage[t]
     
    def enforce_regulating_down_reserve_requirement_rule(m, t):
         return sum(m.RegulatingReserveDnAvailable[g,t] for g in m.ThermalGenerators) >= m.RegulatingReserveRequirement[t] #- m.RegulatingReserveShortage[t]
    
    
    def enforce_regulating_up_reserve_limit(m, g, t):
         return m.RegulatingReserveUpAvailable[g,t]  <= m.UnitOn[g, t]*m.NominalRampUpLimit[g]
    
    def enforce_regulating_down_reserve_limit(m, g, t):
         return m.RegulatingReserveDnAvailable[g,t]  <= m.UnitOn[g, t]*m.NominalRampDownLimit[g]
    
    
    
    model.CalculateRegulatingReserveUpPerGenerator = Constraint(model.ThermalGenerators, model.TimePeriods, rule=calculate_spinning_reserve_up_available_per_generator)
    model.EnforceSpinningReserveUp = Constraint(model.TimePeriods, rule=enforce_spinning_reserve_requirement_rule) 
    model.EnforceRegulatingUpReserveRequirements = Constraint(model.TimePeriods, rule=enforce_regulating_up_reserve_requirement_rule)
    model.EnforceRegulatingDnReserveRequirements = Constraint(model.TimePeriods, rule=enforce_regulating_down_reserve_requirement_rule)  
    model.EnforceSpiningReserveRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_SpinningReserve_up_reserve_limit)    
    model.EnforceRegulationUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_regulating_up_reserve_limit)  
    model.EnforceRegulationDnRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_regulating_down_reserve_limit)    
    #------------------------------------------------------
    # Reserve and Ramp Shortage Limit Constraints
    def enforce_spinning_reserve_shortage_limits(m, t):
        return m.SpinningReserveRequirement[t] - m.SpinningReserveShortage[t] >=0
    
    def enforce_regulating_reserve_shortage_limits(m, t):
        return m.RegulatingReserveRequirement[t] - m.RegulatingReserveShortage[t] >=0
    
    def enforce_flexible_up_ramp_shortage_limits(m, t):
        return m.FlexibleRampUpRequirement[t] - m.FlexibleRampUpShortage[t] >=0
    
    def enforce_flexible_down_ramp_shortage_limits(m, t):
        return m.FlexibleRampDnRequirement[t] - m.FlexibleRampDnShortage[t] >=0
    
    #model.EnforceSpinningReserveShortageLimits = Constraint(model.TimePeriods, rule=enforce_spinning_reserve_shortage_limits)
    #model.EnforceRegulatingReserveShortageLimits = Constraint(model.TimePeriods, rule=enforce_regulating_reserve_shortage_limits)
    #model.EnforceFlexibleRampUpShortageLimits = Constraint(model.TimePeriods, rule=enforce_flexible_up_ramp_shortage_limits)
    #model.EnforceFlexibleRampDnShortageLimits = Constraint(model.TimePeriods, rule=enforce_flexible_down_ramp_shortage_limits)
    #
    #-------------------------------------------------------------
    #Ramping Cost
    def compute_total_ramping_cost_rule(m,t):
        return m.RampingCost[t] == sum((m.FlexibleRampUpAvailable[g,t,s] + m.FlexibleRampDnAvailable[g,t,s])*m.RampCost[g] for s in m.TimeSlots for g in m.ThermalGenerators | m.WindGenerators) 
    model.EnforceFlexibleRampCost = Constraint(model.TimePeriods, rule=compute_total_ramping_cost_rule)
    # Objectives
    #
    
    def total_cost_objective_rule(m):
       return m.TotalProductionCost + m.TotalFixedCost + m.TotalCurtailmentCost + sum(m.RampingCost[t] for t in m.TimePeriods) +\
   100000000*(sum(m.MaxWindAvailable[g,t,s] for g in m.WindGenerators for t in m.TimePeriods for s in m.TimeSlots)) +\
   10000*(sum(m.FlexibleRampDnShortage[t,s] + m.FlexibleRampUpShortage[t,s] for t in m.TimePeriods for s in m.TimeSlots))
    
    
    model.TotalCostObjective = Objective(rule=total_cost_objective_rule, sense=minimize)

    print('Done with all constraints')
    
    return model
#genfor_df.loc[[operator.eq(genfor_df['Day'][i],day_idx) for i in range(len(genfor_df.index))],:]
    




    
def int2ext(dataframe,start):
    df = dataframe
    df['Hour'] = df['Hour']+start-1
    #print(dataframe)
    #print(df)
    return df

def ext2int(dataframe,start):
    df = dataframe
    df['Hour'] = df['Hour']-start+1
    #print(dataframe)
    #print(df)
    return df
    
    


         
def store_results(Demand, SlotDemand, WindPowerForecasted,WindPowerGenerated,SlotWindPowerGenerated,WindTotalFlexRamp,WindTotalFlexRampDn,TotalFlexRampRequired,TotalFlexRampDnRequired,TotalFlexRampProvided,WindTotalCurtailments):
    ha_dispatch_df = pd.DataFrame(0,index=range(1,len(WindPowerForecasted)+1),columns=['WindPowerForecasted','WindPowerGenerated','Actual_WindTotalFlexRamp','TotalFlexRampRequired', 'TotalFlexRampProvided','Demand'])
    ha_s_dispatch_df = pd.DataFrame(0,index=range(1,len(WindPowerForecasted)*6+1),columns=['WindPowerForecasted','WindPowerGenerated', 'WindTotalCurtailments','Actual_WindTotalFlexRamp','TotalFlexRampRequired', 'TotalFlexRampProvided','Demand'])

    ha_dispatch_df['Demand'] = np.array(Demand)
    ha_dispatch_df['WindPowerGenerated'] = np.array(WindPowerGenerated)
    #ha_dispatch_df['WindTotalFlexRamp'] = np.array(WindTotalFlexRamp)
    
    ha_s_dispatch_df['Demand'] = np.array(SlotDemand).reshape(len(SlotDemand)*6)
    ha_s_dispatch_df['WindPowerForecasted'] = np.array(WindPowerForecasted).reshape(len(WindPowerForecasted)*6)
    ha_s_dispatch_df['TotalFlexRampRequired'] = np.array(TotalFlexRampRequired).reshape(len(TotalFlexRampRequired)*6)
    ha_s_dispatch_df['Actual_WindTotalFlexRamp'] = np.array(WindTotalFlexRamp).reshape(len(TotalFlexRampProvided)*6) 
    with np.errstate(divide='ignore', invalid='ignore'):
    	ha_s_dispatch_df['WindRampContribution'] = np.array(WindTotalFlexRamp).reshape(len(WindTotalFlexRamp)*6)/np.array(TotalFlexRampRequired).reshape(len(TotalFlexRampRequired)*6)
    ha_s_dispatch_df['WindTotalCurtailments'] = np.array(WindTotalCurtailments).reshape(len(WindTotalCurtailments)*6)  
    ha_s_dispatch_df['WindPowerGenerated'] = np.array(SlotWindPowerGenerated).reshape(len(SlotWindPowerGenerated)*6)
    ha_s_dispatch_df['TotalFlexRampDnRequired'] = np.array(TotalFlexRampDnRequired).reshape(len(TotalFlexRampDnRequired)*6)
    ha_s_dispatch_df['Actual_WindTotalFlexRampDn'] = np.array(WindTotalFlexRampDn).reshape(len(WindTotalFlexRampDn)*6) 
    
    # WindCurtailments
    return ha_dispatch_df, ha_s_dispatch_df
    
def plot_generation_mix(Generation_by_fueltype):
    import matplotlib.pyplot as plt
    FuelType = list(Generation_by_fueltype.columns)
    FuelType = [f for f in FuelType if max(Generation_by_fueltype[f])>0]
    x=np.arange(1,25)
    colors=['m','c','b','r','k','g','y']
    
    plt.figure(1)
    for i in range(len(FuelType)):
        plt.plot(x,Generation_by_fueltype[FuelType[i]],color=colors[i], label=FuelType[i], linewidth=2)
    
    plt.xlabel('Time (hour)')
    plt.ylabel('Generation output (MW)')
    
    #plt.ylabel('Generation output (MW)')
    plt.title('System generation shares')
    plt.legend(loc = 'lower center')
    
    plt.show()
    #return figure
 
"""Show individual wind farm ramping"""
"""Day-ahead: clearing prices, RT as well, operating cost for all market layers, market settlement updates"""
    

previous_dispatch = dict()
#def remove_param_constraints(sced_instance, cons_list = ['SetCommitment','SetEnergyBid',\
#                                                         'ReserveUp','ReserveDn',\
#                                                         'UpperDispatchLimit','LowerDispatchLimit',\
#                                                         'BusLoadData','LoadData','GenForecastData']):
#     for con in cons_list:
#         sced_instance.del_component(con)
#         
#     return sced_instance
     
# def reset_sced_parameters(previous_dispatch, ha_instance, sced_instance, bus_slot_load_dict, slot_load_dict, genforren_dict, start, slot, shift=0):
#      PowerForecast = extract_dictionary_for_sced(genforren_dict, 1, 2, 0, slot, 1, shift)
#      print(PowerForecast)
#      sced_instance.SlotDemand = slot_load_dict[1+shift,slot]
#      BusDemand = extract_dictionary_for_sced(bus_slot_load_dict, 1,2, 0, slot, 1, shift)
#      sced_instance.Start=start
#      sced_instance.Slot=slot
     
#      for g in sced_instance.ThermalGenerators:
#          sced_instance.UnitOn[g] = round(value(ha_instance.UnitOn[g,1+shift]))
#          sced_instance.ReserveDn[g] = value(ha_instance.RegulatingReserveDnAvailable[g,1+shift])
         
#          sced_instance.ReserveUp[g] = value(ha_instance.SpinningReserveUpAvailable[g,1+shift]) +\
#                                       value(ha_instance.RegulatingReserveUpAvailable[g,1+shift])
#          sced_instance.EnergyBid[g] = value(ha_instance.PowerGenerated[g,1+shift])      
#          sced_instance.UpperDispatchLimit[g] = value(sced_instance.EnergyBid[g]) + value(sced_instance.ReserveUp[g])
#          sced_instance.LowerDispatchLimit[g] = max(0,value(sced_instance.EnergyBid[g]) - value(sced_instance.ReserveDn[g]))
         
#          if start==1 and slot==1:
#              pass
#          else:
#              sced_instance.PreviousDispatch[g] = max(0,previous_dispatch[g])
#      #print (value(sced_instance.PreviousDispatch[g]))            
         
         
#      for g in sced_instance.NonThermalGenerators:           
#          sced_instance.PowerForecast[g] = PowerForecast[g]
#          sced_instance.EnergyBid[g] = value(ha_instance.PowerGenerated[g,1+shift]) 
         
#      for b in sced_instance.LoadBuses:
#          sced_instance.BusDemand[b] = BusDemand[b]
         
#      return sced_instance    
                                      
                                      
    
        
    
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
                                     


def compute_LMPs(sced_instance, ptdf_dict, shadow_prices, congestion_prices, LMPs, start,slot,shift=0):
    t = start+shift
    s=slot
    shadow_prices[t,s] = get_shadow_price(sced_instance)
    
    for l in sced_instance.EnforcedBranches:
        congestion_prices[l,t,s] = get_congestion_dual(sced_instance)[l]
    
    for b in sced_instance.Buses:
        #print(shadow_prices[t,s],ptdf_dict[l][b], congestion_prices[l,t,s])
        LMPs[b,t,s] = [ b, t, s, shadow_prices[t,s] + sum(ptdf_dict[l][b]*congestion_prices[l,t,s] for l in sced_instance.EnforcedBranches)] 
        
    return LMPs, shadow_prices, congestion_prices


def compute_da_LMPs(da_lp_instance, ptdf_dict, da_shadow_prices, da_congestion_prices, da_LMPs):
    for t in da_lp_instance.TimePeriods:
        print('DA lmp for time:',t)
        da_shadow_prices[t] = get_shadow_price(da_lp_instance,t)
        congestion_duals = get_congestion_dual(da_lp_instance,t)        
        for l in da_lp_instance.EnforcedBranches:
            da_congestion_prices[l,t] = congestion_duals[l]
        
        for b in da_lp_instance.Buses:
            #print(shadow_prices[t,s],ptdf_dict[l][b], congestion_prices[l,t,s])
            da_LMPs[b,t] = [ b, t, da_shadow_prices[t] + sum(ptdf_dict[l][b]*da_congestion_prices[l,t] for l in da_lp_instance.EnforcedBranches)] 
        
    return da_LMPs, da_shadow_prices, da_congestion_prices
 
def get_congestion_dual(my_instance, t=None):
    congestion_dual_dict = dict()  
    for br in my_instance.EnforcedBranches:
            
           try:
               if t==None:
                   dualA = -my_instance.dual[my_instance.EnforceLineCapacityLimitsA[br]]
               else:
                   dualA = -my_instance.dual[my_instance.EnforceLineCapacityLimitsA[br,t]]
           except KeyError:
               #print('errorA')
               dualA = 0
               
           try:
               if t==None:
                   dualB = my_instance.dual[my_instance.EnforceLineCapacityLimitsB[br]]
               else:
                   dualB = my_instance.dual[my_instance.EnforceLineCapacityLimitsB[br,t]]
           except KeyError:
               #print('errorB')
               dualB = 0
        
           congestion_dual_dict[br] = dualA+dualB
           
    return congestion_dual_dict

def get_shadow_price(my_instance, t=None):
    try:
           if t==None:
               shadow_price = my_instance.dual[my_instance.ProductionEqualsDemand]
           else:
               shadow_price = my_instance.dual[my_instance.ProductionEqualsDemand[t]]
    except KeyError:
           #print('error')
           shadow_price = 0
           
    return shadow_price

def aggregate_wind_forecast(genforren_dict, set_name):
    windfor =dict()
    
    for i in range(1,25):
        for j in range(1,7):
            windfor[i,j]=0
            for name in set_name:
                windfor[i,j]+= genforren_dict[name,i,j]
    return windfor

def assign_unit_commitment_status(input_instance, output_instance):
    """ Gets unit statuses from MILP input_instance and assign them to LP output instance"""
    #output_instance.UnitOn = input_instance.UnitOn[g,t]
    for g in output_instance.ThermalGenerators:
        for t in output_instance.TimePeriods:
            output_instance.UnitOn[g,t] = round(input_instance.UnitOn[g,t].value)
    return output_instance

def write_out_lmps_hourly (wrp_status,day_idx,result_path,LMPs, start,shift=0):
    df=pd.DataFrame.from_dict(LMPs,orient='index')
    df.columns=['Bus','Hour','Slot','LMP']
    df.to_csv(os.path.join(result_path,'case_'+str(wrp_status['da'])+str(wrp_status['ha'])+'_sced_LMP_slot_day'+str(day_idx)+'_hour_'+str(start+shift)+'.csv'))
    #LMPs=dict()
    


def get_fuel_type_list(generator_name_list):
	RE_D = re.compile('\d')
	return list(np.unique(([i[:RE_D.search(i).span()[0]] for i in generator_name_list])))


def get_fuel_type(generator_name):
	RE_D = re.compile('\d')
	return generator_name[:RE_D.search(generator_name).span()[0]] 


def compute_wind_scaling_factor(gen_df, wind_penetration):
    
    """
    Scaling Wind Generation
    """    
    wind_penetration_wanted = wind_penetration # 
    wind_penetration_current = sum(gen_df.loc[x ,'PMAX'] for x in gen_df.index if x.startswith('wind'))/sum(gen_df['PMAX'])# 
    wind_scaling_factor = wind_penetration_wanted * (1/wind_penetration_current -1)/(1-wind_penetration_wanted)  
    print('The wind scaling factor is: ', wind_scaling_factor)
    return wind_scaling_factor
