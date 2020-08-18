# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 08:57:32 2018

@author: ksedzro
"""
import os

def get_global_parameters():

    """day_idx, data_path, result_path, wrp_status, input_mode, casename,wind_penetration = get_global_parameters()"""

    casename = '5-bus_system'
    #casename = 'Texas_2000-bus_system'
    day_idx = 2

    os.chdir('..') # Move up one directory

    cwd = os.getcwd() # Get current directory
    data_path = os.path.join(cwd,'Data',casename,'Input-data')
    result_path = os.path.join(cwd,'Data',casename,'Results')
    wrp_status = {'da':1, 'ha':1}
    input_mode = 'dynamic'
    wind_penetration = 25
    if wind_penetration>1:
        wind_penetration = wind_penetration/100
        
    return day_idx, data_path, result_path, wrp_status, input_mode, casename, wind_penetration

	 
     
