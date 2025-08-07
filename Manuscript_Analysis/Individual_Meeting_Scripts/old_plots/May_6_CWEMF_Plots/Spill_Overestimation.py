# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:26:19 2025

@author: amonkar

Code to estimate the spill overestimation at the monthly versus daily time step. 
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)

cfs_tafd = 2.29568411*10**-5 * 86400 / 1000

# %% Spill Overestimation --- All Reservoirs

def spill_overest(inflows, penstock):
    '''
    Parameters
    ----------
    inflows : TYPE
        Units: TAF
    penstock : TYPE
        Units: TAF

    Returns
    -------
    1. Spill Overestimation in %

    '''
    
    #Daily Time Step Model
    spill = (inflows-penstock).clip(lower=0)
    daily_spill = spill.resample('M').sum()

    #Monthly Time Step Models
    monthly_inflows = inflows.resample('M').sum()
    monthly_spill = (monthly_inflows - monthly_inflows.index.day*penstock).clip(lower=0)
    
    if spill.sum() == 0:
        ans = 0 
    else:
        ans = 100*(spill.sum() - monthly_spill.sum())/spill.sum()
    
    return ans


spill_overest(input_data['SHA_otf']*cfs_tafd,34.909) #Shasta
spill_overest(input_data['FOL_otf']*cfs_tafd,13.682) #Folsom
spill_overest(input_data['NML_otf']*cfs_tafd,16.450) #New Melones
spill_overest(input_data['TRT_otf']*cfs_tafd,7.317) #Trinity


# %% Penstock flows --- All Reservoirs


def penstock_flows(inflows, penstock):
    '''
    Parameters
    ----------
    inflows : TYPE
        Units: TAF
    penstock : TYPE
        Units: TAF

    Returns
    -------
    1. Penstock Underestimation in %

    '''
    
    #Daily Time Step Model
    daily_penstock = inflows.clip(upper=penstock)
    daily_penstock = daily_penstock.resample('M').sum()

    #Monthly Time Step Models
    monthly_inflows = inflows.resample('M').sum()
    monthly_penstock = monthly_inflows.clip(upper=penstock*monthly_inflows.index.day)
    
    #Subset to Jan-April
    monthly_penstock = monthly_penstock[monthly_penstock.index.month.isin([1, 2, 3, 4])]
    daily_penstock = daily_penstock[daily_penstock.index.month.isin([1, 2, 3, 4])]

    ans = 100*(monthly_penstock.sum() - daily_penstock.sum())/monthly_penstock.sum()
    
    return ans


penstock_flows(input_data['SHA_otf']*cfs_tafd,34.909) #Shasta
penstock_flows(input_data['FOL_otf']*cfs_tafd,13.682) #Folsom
penstock_flows(input_data['NML_otf']*cfs_tafd,16.450) #New Melones
penstock_flows(input_data['TRT_otf']*cfs_tafd,7.317) #Trinity