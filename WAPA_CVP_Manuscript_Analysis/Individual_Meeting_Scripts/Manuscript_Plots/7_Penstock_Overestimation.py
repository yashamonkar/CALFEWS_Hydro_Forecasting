# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:39:21 2025

@author: amonkar

Code to analyze the validity of the Uniform Monthly Release Assumption. 
This code is for Shasta only. 

Input Data Sources:- 
1. Daily Releases - CDEC 
2. Penstock Capacity - 
3. Historic Water Year types - https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST
"""


# %% Initial Loading

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)


#Historic Water Year Types 
wy_types = pd.read_csv("Yash\Misc_Data\Water_Year_Types.csv", index_col = 0)
wy_types.index = pd.to_datetime(wy_types.index)
wy_types = wy_types[wy_types.index > '1995-09-30']
wy_types.index = wy_types.index + pd.offsets.MonthEnd(0) #Align index to the end of the month. 

#Global Hyperparameters
cfs_to_afd = 1.983  #Unit conversion CFS to TAFD. 
shasta_penstock = 17600 #CFS

# %% Analysis

### -------------------------Daily Time Step ---------------------------------#
#Spill Analysis
shasta_daily_spill = (input_data['SHA_otf']-shasta_penstock).clip(lower=0)
shasta_daily_spill = shasta_daily_spill*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day
shasta_daily_spill = shasta_daily_spill.resample('M').sum() #Aggregate to monthly

#Flow through the penstock
shasta_daily_penstock = input_data['SHA_otf'].clip(upper=shasta_penstock)
shasta_daily_penstock = shasta_daily_penstock*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day
shasta_daily_penstock = shasta_daily_penstock.resample('M').sum() #Aggregate to monthly

### -------------------------Monthly Time Step ---------------------------------#
#Aggregate total releases to the monthly time-step
shasta_monthly_flow = pd.DataFrame({
    'Flow': input_data['SHA_otf'].resample('M').sum(), #Total Monthly Releases
    'DPM': input_data['SHA_otf'].resample('M').count() #Days per month
})

#Spill Analysis
shasta_monthly_spill = (shasta_monthly_flow['Flow']-shasta_monthly_flow['DPM']*shasta_penstock).clip(lower=0)
shasta_monthly_spill = shasta_monthly_spill*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day

#Flow through the penstock
shasta_monthly_penstock = shasta_monthly_flow['Flow'].clip(upper = shasta_monthly_flow['DPM']*shasta_penstock)
shasta_monthly_penstock = shasta_monthly_penstock*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day


# %% Metric Calculations 

#------------------Metric I -- Underestimation of spill-----------------------#
spill = 100*(sum(shasta_daily_spill) - sum(shasta_monthly_spill))/sum(shasta_monthly_spill)
f'Underestimation of spill for Shasta is {round(spill,2)}%'

#--------------Metric II -- Overestimation of penstock flows------------------#
overestimate = 100*(sum(shasta_monthly_penstock) - sum(shasta_daily_penstock))/sum(shasta_daily_penstock)
f'Over-estimation of penstock flow for Shasta is {round(overestimate,2)}%'


#-------------Monthly Distribution of penstock flow overestimation------------#
penstock_overestimate = pd.DataFrame({
    'Daily': shasta_daily_penstock,
    'Monthly': shasta_monthly_penstock, 
    'WYT': wy_types['SAC_Index']
})

#Extract the month from the date. 
penstock_overestimate['Month'] = pd.to_datetime(penstock_overestimate.index)
penstock_overestimate['Month'] = penstock_overestimate['Month'].dt.month

#Code to get the aggregated amounts
penstock_overestimate.groupby('Month').apply(lambda x:round((x['Monthly'].sum() - x['Daily'].sum()), 0))
penstock_overestimate.groupby('WYT').apply(lambda x:round((x['Monthly'].sum() - x['Daily'].sum()), 0))
penstock_overestimate.groupby(['WYT', 'Month']).apply(lambda x:round((x['Monthly'].sum() - x['Daily'].sum()), 0))
penstock_overestimate['Monthly'].sum() - penstock_overestimate['Daily'].sum()


#Penstock overestimates in Jan-Apr
filtered_penstock_overestimate = penstock_overestimate[penstock_overestimate['Month'].isin([1, 2, 3, 4])]
overestimate = 100*(filtered_penstock_overestimate['Monthly'].sum() - filtered_penstock_overestimate['Daily'].sum()) / filtered_penstock_overestimate['Daily'].sum()
f'Over-estimation of penstock flow for Shasta during Jan-Apr is {round(overestimate,2)}%'

#Compute the values


#Function to calculate overestimate value
def calculate_penstock_overestimate(group):
    daily_sum = group['Daily'].sum()
    monthly_sum = group['Monthly'].sum()
    return round(100 * (monthly_sum - daily_sum) / daily_sum,2)


# Calculate for all WYTs (by Month)
all_wyt_by_month = penstock_overestimate.groupby('Month').apply(calculate_penstock_overestimate).reset_index(name='Calculation')
all_wyt_by_month['WYT'] = 'All Types'

# Calculate for all Months (by WYT)
all_months_by_wyt = penstock_overestimate.groupby('WYT').apply(calculate_penstock_overestimate).reset_index(name='Calculation')
all_months_by_wyt['Month'] = 'All Months'

# Calculate for all WYTs and all Months
all_wyt_all_months = pd.DataFrame({
    'WYT': ['All Types'],
    'Month': ['All Months'],
    'Calculation': [calculate_penstock_overestimate(penstock_overestimate)]
})

#Individual Months and Water Year Tyopes. 
months_by_wyt = penstock_overestimate.groupby(['WYT', 'Month']).apply(calculate_penstock_overestimate).reset_index(name='Calculation')




# %% Visual Analysis


# Penstock Flow Scatterplot
plt.figure(figsize=(10, 6))
max_val = max(shasta_monthly_penstock.max(), shasta_daily_penstock.max())
min_val = min(shasta_monthly_penstock.min(), shasta_daily_penstock.min())
plt.fill_between([min_val, max_val*2], [min_val, max_val*2], [max_val*2, max_val*2], 
                 color='red', alpha=0.1, label='Penstock Flow Overestimation')
plt.scatter(shasta_daily_penstock, shasta_monthly_penstock, label="Monthly Penstock Flows (TAF)")
plt.plot([min_val, max_val*1.1], [min_val, max_val*1.1], color='red', linestyle='--')
plt.xlim([min_val, max_val*1.05]) 
plt.ylim([min_val, max_val*1.05]) 
plt.xlabel('Daily CDEC Release Data', fontsize = 16)
plt.ylabel('Monthly CDEC Release Data', fontsize = 16)
plt.legend(loc='lower right', fontsize=12, ncol=1)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True, alpha = 0.5)
plt.show()



# %% Function to compute the penstock flow overestimations

#outflows = input_data['SHA_otf']*cfs_to_afd/1000
#pen_cap = 34.9

def get_overestimation(outflows, pen_cap):
    '''
    #Enter both outflows and penstock capacity in TAF
 
    Returns
    -------
    Overestimation

    '''
    
    ### -------------------------Daily Time Step ---------------------------------#
    #Flow through the penstock
    daily_penstock = outflows.clip(upper=pen_cap)
    daily_penstock = daily_penstock.resample('M').sum() #Aggregate to monthly
    
    ### -------------------------Monthly Time Step ---------------------------------#
    #Aggregate total releases to the monthly time-step
    monthly_flow = pd.DataFrame({
        'Flow': outflows.resample('M').sum(), #Total Monthly Releases
        'DPM': outflows.resample('M').count() #Days per month
    })

    #Flow through the penstock
    monthly_penstock = monthly_flow['Flow'].clip(upper = monthly_flow['DPM']*pen_cap)
    
    #Compute the overestimation value
    return 100*(sum(monthly_penstock) - sum(daily_penstock))/sum(daily_penstock)
 
#Shata
get_overestimation(outflows = input_data['SHA_otf']*cfs_to_afd/1000, 
                   pen_cap = 34.9)

#Folsom
get_overestimation(outflows = input_data['FOL_otf']*cfs_to_afd/1000, 
                   pen_cap = 13.682)

#New Melones
get_overestimation(outflows = input_data['NML_otf']*cfs_to_afd/1000, 
                   pen_cap = 16.450)
    
#Trinity
get_overestimation(outflows = input_data['TRT_otf']*cfs_to_afd/1000, 
                   pen_cap = 14.6)    
