# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:54:34 2025

@author: amonkar
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS_Hydro_Forecasting'
os.chdir(working_directory)


# import libraries
import numpy as np
import pandas as pd
import h5py
import json
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import compress
from datetime import datetime
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec
from scipy import stats

#Hyper-parameters
cfs_tafd = 2.29568411*10**-5 * 86400 / 1000


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)



#%% Compute the exceedance probabilities

#Initialize the empty dataframes
p10 = []
p90 = []
p50 = []
p75 = []
p25 = []
p95 = []

for year in range(1996,2024):
    
    print(year)
    
    # Initialize an empty list to store DataFrames
    san_luis = []
    
    #Second loop
    for all_years in range(1996,2024):
        SL = pd.read_csv(f"Annual_Ensembles/San_Luis/{year}/{all_years}/Daily_San_Luis_Outputs.csv", index_col =0)
        
        # Append to our list
        san_luis.append(SL)
    
    #Exit the loop and compute the exceedances for that year
    scenario_matrix = np.array([data['Storage'] for data in san_luis])
    p10_sub = np.percentile(scenario_matrix, 10, axis=0)
    p20_sub = np.percentile(scenario_matrix, 20, axis=0)
    p90_sub = np.percentile(scenario_matrix, 90, axis=0)
    p50_sub = np.percentile(scenario_matrix, 50, axis=0)
    p75_sub = np.percentile(scenario_matrix, 75, axis=0)
    p25_sub = np.percentile(scenario_matrix, 25, axis=0)
    p95_sub = np.percentile(scenario_matrix, 95, axis=0)
    
    
    #Save the exceedances in a bigger loop
    p10.append(p10_sub)
    p90.append(p90_sub)
    p50.append(p50_sub)
    p75.append(p75_sub)
    p25.append(p25_sub)
    p95.append(p95_sub)
    
    
    
p10_array = np.concatenate(p10)
p90_array = np.concatenate(p90)
p50_array = np.concatenate(p50)
p75_array = np.concatenate(p75)
p25_array = np.concatenate(p25)
p95_array = np.concatenate(p95)

# Create the DataFrame with the three percentile columns
exceedances = pd.DataFrame({
    'p10': p10_array,
    'p50': p50_array,
    'p90': p90_array,
    'p75': p75_array,
    'p25': p25_array,
    'p95': p95_array
})
    
#Set the index for the exceedance data frame
total_days = len(exceedances)
start_date = '1995-10-01'
date_index = pd.date_range(start=start_date, periods=total_days, freq='D')
exceedances.index = date_index    


#-----------------------------------------------------------------------------#
#All Years + Exceedance + Median Generation Value + EIA 
plt.figure(figsize = (15, 8)) 
plt.plot(input_data.index, input_data['SL_storage']/1000, label = "CDEC", color='b')
plt.plot(exceedances.index, exceedances['p50'], 
         linewidth=2, alpha=1, linestyle = 'dashdot', label = "Median Forecast") 
plt.fill_between(exceedances.index, exceedances['p10'], exceedances['p90'], 
                 color='r', alpha=0.5, label='10th-90th Ensemble Range')
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Year", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0, 2100)  # Set y-axis limits from 0 to 2000
plt.xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2023-10-31'))  # Set x-axis limits
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = True)
plt.tight_layout()

