# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 20:46:49 2025

@author: amonkar

Script to show the results for the ensemble forecasting for the meeting. 
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
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
from calfews_src import *
from calfews_src.visualizer import Visualizer
import shutil
from scipy.stats import linregress


#Extract custom functions
from WAPA_Runs.post_processing.functions.get_CALFEWS_results import get_results_sensitivity_number_outside_model
from WAPA_Runs.post_processing.functions.get_comparison_plots import get_comparison_plots
from WAPA_Runs.post_processing.functions.get_CVP_hydro_gen import get_CVP_hydro_gen


#Hyper-parameters
cfs_tafd = 2.29568411*10**-5 * 86400 / 1000

# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)

#Read the WAPA generation dataset
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)
eia = eia/1000
eia.index = pd.to_datetime(eia.index)
eia = eia.drop(['W R Gianelli', 'ONeill'], axis=1)
eia['CVP_Gen'] = eia.sum(axis=1)
eia = eia[eia.index > pd.Timestamp("2018-09-30")]
eia = eia[eia.index < pd.Timestamp("2023-10-01")]

#%% 
#Read the simulation dataset

# Initialize an empty list to store DataFrames
all_data = []

# Loop through years from 1996 to 2023
for year in range(1996, 2020):
    
    #Print the year
    print(year)
    
    # Construct the path for each year
    output_folder = f"Five_Year_Runs/results/{year}/"
    output_file = os.path.join(output_folder, 'results.hdf5')
    
    # Load the data for current year
    yearly_data = get_results_sensitivity_number_outside_model(output_file, '')
    yearly_data = yearly_data[['shasta_S', 'shasta_R', 'oroville_S', 'oroville_R', 'trinity_S', \
                               'trinity_R', 'folsom_S', 'folsom_R', 'newmelones_S', 'newmelones_R', \
                                   'sanluisstate_S', 'sanluisfederal_S', 'trinity_diversions', 'delta_TRP_pump']]
    # Append to our list
    all_data.append(yearly_data)

# %%    Shasta Storage Levels

input_subset = input_data[input_data.index > pd.Timestamp("2018-09-30")]
    
#Visualize the results for Shasta Storage Levels
plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', 
         linewidth=2, label = "CDEC Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)

# Plot all scenarios with alpha for transparency
for i in range(len(all_data)):
    plt.plot(all_data[i]['shasta_S'].index, all_data[i]['shasta_S'], 
             'r-', linewidth=1, alpha=0.3)  # Use blue with transparency

#Plotting the exceedances
scenario_matrix = np.array([data['shasta_S'] for data in all_data])
p10 = np.percentile(scenario_matrix, 10, axis=0)
p90 = np.percentile(scenario_matrix, 90, axis=0)
p25 = np.percentile(scenario_matrix, 25, axis=0)
p75 = np.percentile(scenario_matrix, 75, axis=0)
plt.fill_between(all_data[0]['shasta_S'].index, p10, p90, 
                 color='r', alpha=0.2, label='10-90th Exceedance')
plt.fill_between(all_data[0]['shasta_S'].index, p25, p75, 
                 color='r', alpha=0.5, label='25-75th Exceedance')
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Storage Levels \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


# %%    Folsom Storage Levels

    
#Visualize the results for Shasta Storage Levels
plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_storage']/1000, 'b-', 
         linewidth=2, label = "CDEC Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)

# Plot all scenarios with alpha for transparency
for i in range(len(all_data)):
    plt.plot(all_data[i]['folsom_S'].index, all_data[i]['folsom_S'], 
             'r-', linewidth=1, alpha=0.3)  # Use blue with transparency

#Plotting the exceedances
scenario_matrix = np.array([data['folsom_S'] for data in all_data])
p10 = np.percentile(scenario_matrix, 10, axis=0)
p90 = np.percentile(scenario_matrix, 90, axis=0)
p25 = np.percentile(scenario_matrix, 25, axis=0)
p75 = np.percentile(scenario_matrix, 75, axis=0)
plt.fill_between(all_data[0]['folsom_S'].index, p10, p90, 
                 color='r', alpha=0.2, label='10-90th Exceedance')
plt.fill_between(all_data[0]['folsom_S'].index, p25, p75, 
                 color='r', alpha=0.5, label='25-75th Exceedance')
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Storage Levels \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()



#%% Compute total hydropower 
# Initialize an empty list to store DataFrames
sim_hydro = []

for i in range(len(all_data)):
    
    print(i)
    #Get the storage data set
    storage_input = all_data[i][['shasta_S', 'trinity_S', 'folsom_S', \
                                 'newmelones_S']]
    storage_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
    storage_input['San Luis'] = all_data[i]['sanluisstate_S'] + all_data[i]['sanluisfederal_S']
    
    #Get the releases dataset
    release_input = all_data[i][['shasta_R', 'trinity_R', 'folsom_R', \
                                 'newmelones_R']]
    release_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
    release_input.loc[:,'Diversions'] = all_data[i]['trinity_diversions']
    release_input['San Luis'] =  storage_input['San Luis'].diff().mul(-1).fillna(0)
    release_input.loc[(release_input.index.month == 10) & (release_input.index.day == 1), 'San Luis'] = 0
    
    #Compute the hydropower generation 
    cvp_gen = get_CVP_hydro_gen(storage_input, release_input)
    cvp_gen = cvp_gen.drop(["San_Luis", "Oneill", "CVP_Gen"], axis=1)  
    cvp_gen['CVP_Gen'] = cvp_gen.sum(axis=1)
    cvp_gen = cvp_gen.resample('MS').sum()
    
    sim_hydro.append(cvp_gen)
    
    

# %%    CVP Hydropower Generation

    
#Visualize the results for Shasta Storage Levels
plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=2, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)

# Plot all scenarios with alpha for transparency
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  # Use blue with transparency

#Plotting the exceedances
scenario_matrix = np.array([data['CVP_Gen'] for data in sim_hydro])
p10 = np.percentile(scenario_matrix, 10, axis=0)
p90 = np.percentile(scenario_matrix, 90, axis=0)
p25 = np.percentile(scenario_matrix, 25, axis=0)
p75 = np.percentile(scenario_matrix, 75, axis=0)
#plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p25, p75, 
#                 color='r', alpha=0.5, label='25-75th Exceedance')
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()
    
    
