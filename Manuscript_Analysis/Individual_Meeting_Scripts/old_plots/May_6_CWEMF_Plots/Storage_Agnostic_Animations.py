# -*- coding: utf-8 -*-
"""
Created on Thu May  8 07:50:25 2025

@author: amonkar

Script to show the results for the ensemble forecasting for the meeting with animations
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


#%% Dry Storage/Conditions start - Oct 1st 2015
#Read the simulation dataset

# Initialize an empty list to store DataFrames
dry_data = []

# Loop through years from 1996 to 2023
for year in range(1996, 2020):
    
    #Print the year
    print(year)
    
    # Construct the path for each year
    output_folder = f"Five_Year_Runs/dry_start/{year}/"
    output_file = os.path.join(output_folder, 'results.hdf5')
    
    # Load the data for current year
    yearly_data = get_results_sensitivity_number_outside_model(output_file, '')
    yearly_data = yearly_data[['shasta_S', 'shasta_R', 'oroville_S', 'oroville_R', 'trinity_S', \
                               'trinity_R', 'folsom_S', 'folsom_R', 'newmelones_S', 'newmelones_R', \
                                   'sanluisstate_S', 'sanluisfederal_S', 'trinity_diversions', 'delta_TRP_pump', 'caa_SNL_flow']]
    # Append to our list
    dry_data.append(yearly_data)

#Compute total hydropower 
# Initialize an empty list to store DataFrames
sim_hydro_dry = []

for i in range(len(dry_data)):
    
    print(i)
    #Get the storage data set
    storage_input = dry_data[i][['shasta_S', 'trinity_S', 'folsom_S', \
                                 'newmelones_S']]
    storage_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
    storage_input['San Luis'] = dry_data[i]['sanluisstate_S'] + dry_data[i]['sanluisfederal_S']
    
    #Get the releases dataset
    release_input = dry_data[i][['shasta_R', 'trinity_R', 'folsom_R', \
                                 'newmelones_R']]
    release_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
    release_input.loc[:,'Diversions'] = dry_data[i]['trinity_diversions']
    release_input['San Luis'] =  storage_input['San Luis'].diff().mul(-1).fillna(0)
    release_input.loc[(release_input.index.month == 10) & (release_input.index.day == 1), 'San Luis'] = 0
    
    #Compute the hydropower generation 
    cvp_gen = get_CVP_hydro_gen(storage_input, release_input)
    
    #Compute San Luis Pumping and save the value
    san_luis_pump = cvp_gen[["San_Luis", "Oneill"]]
    san_luis_pump = -1*san_luis_pump.clip(upper=0) #Only save pumping values
    san_luis_pump = san_luis_pump.resample('MS').sum()
    
    #Compute the pumping at Tracy
    tracy_efficiency = 232.7 #kWh/AF
    tracy_pump = dry_data[i]['delta_TRP_pump']*tracy_efficiency/1000 #GWh
    tracy_pump = tracy_pump.resample('MS').sum()
    
    #Compute pumping at Dos Amigoes
    dos_amigos_ei = 135.6 #Kwh/AF
    dos_amigos_pump = dry_data[i]['caa_SNL_flow']*dos_amigos_ei/1000 #Gwh
    dos_amigos_pump =  dos_amigos_pump.resample('MS').sum()
    
    #Overall CVP Generation
    cvp_gen = cvp_gen.drop(["San_Luis", "Oneill", "CVP_Gen"], axis=1)  
    cvp_gen['CVP_Gen'] = cvp_gen.sum(axis=1)
    cvp_gen = cvp_gen.resample('MS').sum()
    
    #Classify the values
    cvp_gen['San_Luis_Pump'] = san_luis_pump['San_Luis'] + san_luis_pump['Oneill']
    cvp_gen['Tracy_Pump'] = tracy_pump
    cvp_gen['Dos_Amigos'] = dos_amigos_pump
    cvp_gen['Project_Use'] = (cvp_gen['San_Luis_Pump'] + cvp_gen['Tracy_Pump'] + cvp_gen['Dos_Amigos'])
    cvp_gen['BR'] = cvp_gen['CVP_Gen'] - cvp_gen['Project_Use']
    cvp_gen['BR'] = cvp_gen['BR'].clip(lower=0)
    
    
    sim_hydro_dry.append(cvp_gen)
    
    
    
#%% Wet/High storage conditions start - Oct 1st 2011
#Read the simulation dataset

# Initialize an empty list to store DataFrames
wet_data = []

# Loop through years from 1996 to 2023
for year in range(1996, 2020):
    
    #Print the year
    print(year)
    
    # Construct the path for each year
    output_folder = f"Five_Year_Runs/wet_start/{year}/"
    output_file = os.path.join(output_folder, 'results.hdf5')
    
    # Load the data for current year
    yearly_data = get_results_sensitivity_number_outside_model(output_file, '')
    yearly_data = yearly_data[['shasta_S', 'shasta_R', 'oroville_S', 'oroville_R', 'trinity_S', \
                               'trinity_R', 'folsom_S', 'folsom_R', 'newmelones_S', 'newmelones_R', \
                                   'sanluisstate_S', 'sanluisfederal_S', 'trinity_diversions', 'delta_TRP_pump', 'caa_SNL_flow']]
    # Append to our list
    wet_data.append(yearly_data)

#Compute total hydropower 
# Initialize an empty list to store DataFrames
sim_hydro_wet = []

for i in range(len(wet_data)):
    
    print(i)
    #Get the storage data set
    storage_input = wet_data[i][['shasta_S', 'trinity_S', 'folsom_S', \
                                 'newmelones_S']]
    storage_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
    storage_input['San Luis'] = wet_data[i]['sanluisstate_S'] + wet_data[i]['sanluisfederal_S']
    
    #Get the releases dataset
    release_input = wet_data[i][['shasta_R', 'trinity_R', 'folsom_R', \
                                 'newmelones_R']]
    release_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
    release_input.loc[:,'Diversions'] = wet_data[i]['trinity_diversions']
    release_input['San Luis'] =  storage_input['San Luis'].diff().mul(-1).fillna(0)
    release_input.loc[(release_input.index.month == 10) & (release_input.index.day == 1), 'San Luis'] = 0
    
    #Compute the hydropower generation 
    cvp_gen = get_CVP_hydro_gen(storage_input, release_input)
    
    #Compute San Luis Pumping
    san_luis_pump = cvp_gen[["San_Luis", "Oneill"]]
    san_luis_pump = -1*san_luis_pump.clip(upper=0) #Only save pumping values
    san_luis_pump = san_luis_pump.resample('MS').sum()
    
    #Compute the pumping at Tracy
    tracy_efficiency = 232.7 #kWh/AF
    tracy_pump = wet_data[i]['delta_TRP_pump']*tracy_efficiency/1000 #GWh
    tracy_pump = tracy_pump.resample('MS').sum()
    
    #Compute pumping at Dos Amigoes
    dos_amigos_ei = 135.6 #Kwh/AF
    dos_amigos_pump = wet_data[i]['caa_SNL_flow']*dos_amigos_ei/1000 #Gwh
    dos_amigos_pump =  dos_amigos_pump.resample('MS').sum()
    
    cvp_gen = cvp_gen.drop(["San_Luis", "Oneill", "CVP_Gen"], axis=1)  
    cvp_gen['CVP_Gen'] = cvp_gen.sum(axis=1)
    cvp_gen = cvp_gen.resample('MS').sum()
    
    #Classify the values
    cvp_gen['San_Luis_Pump'] = san_luis_pump['San_Luis'] + san_luis_pump['Oneill']
    cvp_gen['Tracy_Pump'] = tracy_pump
    cvp_gen['Dos_Amigos'] = dos_amigos_pump
    cvp_gen['Project_Use'] = (cvp_gen['San_Luis_Pump'] + cvp_gen['Tracy_Pump'] + cvp_gen['Dos_Amigos'])
    cvp_gen['BR'] = cvp_gen['CVP_Gen'] - cvp_gen['Project_Use']
    cvp_gen['BR'] = cvp_gen['BR'].clip(lower=0)
    
    sim_hydro_wet.append(cvp_gen)
    
    

# %%    CVP Total Hydropower Generation

#Compute the exceedances (Wet)
scenario_matrix = np.array([data['CVP_Gen'] for data in sim_hydro_wet])
p10_wet = np.percentile(scenario_matrix, 10, axis=0)
p90_wet = np.percentile(scenario_matrix, 90, axis=0)
p25_wet = np.percentile(scenario_matrix, 25, axis=0)
p75_wet = np.percentile(scenario_matrix, 75, axis=0)
p50_wet = np.percentile(scenario_matrix, 50, axis=0)

#Compute the exceedances (Dry)
scenario_matrix = np.array([data['CVP_Gen'] for data in sim_hydro_dry])
p10_dry = np.percentile(scenario_matrix, 10, axis=0)
p90_dry = np.percentile(scenario_matrix, 90, axis=0)
p25_dry = np.percentile(scenario_matrix, 25, axis=0)
p75_dry = np.percentile(scenario_matrix, 75, axis=0)
p50_dry = np.percentile(scenario_matrix, 50, axis=0)

#Create the index
Days = range(0, 1827)

#Define start data
start_date = pd.Timestamp('2015-10-01')  # Assuming this is the start of your data
oct1_dates = [pd.Timestamp(f'{year}-10-01') for year in range(2015, 2021)]

# Empty Plot
plt.figure(figsize = (15, 8))
for year in range(2015, 2020 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_dry[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0)  
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_wet[i]['CVP_Gen'], 
             'b-', linewidth=1, alpha=0)  
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_dry, p90_dry, 
                 color='r', alpha=0.0, label='Low Storage Start (Oct 2015)')
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_wet, p90_wet, 
                 color='b', alpha=0.0, label='High Storage Start (Oct 2011)')
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_wet,
         'b-', linewidth=0) 
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_dry,
         'r-', linewidth=0) 
plt.ylabel("Monthly Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
tick_positions = oct1_dates
tick_labels = [f'Year {year}' for year in range(1,6)]
plt.xticks(tick_positions[:-1], tick_labels, fontsize=18)  # Remove the last date as it's just for the boundary
plt.yticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


# Low Storage Start
plt.figure(figsize = (15, 8))
for year in range(2015, 2020 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_dry[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.2)  
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_wet[i]['CVP_Gen'], 
             'b-', linewidth=1, alpha=0)  
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_dry, p90_dry, 
                 color='r', alpha=0.3, label='Low Storage Start (Oct 2015)')
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_wet, p90_wet, 
                 color='b', alpha=0.0, label='High Storage Start (Oct 2011)')
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_wet,
         'b-', linewidth=0) 
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_dry,
         'r-', linewidth=2, alpha=1, linestyle = 'dashdot') 
plt.ylabel("Monthly Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
tick_positions = oct1_dates
tick_labels = [f'Year {year}' for year in range(1,6)]
plt.xticks(tick_positions[:-1], tick_labels, fontsize=18)  # Remove the last date as it's just for the boundary
plt.yticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()




# All plots
plt.figure(figsize = (15, 8))
for year in range(2015, 2020 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_dry[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.2)  
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_wet[i]['CVP_Gen'], 
             'b-', linewidth=1, alpha=0.2)  
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_dry, p90_dry, 
                 color='r', alpha=0.3, label='Low Storage Start (Oct 2015)')
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_wet, p90_wet, 
                 color='b', alpha=0.3, label='High Storage Start (Oct 2011)')
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_wet,
         'b-', linewidth=2, alpha=1, linestyle = 'dashdot') 
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_dry,
         'r-', linewidth=2, alpha=1, linestyle = 'dashdot') 
plt.ylabel("Monthly Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
tick_positions = oct1_dates
tick_labels = [f'Year {year}' for year in range(1,6)]
plt.xticks(tick_positions[:-1], tick_labels, fontsize=18)  # Remove the last date as it's just for the boundary
plt.yticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()



# All plots
plt.figure(figsize = (15, 8))
for year in range(2015, 2020 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_dry[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0)  
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_wet[i]['CVP_Gen'], 
             'b-', linewidth=1, alpha=0)  
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_dry, p90_dry, 
                 color='r', alpha=0.3, label='Low Storage Start (Oct 2015)')
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_wet, p90_wet, 
                 color='b', alpha=0.3, label='High Storage Start (Oct 2011)')
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_wet,
         'b-', linewidth=2, alpha=1, linestyle = 'dashdot') 
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_dry,
         'r-', linewidth=2, alpha=1, linestyle = 'dashdot') 
plt.ylabel("Monthly Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
tick_positions = oct1_dates
tick_labels = [f'Year {year}' for year in range(1,6)]
plt.xticks(tick_positions[:-1], tick_labels, fontsize=18)  # Remove the last date as it's just for the boundary
plt.yticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


#%%



start_date = pd.Timestamp('2015-10-01') 
end_date = pd.Timestamp('2016-10-01')
monthly_ticks = pd.date_range(start=start_date, end=end_date, freq='MS')
month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']

# All plots
plt.figure(figsize = (15, 8))
for year in range(2015, 2020 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_dry[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0)  
for i in range(len(sim_hydro_dry)):
    plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, sim_hydro_wet[i]['CVP_Gen'], 
             'b-', linewidth=1, alpha=0)  
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_dry, p90_dry, 
                 color='r', alpha=0.3, label='Low Storage Start (Oct 2015)')
plt.fill_between(sim_hydro_dry[0]['CVP_Gen'].index, p10_wet, p90_wet, 
                 color='b', alpha=0.3, label='High Storage Start (Oct 2011)')
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_wet,
         'b-', linewidth=2, alpha=1, linestyle = 'dashdot') 
plt.plot(sim_hydro_dry[0]['CVP_Gen'].index, p50_dry,
         'r-', linewidth=2, alpha=1, linestyle = 'dashdot') 
plt.ylabel("Monthly Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.xticks(monthly_ticks, month_labels, fontsize=22, rotation=45)
tick_positions = oct1_dates
tick_labels = [f'Year {year}' for year in range(1,6)]
plt.yticks(fontsize=18)
plt.xlim(pd.Timestamp('2015-09-25'), pd.Timestamp('2016-10-01') )
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid(True)
plt.tight_layout()


#%%
