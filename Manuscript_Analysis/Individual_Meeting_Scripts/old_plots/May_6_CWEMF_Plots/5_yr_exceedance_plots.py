# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:54:12 2025

@author: amonkar

Script to show the results for the ensemble forecasting for the meeting with animations
Note:- THe first prefence and base resource data are added manually
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

#First Preference & COTP Losses
fp_use = [17.8, 19.5, 20.6, 21.5, 19.5, 20.1, 17.3, 17.1, 17.4, 19.7, 19.3, 16.6]  #GWh/Month
cotp_losses = [8.6, 10.4, 8, 8.6, 7.4, 9.5, 8.8, 16.3, 19.8, 16, 13.7, 13.6]
fp_use = fp_use * 5
cotp_losses = cotp_losses * 5

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
                                   'sanluisstate_S', 'sanluisfederal_S', 'trinity_diversions', 'delta_TRP_pump', 'caa_SNL_flow']]
    # Append to our list
    all_data.append(yearly_data)








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
    
    #Compute San Luis Pumping
    san_luis_pump = cvp_gen[["San_Luis", "Oneill"]]
    san_luis_pump = -1*san_luis_pump.clip(upper=0) #Only save pumping values
    san_luis_pump = san_luis_pump.resample('MS').sum()
    
    #Compute the pumping at Tracy
    tracy_ei = 232.7 #kWh/AF
    tracy_pump = all_data[i]['delta_TRP_pump']*tracy_ei/1000 #GWh
    tracy_pump = tracy_pump.resample('MS').sum()
    
    #Compute pumping at Dos Amigoes
    dos_amigos_ei = 135.6 #Kwh/AF
    dos_amigos_pump = all_data[i]['caa_SNL_flow']*dos_amigos_ei/1000 #Gwh
    dos_amigos_pump =  dos_amigos_pump.resample('MS').sum()
    
    #Compute the CVP Gen
    cvp_gen = cvp_gen.drop(["San_Luis", "Oneill", "CVP_Gen"], axis=1)  
    cvp_gen['CVP_Gen'] = cvp_gen.sum(axis=1)
    cvp_gen = cvp_gen.resample('MS').sum()
    
    #Classify the values
    cvp_gen['San_Luis_Pump'] = san_luis_pump['San_Luis'] + san_luis_pump['Oneill']
    cvp_gen['Tracy_Pump'] = tracy_pump
    cvp_gen['Dos_Amigos'] = dos_amigos_pump
    cvp_gen['CVPT_Losses'] = 0.018*cvp_gen['CVP_Gen'] #1.8% Losses assumed
    cvp_gen['First_Prefence'] = fp_use
    cvp_gen['COTP_Losses'] = cotp_losses
    cvp_gen['Project_Use'] = (cvp_gen['San_Luis_Pump'] + cvp_gen['Tracy_Pump'] + cvp_gen['Dos_Amigos'] + cvp_gen['CVPT_Losses'] + cvp_gen['First_Prefence'] + cvp_gen['COTP_Losses'])
    cvp_gen['BR'] = cvp_gen['CVP_Gen'] - cvp_gen['Project_Use']
    cvp_gen['BR'] = cvp_gen['BR'].clip(lower=0)
    
    sim_hydro.append(cvp_gen)
    
    

# %%    CVP Hydropower Generation


#Compute the exceedances
scenario_matrix = np.array([data['CVP_Gen'] for data in sim_hydro])
p10 = np.percentile(scenario_matrix, 10, axis=0)
p90 = np.percentile(scenario_matrix, 90, axis=0)
p25 = np.percentile(scenario_matrix, 25, axis=0)
p75 = np.percentile(scenario_matrix, 75, axis=0)
p50 = np.percentile(scenario_matrix, 50, axis=0)

#COV CVP Gen
((p90 - p10)/p50).mean()

#-----------------------------------------------------------------------------#
#Empty Plot
plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=0, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=0, alpha=0.5)  # Use blue with transparency
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.0, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=0, label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()

#-----------------------------------------------------------------------------#
#One Year
plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=0, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(1):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  # Use blue with transparency
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.0, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=0, label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


#-----------------------------------------------------------------------------#
#Two Year
plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=0, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(2):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  # Use blue with transparency
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.0, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=0, label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


#-----------------------------------------------------------------------------#
#All Year
plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=0, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  # Use blue with transparency
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.0, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=0, label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


#-----------------------------------------------------------------------------#
#All Years + Exceedance
plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=0, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  # Use blue with transparency
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=0, label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


#-----------------------------------------------------------------------------#
# First add the drought period highlighting (2020-2022)
# Create start and end dates for each drought year
drought_periods = [
    (pd.Timestamp('2019-10-01'), pd.Timestamp('2020-09-30'), '#FFEEAA'),  # 2020 Water Year (light yellow)
    (pd.Timestamp('2020-10-01'), pd.Timestamp('2021-09-30'), '#FFDD88'),  # 2021 Water Year (medium yellow)
    (pd.Timestamp('2021-10-01'), pd.Timestamp('2022-09-30'), '#E6B800')   # 2022 Water Year (light yellow)
]

#All Years + Exceedance + Median Generation Value
plt.figure(figsize = (15, 8))
for start_date, end_date, color in drought_periods:
    plt.axvspan(start_date, end_date, color=color, alpha=0.5, zorder=0)
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=0, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  # Use blue with transparency
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=2, alpha=1, linestyle = 'dashdot', label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()



#-----------------------------------------------------------------------------#
#All Years + Exceedance + Median Generation Value + EIA 
plt.figure(figsize = (15, 8))
for start_date, end_date, color in drought_periods:
    plt.axvspan(start_date, end_date, color=color, alpha=0.5, zorder=0)
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=2, label = "EIA Data")
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=2, alpha=1, linestyle = 'dashdot', label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


#-----------------------------------------------------------------------------#
#All Years + Exceedance + Median Generation Value 
plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=0)
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['CVP_Gen'].index, sim_hydro[i]['CVP_Gen'], 
             'r-', linewidth=1, alpha=0.5)  
plt.fill_between(sim_hydro[0]['CVP_Gen'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['CVP_Gen'].index, p50, linewidth=2, alpha=1, linestyle = 'dashdot', label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Total Hydropower Generation (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()
    
    
# %%    Project Use


#Compute the exceedances
scenario_matrix = np.array([data['Project_Use'] for data in sim_hydro])
p10 = np.percentile(scenario_matrix, 10, axis=0)
p90 = np.percentile(scenario_matrix, 90, axis=0)
p25 = np.percentile(scenario_matrix, 25, axis=0)
p75 = np.percentile(scenario_matrix, 75, axis=0)
p50 = np.percentile(scenario_matrix, 50, axis=0)

#COV Project Use
((p90 - p10)/p50).mean()

plt.figure(figsize = (15, 8))
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['Project_Use'].index, sim_hydro[i]['Project_Use'], 
             'r-', linewidth=1, alpha=0.5)  
plt.fill_between(sim_hydro[0]['Project_Use'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['Project_Use'].index, p50, linewidth=2, alpha=1, linestyle = 'dashdot', label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("CVP Project Use (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


# %%    Base Resource


#Compute the exceedances
scenario_matrix = np.array([data['BR'] for data in sim_hydro])
p10 = np.percentile(scenario_matrix, 10, axis=0)
p90 = np.percentile(scenario_matrix, 90, axis=0)
p25 = np.percentile(scenario_matrix, 25, axis=0)
p75 = np.percentile(scenario_matrix, 75, axis=0)
p50 = np.percentile(scenario_matrix, 50, axis=0)


#COV Base Resource
mask = p50 != 0
((p90[mask] - p10[mask])/p50[mask]).mean()


plt.figure(figsize = (15, 8))
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['BR'].index, sim_hydro[i]['BR'], 
             'r-', linewidth=1, alpha=0.5)  
plt.fill_between(sim_hydro[0]['BR'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['BR'].index, p50, linewidth=2, alpha=1, linestyle = 'dashdot', label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("Base Resource (GWh) \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()




# %%    Base Resource without New Melones Generation


#Compute the exceedances
scenario_matrix = np.array([(data['BR'] - data['New_Melones']).clip(lower=0) for data in sim_hydro])
p10 = np.percentile(scenario_matrix, 10, axis=0)
p90 = np.percentile(scenario_matrix, 90, axis=0)
p25 = np.percentile(scenario_matrix, 25, axis=0)
p75 = np.percentile(scenario_matrix, 75, axis=0)
p50 = np.percentile(scenario_matrix, 50, axis=0)


plt.figure(figsize = (15, 8))
for year in range(2018, 2023 + 1):
    oct1_date = pd.Timestamp(f'{year}-10-01')
    plt.axvline(x=oct1_date, color='black', linestyle=':', linewidth = 2)
for i in range(len(all_data)):
    plt.plot(sim_hydro[i]['BR'].index, (sim_hydro[i]['BR']- sim_hydro[i]['New_Melones']).clip(lower=0), 
             'r-', linewidth=1, alpha=0.5)  
plt.fill_between(sim_hydro[0]['BR'].index, p10, p90, 
                 color='r', alpha=0.5, label='10-90th Percentile Exceedance')
plt.plot(sim_hydro[0]['BR'].index, p50, linewidth=2, alpha=1, linestyle = 'dashdot', label = "Median Gen") 
plt.ylabel("Hydropower (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
plt.title("Base Resource (GWh) \n With New Melones Hydroplant Failure", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()



#%% Simplified plots

#Compute the exceedances -- Base Resource
scenario_matrix = np.array([data['CVP_Gen'] for data in sim_hydro])
p50_total_gen = np.percentile(scenario_matrix, 50, axis=0)[0:12]
p40_total_gen = np.percentile(scenario_matrix, 40, axis=0)[0:12]


#Compute the exceedances -- Base Resource
scenario_matrix = np.array([data['Project_Use'] for data in sim_hydro])
p50_project_use = np.percentile(scenario_matrix, 50, axis=0)[0:12]
p40_project_use = np.percentile(scenario_matrix, 40, axis=0)[0:12]


#Create plot of Total CVP Gen
months = ['Oct 23', 'Nov 23', 'Dec 23', 'Jan 24', 'Feb 24', 'Mar 24', 
          'Apr 24', 'May 24', 'Jun 24', 'Jul 24', 'Aug 24', 'Sep 24']


# Plot 1 -- CVP Gen
fig, ax = plt.subplots(figsize=(15, 8))
bars = ax.bar(months, p50_total_gen, color='blue', edgecolor='black', label = "Total CVP Gen")
ax.set_ylabel('Forecasted Monthly Generation \n (GWh)', fontsize=24)
ax.set_xlabel('Month', fontsize=24)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=18)
ax.set_ylim(0, 800)
ax.set_yticks(np.arange(0, 801, 100))
plt.legend(bbox_to_anchor=(0.2, 0.99), loc='upper center', \
           fontsize=24, frameon = True)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2 -- CVP Gen + Project Use
fig, ax = plt.subplots(figsize=(15, 8))
p50_base_resource = p50_total_gen - p50_project_use
bottom_bars = ax.bar(months, p50_project_use, color='yellow', edgecolor='black', 
                     label="Project Use & Losses")
top_bars = ax.bar(months, p50_base_resource, color='red', edgecolor='black',
                  bottom=p50_project_use, label="Base Resource")
ax.set_ylabel('Forecasted Monthly Generation \n (GWh)', fontsize=24)
ax.set_xlabel('Month', fontsize=24)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=18)
ax.set_ylim(0, 800)
ax.set_yticks(np.arange(0, 801, 100))
plt.legend(bbox_to_anchor=(0.2, 0.99), loc='upper center',
           fontsize=24, frameon=True)
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot 3 -- Assuming a 10% forecast error
err = 1
fig, ax = plt.subplots(figsize=(15, 8))
p40_base_resource = (p40_total_gen - p40_project_use).clip(min=0)
bottom_bars = ax.bar(months, p40_project_use, color='yellow', edgecolor='black', 
                     label="Project Use & Losses (Updated)")
top_bars = ax.bar(months, p40_base_resource, color='red', edgecolor='black',
                  bottom=p40_project_use, label="Base Resource (Updated)")
ax.set_ylabel('Forecasted Monthly Generation \n (GWh)', fontsize=24)
ax.set_xlabel('Month', fontsize=24)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=18)
ax.set_ylim(0, 800)
ax.set_yticks(np.arange(0, 801, 100))
plt.legend(bbox_to_anchor=(0.25, 0.99), loc='upper center',
           fontsize=24, frameon=True)
plt.grid(True)
plt.tight_layout()
plt.show()


#Reductions in base resource
(p50_base_resource - p40_base_resource)/p50_base_resource
(p50_base_resource.sum() - p40_base_resource.sum())/p40_base_resource.sum()