# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:55:35 2025

@author: amonkar
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
eia = eia[eia.index < pd.Timestamp("2023-10-01")]


# CALFEWS OUTPUT VALIDATION RUN
output_folder = "results/Historical_validation_1996-2023/"
output_file = output_folder + 'results.hdf5'
datDaily = get_results_sensitivity_number_outside_model(output_file, '')


#%% Compute total hydropower and individual hydropower for CALFEWS
#Get the storage data set
storage_input = datDaily[['shasta_S', 'trinity_S', 'folsom_S', \
                             'newmelones_S']]
storage_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
storage_input['San Luis'] = datDaily['sanluisstate_S'] + datDaily['sanluisfederal_S']

#Get the releases dataset
release_input = datDaily[['shasta_R', 'trinity_R', 'folsom_R', \
                             'newmelones_R']]
release_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
release_input.loc[:,'Diversions'] = datDaily['trinity_diversions']
release_input['San Luis'] =  storage_input['San Luis'].diff().mul(-1).fillna(0)
release_input.loc[(release_input.index.month == 10) & (release_input.index.day == 1), 'San Luis'] = 0

#Compute the hydropower generation 
cvp_gen = get_CVP_hydro_gen(storage_input, release_input)
cvp_gen = cvp_gen.resample('MS').sum()
cvp_gen = cvp_gen[cvp_gen.index > pd.Timestamp("2002-12-31")]


#%% Shasta Hydropower Plot

# First compute the correlation
correlation = np.corrcoef(
    cvp_gen['Shasta'].dropna(),
    eia['Shasta'].dropna())[0,1]

plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['Shasta'], 'b-', 
         linewidth=2, label = "EIA Gen")
plt.plot(cvp_gen.index, cvp_gen['Shasta'], 'r-', 
         linewidth=2, label = "CALFEWS Gen")
plt.ylabel("Hydropower Generation (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title(f"Monthly Hydropower Generation at Shasta (2003-2023)\nPearson Correlation: {correlation:.2f}",  fontsize=28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=24, ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()



#%% Folsom Hydropower Plot

# First compute the correlation
correlation = np.corrcoef(
    cvp_gen['Folsom'].dropna(),
    eia['Folsom'].dropna())[0,1]

plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['Folsom'], 'b-', 
         linewidth=2, label = "EIA Gen")
plt.plot(cvp_gen.index, cvp_gen['Folsom'], 'r-', 
         linewidth=2, label = "CALFEWS Gen")
plt.ylabel("Hydropower Generation (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title(f"Monthly Hydropower Generation at Folsom (2003-2023)\nPearson Correlation: {correlation:.2f}",  fontsize=28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=24, ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()



#%% Trinity Hydropower Plot

# First compute the correlation
correlation = np.corrcoef(
    cvp_gen['Trinity'].dropna(),
    eia['Trinity'].dropna())[0,1]

plt.figure(figsize = (15, 8))
plt.plot(eia.index, eia['Trinity'], 'b-', 
         linewidth=2, label = "EIA Gen")
plt.plot(cvp_gen.index, cvp_gen['Trinity'], 'r-', 
         linewidth=2, label = "CALFEWS Gen")
plt.ylabel("Hydropower Generation (GWh)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title(f"Monthly Hydropower Generation at Trinity (2003-2023)\nPearson Correlation: {correlation:.2f}",  fontsize=28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=24, ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()