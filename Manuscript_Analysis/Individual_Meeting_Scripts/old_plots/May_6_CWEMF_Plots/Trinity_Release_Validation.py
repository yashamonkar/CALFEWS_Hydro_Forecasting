# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:10:36 2025

@author: amonkar

Trinity River Diversions Validation
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
from matplotlib.gridspec import GridSpec


#Extract custom functions
from WAPA_Runs.post_processing.functions.get_CALFEWS_results import get_results_sensitivity_number_outside_model
from WAPA_Runs.post_processing.functions.get_comparison_plots import get_comparison_plots
from WAPA_Runs.post_processing.functions.get_CVP_hydro_gen import get_CVP_hydro_gen


#Hyper-parameters
cfs_tafd = 2.29568411*10**-5 * 86400 / 1000


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)
input_data = input_data.resample('MS').sum()

# CALFEWS OUTPUT VALIDATION RUN
output_folder = "results/Historical_validation_1996-2023/"
output_file = output_folder + 'results.hdf5'
datDaily = get_results_sensitivity_number_outside_model(output_file, '')
datDaily = datDaily.resample('MS').sum()


# %% Simpler plot

# First compute the correlation
correlation = np.corrcoef(
    cfs_tafd*input_data['TRT_otf'],
    datDaily['trinity_R'])[0,1]

# Create a figure with a special layout
plt.figure(figsize=(20, 8))
plt.plot(input_data['TRT_otf'].index, input_data['TRT_otf']*cfs_tafd, 'b-', 
         linewidth=2, label = "CDEC Flows")
plt.plot(datDaily['trinity_R'].index, datDaily['trinity_R'], 'r-', 
         linewidth=0, label = " ")
plt.ylabel("Releases (TAF)", fontsize=24)
plt.xlabel("Month", fontsize=24)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show()



# Create a figure with a special layout
plt.figure(figsize=(20, 8))
plt.plot(input_data['TRT_otf'].index, input_data['TRT_otf']*cfs_tafd, 'b-', 
         linewidth=2, label = "CDEC Flows")
plt.plot(datDaily['trinity_R'].index, datDaily['trinity_R'], 'r-', 
         linewidth=2, label = "CALFEWS Flows")
plt.ylabel("Releases (TAF)", fontsize=24)
plt.xlabel("Month", fontsize=24)
plt.tick_params(axis='both', labelsize=18)
plt.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.06, 0.95), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')
plt.grid(True)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show()


#%% Just CDEC Releases

# First compute the correlation
correlation = np.corrcoef(
    input_data['TRT_otf']*cfs_tafd,
    datDaily['trinity_R'])[0,1]

# Create a figure with a special layout
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, 4)  # 4 columns total

# Main time series plot
ax1 = fig.add_subplot(gs[0, 0:3])  # First 3 columns
ax1.plot(input_data['TRT_otf'].index, input_data['TRT_otf']*cfs_tafd, 'b-', 
         linewidth=2, label = "CDEC Flows")
ax1.plot(datDaily['trinity_R'].index, datDaily['trinity_R'], 'r-', 
         linewidth=0)
ax1.set_ylabel("Releases (TAF)", fontsize=24)
ax1.set_xlabel("Month", fontsize=24)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_title(f"Trinity Reservoir Releases", 
              fontsize=28)
ax1.grid(True)

# Place legend in the upper right corner of the left plot
ax1.legend(loc='upper right', fontsize=18, frameon=True)

# Scatter plot
ax2 = fig.add_subplot(gs[0, 3])  # Last column
ax2.scatter(cfs_tafd*input_data['TRT_otf'].dropna(), datDaily['trinity_R'].dropna(),  
           color='green', alpha=0.7, s=30)

# Add 1:1 line
max_val = max(datDaily['trinity_R'].max(), cfs_tafd*input_data['TRT_otf'].max())
min_val = min(datDaily['trinity_R'].min(), cfs_tafd*input_data['TRT_otf'].min())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

# Add correlation text to scatter plot
ax2.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')

# Set scatter plot labels and appearance
ax2.set_xlabel("CDEC Releases (TAF)", fontsize=24)
ax2.set_ylabel("CALFEWS Gen (GWh)", fontsize=24)  # Fixed typo: Gwh → GWh
ax2.tick_params(axis='both', labelsize=18)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()





#%% Both Releases

# First compute the correlation
correlation = np.corrcoef(
    input_data['TRT_otf']*cfs_tafd,
    datDaily['trinity_R'])[0,1]

# Create a figure with a special layout
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, 4)  # 4 columns total

# Main time series plot
ax1 = fig.add_subplot(gs[0, 0:3])  # First 3 columns
ax1.plot(input_data['TRT_otf'].index, input_data['TRT_otf']*cfs_tafd, 'b-', 
         linewidth=2, label = "CDEC Flows")
ax1.plot(datDaily['trinity_R'].index, datDaily['trinity_R'], 'r-', 
         linewidth=2, label = "CALFEWS Flows")
ax1.set_ylabel("Releases (TAF)", fontsize=24)
ax1.set_xlabel("Month", fontsize=24)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_title(f"Trinity Reservoir Releases", 
              fontsize=28)
ax1.grid(True)

# Place legend in the upper right corner of the left plot
ax1.legend(loc='upper right', fontsize=18, frameon=True)

# Scatter plot
ax2 = fig.add_subplot(gs[0, 3])  # Last column
ax2.scatter(cfs_tafd*input_data['TRT_otf'].dropna(), datDaily['trinity_R'].dropna(),  
           color='green', alpha=0.7, s=30)

# Add 1:1 line
max_val = max(datDaily['trinity_R'].max(), cfs_tafd*input_data['TRT_otf'].max())
min_val = min(datDaily['trinity_R'].min(), cfs_tafd*input_data['TRT_otf'].min())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

# Add correlation text to scatter plot
ax2.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')

# Set scatter plot labels and appearance
ax2.set_xlabel("CDEC Releases (TAF)", fontsize=24)
ax2.set_ylabel("CALFEWS Gen (GWh)", fontsize=24)  # Fixed typo: Gwh → GWh
ax2.tick_params(axis='both', labelsize=18)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

