# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:44:56 2025

@author: amonkar

Figure 4 - Code for the CALFEWS Storage Validation Plots. 
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
from scipy import stats
from matplotlib import gridspec
from scipy import stats
import matplotlib.dates as mdates

#Extract custom functions
from Annual_Runs.post_processing.functions.get_CALFEWS_results import get_results_sensitivity_number_outside_model
from Annual_Runs.post_processing.functions.get_comparison_plots import get_comparison_plots
from Annual_Runs.post_processing.functions.get_CVP_hydro_gen import get_CVP_hydro_gen


#Hyper-parameters
cfs_tafd = 2.29568411*10**-5 * 86400 / 1000


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)


# %% Extract data from the CALFEWS results

# Initialize an empty list to store DataFrames
all_data = []

# Loop through years from 1996 to 2023
for year in range(1996, 2024):
    
    #Print the year
    print(year)
    
    # Construct the path for each year
    output_folder = f"Annual_Runs/results/annual_runs/{year}/"
    output_file = os.path.join(output_folder, 'results.hdf5')
    
    # Load the data for current year
    yearly_data = get_results_sensitivity_number_outside_model(output_file, '')
    yearly_data = yearly_data[['shasta_S', 'shasta_R', 'oroville_S', 'oroville_R', 'trinity_S', 'trinity_R', 'folsom_S', 'folsom_R', 'newmelones_S', 'newmelones_R','sanluisstate_S', 'sanluisfederal_S', 'trinity_diversions']]
    
    # Append to our list
    all_data.append(yearly_data)
    
# Concatenate all DataFrames into one
datDaily = pd.concat(all_data,)
datDaily.head()


#Aggregate to Monthly
input_data_mean = input_data.resample('ME').mean()
datDaily_mean = datDaily.resample('ME').mean()

input_data_sum = input_data.resample('ME').sum()
datDaily_sum = datDaily.resample('ME').sum()


#Storage
np.corrcoef(input_data_mean['SHA_storage']/1000,datDaily_mean['shasta_S'])
np.sqrt(((input_data_mean['SHA_storage']/1000 - datDaily_mean['shasta_S'])**2).mean())
np.corrcoef(input_data_mean['TRT_storage']/1000,datDaily_mean['trinity_S'])
np.sqrt(((input_data_mean['TRT_storage']/1000 - datDaily_mean['trinity_S'])**2).mean())


#Storage
np.corrcoef(input_data_sum['SHA_otf']*cfs_tafd,datDaily_sum['shasta_R'])
np.sqrt(((input_data_sum['SHA_otf']*cfs_tafd-datDaily_sum['shasta_R'])**2).mean())
np.corrcoef(input_data_sum['TRT_otf']*cfs_tafd,datDaily_sum['trinity_R'])
np.sqrt(((input_data_sum['TRT_otf']*cfs_tafd-datDaily_sum['trinity_R'])**2).mean())


#%%

# Create a figure with 2x2 subplots
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 2, figure=fig)

# Define the four subplots
ax1 = fig.add_subplot(gs[0, 0])  # Shasta Storage (top-left)
ax2 = fig.add_subplot(gs[0, 1])  # Trinity Storage (top-right)
ax3 = fig.add_subplot(gs[1, 0])  # Shasta Releases (bottom-left)
ax4 = fig.add_subplot(gs[1, 1])  # Trinity Releases (bottom-right)

# Increase font size for all text elements
plt.rcParams.update({'font.size': 14})

# Calculate correlations (you'll need to adjust the data alignment as needed)
# Assuming the data needs to be aligned by index for correlation calculation
corr_shasta_storage = input_data_mean['SHA_storage'].corr(datDaily_mean['shasta_S'])
corr_trinity_storage = input_data_mean['TRT_storage'].corr(datDaily_mean['trinity_S'])
corr_shasta_releases = input_data_sum['SHA_otf'].corr(datDaily_sum['shasta_R'])
corr_trinity_releases = input_data_sum['TRT_otf'].corr(datDaily_sum['trinity_R'])

# Shasta Storage (top-left)
ax1.plot(input_data_mean.index, input_data_mean['SHA_storage']/1000, 'b-', linewidth=2, label="CDEC")
ax1.plot(datDaily_mean.index, datDaily_mean['shasta_S'], 'r-', linewidth=2, label="CALFEWS")
ax1.set_ylabel("Storage (TAF)", fontsize=26)
ax1.set_xlabel("Year", fontsize=26)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_ylim([500, 5200]) 
ax1.grid(True)
ax1.text(0.05, 0.92, '(A)', transform=ax1.transAxes, fontsize=28, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
# Add correlation value in top-right
ax1.text(0.33, 0.92, f'r = {corr_shasta_storage:.2f}', transform=ax1.transAxes, 
         fontsize=20, ha='right',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

# Trinity Storage (top-right)
ax2.plot(input_data_mean.index, input_data_mean['TRT_storage']/1000, 'b-', linewidth=2, label="CDEC")
ax2.plot(datDaily_mean.index, datDaily_mean['trinity_S'], 'r-', linewidth=2, label="CALFEWS")
ax2.set_ylabel("Storage (TAF)", fontsize=26)
ax2.set_xlabel("Year", fontsize=26)
ax2.tick_params(axis='both', labelsize=18)
ax2.set_ylim([400, 2900]) 
ax2.grid(True)
ax2.text(0.05, 0.92, '(B)', transform=ax2.transAxes, fontsize=28, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
# Add correlation value in top-right
ax2.text(0.33, 0.92, f'r = {corr_trinity_storage:.2f}', transform=ax2.transAxes, 
         fontsize=20, ha='right',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

# Shasta Releases (bottom-left)
ax3.plot(input_data_sum.index, input_data_sum['SHA_otf']*cfs_tafd, 'b-', linewidth=2, label="CDEC")
ax3.plot(datDaily_sum.index, datDaily_sum['shasta_R'], 'r-', linewidth=2, label="CALFEWS")
ax3.set_ylabel("Releases (TAF)", fontsize=26)
ax3.set_xlabel("Year", fontsize=26)
ax3.tick_params(axis='both', labelsize=18)
ax3.set_title(" ", fontsize=28)
ax3.grid(True)
#ax3.set_ylim([0, 200]) 
ax3.text(0.05, 0.92, '(C)', transform=ax3.transAxes, fontsize=28, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
# Add correlation value in top-right
ax3.text(0.33, 0.92, f'r = {corr_shasta_releases:.2f}', transform=ax3.transAxes, 
         fontsize=20, ha='right',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

# Trinity Releases (bottom-right)
ax4.plot(input_data_sum.index, input_data_sum['TRT_otf']*cfs_tafd, 'b-', linewidth=2, label="CDEC")
ax4.plot(datDaily_sum.index, datDaily_sum['trinity_R'], 'r-', linewidth=2, label="CALFEWS")
ax4.set_ylabel("Releases (TAF)", fontsize=26)
ax4.set_xlabel("Year", fontsize=26)
ax4.tick_params(axis='both', labelsize=18)
ax4.set_title(" ", fontsize=28)
ax4.grid(True)
#ax4.set_ylim([0, 45]) 
ax4.text(0.05, 0.92, '(D)', transform=ax4.transAxes, fontsize=28, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
# Add correlation value in top-right
ax4.text(0.33, 0.92, f'r = {corr_trinity_releases:.2f}', transform=ax4.transAxes, 
         fontsize=20, ha='right',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
# Add legend to bottom-left subplot (ax3)
handles, labels = ax3.get_legend_handles_labels()
ax4.legend(handles, labels, loc='upper right', fontsize=26, framealpha=0.9)

# Adjust spacing between subplots
plt.tight_layout()

# Show plot
#plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
# Create a figure with 2x2 subplots
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 2, figure=fig)

# Define the four subplots
ax1 = fig.add_subplot(gs[0, 0])  # Shasta Storage (top-left)
ax2 = fig.add_subplot(gs[0, 1])  # Trinity Storage (top-right)

# Increase font size for all text elements
plt.rcParams.update({'font.size': 14})

# Calculate correlations (you'll need to adjust the data alignment as needed)
# Assuming the data needs to be aligned by index for correlation calculation
corr_shasta_storage = input_data_mean['SHA_storage'].corr(datDaily_mean['shasta_S'])**2
corr_trinity_storage = input_data_mean['TRT_storage'].corr(datDaily_mean['trinity_S'])**2

# Shasta Storage (top-left)
ax1.plot(input_data_mean.index, input_data_mean['SHA_storage']/1000, 'b-', linewidth=2, label="CDEC")
ax1.plot(datDaily_mean.index, datDaily_mean['shasta_S'], 'r-', linewidth=2, label="CALFEWS")
ax1.set_ylabel("Storage (TAF)", fontsize=26)
ax1.set_xlabel("Year", fontsize=26)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_ylim([500, 5200]) 
ax1.grid(True)
ax1.text(0.05, 0.92, '(A)', transform=ax1.transAxes, fontsize=28, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
# Add correlation value in top-right
ax1.text(0.33, 0.92, f'r² = {corr_shasta_storage:.2f}', transform=ax1.transAxes, 
         fontsize=20, ha='right',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

# Trinity Storage (top-right)
ax2.plot(input_data_mean.index, input_data_mean['TRT_storage']/1000, 'b-', linewidth=2, label="CDEC")
ax2.plot(datDaily_mean.index, datDaily_mean['trinity_S'], 'r-', linewidth=2, label="CALFEWS")
ax2.set_ylabel("Storage (TAF)", fontsize=26)
ax2.set_xlabel("Year", fontsize=26)
ax2.tick_params(axis='both', labelsize=18)
ax2.set_ylim([0, 2900]) 
ax2.grid(True)
ax2.text(0.05, 0.92, '(B)', transform=ax2.transAxes, fontsize=28, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
# Add correlation value in top-right
ax2.text(0.33, 0.92, f'r² = {corr_trinity_storage:.2f}', transform=ax2.transAxes, 
         fontsize=20, ha='right',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

# Add legend to bottom-left subplot (ax3)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, loc='lower left', fontsize=26, framealpha=0.9)

# Adjust spacing between subplots
plt.tight_layout()

# Show plot
#plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')
plt.show()


#%% Supplemental plot -- All reservoirs

#Input Data
storage_columns = [col for col in input_data.columns if col.endswith('_storage')]
input_storage = input_data[storage_columns]
input_storage = input_storage[['ORO_storage', 'YRS_storage', 'FOL_storage','NML_storage', \
                               'DNP_storage', 'EXC_storage', 'MIL_storage', 'ISB_storage', 'SL_storage']]/1000


# Initialize an empty list to store DataFrames
all_data = []

# Loop through years from 1996 to 2023
for year in range(1996, 2024):
    
    #Print the year
    print(year)
    
    # Construct the path for each year
    output_folder = f"Annual_Runs/results/annual_runs/{year}/"
    output_file = os.path.join(output_folder, 'results.hdf5')
    
    # Load the data for current year
    yearly_data = get_results_sensitivity_number_outside_model(output_file, '')
    storage_columns = [col for col in yearly_data.columns if col.endswith('_S')]
    yearly_data = yearly_data[storage_columns]
        
    # Append to our list
    all_data.append(yearly_data)
    
# Concatenate all DataFrames into one
datDaily = pd.concat(all_data,)
datDaily['sanluis_S'] = datDaily['sanluisstate_S'] + datDaily['sanluisfederal_S']
datDaily = datDaily[['oroville_S', 'yuba_S', 'folsom_S', 'newmelones_S',
       'donpedro_S', 'exchequer_S','millerton_S', 'isabella_S', 'sanluis_S']]

input_storage.columns = datDaily.columns = [['Oroville', 'Yuba', 'Folsom', 'New Melones', 'Don Pedro', \
                                             'Exchequer', 'Millerton', 'Isabella', 'San Luis']]

    
# Aggregate to monthly means
datDaily_monthly = datDaily.resample('M').mean()
input_storage_monthly = input_storage.resample('M').mean()

# Get reservoir names
reservoirs = list(datDaily.columns)

# Create 3x3 plot
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, reservoir in enumerate(reservoirs):
    ax = axes[i]
    
    # Get data
    calfews = datDaily_monthly[reservoir]
    cdec = input_storage_monthly[reservoir]
    
    # Calculate R-squared
    correlation = np.corrcoef(calfews, cdec)[0, 1]
    r_squared = correlation ** 2
    
    # Plot
    ax.plot(calfews.index, calfews, 'r-', label='CALFEWS', linewidth=1)
    ax.plot(cdec.index, cdec, 'b-', label='CDEC', linewidth=1)
    
    # Clean title and R-squared in top right
    ax.set_title(str(reservoir).strip("(),'"))
    ax.text(0.98, 0.98, f'$r^2$ = {r_squared:.2f}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_ylabel('Storage (TAF)')
    ax.grid(True, alpha=0.3)
    
    # Fix x-axis ticks
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

# Add common legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

plt.show()