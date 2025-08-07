# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 06:31:53 2025

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

#Read the WAPA generation dataset
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)
eia = eia/1000
eia.index = pd.to_datetime(eia.index)
eia = eia.drop(['W R Gianelli', 'ONeill'], axis=1)
eia['CVP_Gen'] = eia.sum(axis=1)
eia = eia[eia.index < pd.Timestamp("2023-10-01")]
eia = eia[eia.index > pd.Timestamp("2003-09-01")]


# Initialize an empty list to store DataFrames
all_data = []

# Loop through years from 1996 to 2023
for year in range(1996, 2024):
    
    #Print the year
    print(year)
    
    # Construct the path for each year
    output_folder = f"WAPA_Runs/results/annual_runs/{year}/"
    output_file = os.path.join(output_folder, 'results.hdf5')
    
    # Load the data for current year
    yearly_data = get_results_sensitivity_number_outside_model(output_file, '')
    yearly_data = yearly_data[['shasta_S', 'shasta_R', 'oroville_S', 'oroville_R', 'trinity_S', 'trinity_R', 'folsom_S', 'folsom_R', 'newmelones_S', 'newmelones_R','sanluisstate_S', 'sanluisfederal_S', 'trinity_diversions']]
    
    # Append to our list
    all_data.append(yearly_data)
    
# Concatenate all DataFrames into one
datDaily = pd.concat(all_data,)
datDaily.head()


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
release_input['San Luis'] =  0.5*storage_input['San Luis'].diff().mul(-1).fillna(0)
release_input.loc[(release_input.index.month == 10) & (release_input.index.day == 1), 'San Luis'] = 0

#Compute the hydropower generation 
cvp_gen = get_CVP_hydro_gen(storage_input, release_input)

#Compute the updated cvp_gen (Without San Luis & O'Neill)
cvp_gen = cvp_gen.drop(['San_Luis', 'Oneill'], axis=1)
cvp_gen = cvp_gen.drop(['CVP_Gen'], axis=1)
cvp_gen['CVP_Gen'] = cvp_gen.sum(axis=1)
cvp_gen = cvp_gen.resample('MS').sum()
cvp_gen = cvp_gen[cvp_gen.index < pd.Timestamp("2023-10-01")]
cvp_gen = cvp_gen[cvp_gen.index > pd.Timestamp("2003-09-01")]

#%% Total CVP Gen

#eia = eia[eia.index > pd.Timestamp("2013-09-30")]
#cvp_gen = cvp_gen[cvp_gen.index > pd.Timestamp("2013-09-30")]

# First compute the correlation
correlation = np.corrcoef(
    cvp_gen['CVP_Gen'].dropna(),
    eia['CVP_Gen'].dropna())[0,1]

# Create a figure with a special layout
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, 4)  # 4 columns total

# Main time series plot
ax1 = fig.add_subplot(gs[0, 0:3])  # First 3 columns
ax1.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=2, label="EIA")
ax1.plot(cvp_gen.index, cvp_gen['CVP_Gen'], 'r-', 
         linewidth=2, label="CALFEWS")
ax1.set_ylabel("CVP Hydropower (GWh)", fontsize=24)
ax1.set_xlabel("Year", fontsize=24)
ax1.tick_params(axis='both', labelsize=20)
ax1.set_ylim(0, 950)
ax1.grid(True)

# Place legend in the upper right corner of the left plot
ax1.legend(loc='upper right', fontsize=24, frameon=True)

# Scatter plot
ax2 = fig.add_subplot(gs[0, 3])  # Last column
ax2.scatter(eia['CVP_Gen'].dropna(), cvp_gen['CVP_Gen'].dropna(),  
           color='green', alpha=0.7, s=30)

# Add 1:1 line
max_val = max(cvp_gen['CVP_Gen'].max(), eia['CVP_Gen'].max(), 950)
min_val = min(cvp_gen['CVP_Gen'].min(), eia['CVP_Gen'].min(),0)
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
ax2.set_ylim(min_val, max_val)  # Same as ax1
ax2.set_xlim(min_val, max_val) 
# Add correlation text to scatter plot
ax2.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')

# Set scatter plot labels and appearance
ax2.set_xlabel("EIA (GWh)", fontsize=24)
ax2.set_ylabel("\nCALFEWS  (GWh)", fontsize=24)  # Fixed typo: Gwh → GWh
ax2.tick_params(axis='both', labelsize=20)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Calculate MSE between actual and predicted values
mse = ((cvp_gen['CVP_Gen'] - eia['CVP_Gen']) ** 2).mean()
print(f"Mean Squared Error: {mse:.2f}")

mae = (abs(cvp_gen['CVP_Gen'] - eia['CVP_Gen'])).mean()
print(f"Mean Absolute Error: {mae:.2f}")



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


#%%
#Annual Scatter plot
from scipy import stats
from scipy.stats import linregress

def get_water_year(date):
    """
    Convert calendar date to water year.
    Water year runs from Oct 1 to Sep 30.
    """
    return date.to_series().apply(lambda x: x.year + 1 if x.month >= 10 else x.year)

eia['WY'] = get_water_year(eia.index)
cvp_gen['WY'] = get_water_year(cvp_gen.index)


annual_eia = eia.groupby('WY')['CVP_Gen'].sum()
eia['Predicted'] = cvp_gen['CVP_Gen']
annual_predicted = eia.groupby('WY')['Predicted'].sum()

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(annual_eia.values, annual_predicted.values, 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
for i, year in enumerate(annual_eia.index):
    ax.annotate(str(year), 
               (annual_eia.iloc[i], annual_predicted.iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, alpha=0.8)
slope, intercept, r_value, p_value, std_err = stats.linregress(annual_predicted.values, annual_eia.values)
line_x = np.linspace(annual_eia.min(), annual_eia.max(), 100)
line_y = slope * line_x + intercept
min_val = min(annual_eia.min(), annual_predicted.min())
max_val = max(annual_eia.max(), annual_predicted.max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((annual_predicted - annual_eia)**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('EIA Generation (GWh)', fontsize=24)
ax.set_ylabel('CALFEWS Generation (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(min_val - 200, max_val + 500)
ax.set_ylim(min_val - 200, max_val + 500)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()