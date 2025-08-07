# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:45:22 2025

@author: amonkar

Hydropower Validation Plot -- Just Shasta
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)

# import libraries
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec

#Extract the Hydropower Generation Plots
from WAPA_Runs.post_processing.functions.get_CVP_hydro_gen import get_CVP_hydro_gen
from WAPA_Runs.post_processing.functions.get_comparison_plots import get_comparison_plots
cfs_to_tafd = 1.98211/1000


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)

#WAPA generation dataset
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)
eia = eia/1000
eia.index = pd.to_datetime(eia.index)


#Lewiston Operations (Diversions)
lewiston_ops = pd.read_csv("Yash/Individual_Meeting_Scripts/Manuscript_Plots/data/Lewiston_Daily_Operations.csv", index_col=0)
lewiston_ops.index = pd.to_datetime(lewiston_ops.index)

#Subset all to a common period of record
eia = eia[eia.index < pd.Timestamp("2023-10-01")]
input_data = input_data[input_data.index < pd.Timestamp("2023-10-01")]
lewiston_ops = lewiston_ops[lewiston_ops.index < pd.Timestamp("2023-10-01")]

eia = eia[eia.index > pd.Timestamp("2005-09-30")]
input_data = input_data[input_data.index > pd.Timestamp("2005-09-30")]
lewiston_ops = lewiston_ops[lewiston_ops.index > pd.Timestamp("2005-09-30")]

#%% Adjust San Luis Values 
daily_data = input_data['SL_storage'].copy()

# Step 1: Create a series of the first day of each month
monthly_data = daily_data.resample('MS').first()  # MS = Month Start

# Step 2: Calculate month-to-month differences
monthly_diff = monthly_data.diff()

# Step 3: For each month, distribute the difference evenly across all days
result = daily_data.copy()

for i in range(1, len(monthly_diff.index)):
    current_month_start = monthly_diff.index[i]
    prev_month_start = monthly_diff.index[i-1]
    
    # Get all days in the current month
    mask = (daily_data.index >= prev_month_start) & (daily_data.index < current_month_start)
    days_in_month = sum(mask)
    
    if days_in_month > 0:
        # Calculate daily increment to distribute the monthly difference
        daily_increment = monthly_diff.iloc[i] / days_in_month
        
        # Apply the daily increment to each day in the month
        # Create an array of cumulative increments
        day_indices = np.arange(1, days_in_month + 1)
        increments = day_indices * daily_increment
        
        # Add these increments to the base value
        result.loc[mask] = daily_data.loc[mask] + increments

input_data['SL_storage'] = result


#%% Compute total hydropower and individual hydropower for CALFEWS

#Get the storage data set
storage_input = input_data[['SHA_storage', 'TRT_storage', 'FOL_storage', \
                             'NML_storage', 'SL_storage']]
storage_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones', 'San Luis']
storage_input = storage_input/1000 #Convert to TAF

#Get the releases dataset
release_input = input_data[['SHA_otf', 'TRT_otf', 'FOL_otf', \
                             'NML_otf']]
release_input.columns = ['Shasta', 'Trinity', 'Folsom', 'New Melones']
release_input.loc[:,'Diversions'] = lewiston_ops['Diversion']
release_input = release_input*cfs_to_tafd
release_input['San Luis'] =  storage_input['San Luis'].diff().mul(-1).fillna(0)
release_input.loc[(release_input.index.month == 10) & (release_input.index.day == 1), 'San Luis'] = 0

#Compute the hydropower generation 
cvp_gen = get_CVP_hydro_gen(storage_input, release_input)
cvp_gen = cvp_gen.resample('MS').sum()
cvp_gen = cvp_gen.drop('CVP_Gen', axis=1)

#Rearrange the EIA columns
eia = eia[['Shasta', 'Trinity', 'Judge F Carr', 'Spring Creek', 'Keswick', 'Folsom', 'Nimbus', 'New Melones', 'W R Gianelli', 'ONeill']]
eia.columns = cvp_gen.columns
eia.index = cvp_gen.index

# %% Simpler versions

# %%

# Just the EIA Geneation data
correlation = np.corrcoef(
    cvp_gen['Shasta'].dropna(),
    eia['Shasta'].dropna())[0,1]

# Create a figure with a special layout
plt.figure(figsize=(20, 8))
plt.plot(eia.index, eia['Shasta'], 'b-', 
         linewidth=2, label="EIA Gen")
plt.plot(cvp_gen.index, cvp_gen['Shasta'], 'r-', 
         linewidth=0, label=" ")
plt.ylabel("Hydropower Generation (GWh)", fontsize=24)
plt.xlabel("Month", fontsize=24)
plt.tick_params(axis='both', labelsize=18)
plt.title(f"Monthly Hydropower Generation at Shasta", 
              fontsize=28)
plt.grid(True)
plt.legend(loc='upper right', fontsize=18, frameon=True)
plt.tight_layout()
plt.show()



# Just the EIA Geneation data
correlation = np.corrcoef(
    cvp_gen['Shasta'].dropna(),
    eia['Shasta'].dropna())[0,1]

# Create a figure with a special layout
plt.figure(figsize=(20, 8))
plt.plot(eia.index, eia['Shasta'], 'b-', 
         linewidth=2, label="EIA Gen")
plt.plot(cvp_gen.index, cvp_gen['Shasta'], 'r-', 
         linewidth=2, label="CDEC Estimated Gen")
plt.ylabel("Hydropower Generation (GWh)", fontsize=24)
plt.xlabel("Month", fontsize=24)
plt.tick_params(axis='both', labelsize=18)
plt.title(f"Monthly Hydropower Generation at Shasta", 
              fontsize=28)
plt.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.12, 0.95), xycoords='axes fraction',
            fontsize=28, ha='left', va='top')
plt.grid(True)
plt.legend(loc='upper right', fontsize=18, frameon=True)
plt.tight_layout()
plt.show()




# %%

# First compute the correlation
correlation = np.corrcoef(
    cvp_gen['Shasta'].dropna(),
    eia['Shasta'].dropna())[0,1]

# Create a figure with a special layout
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, 4)  # 4 columns total

# Main time series plot
ax1 = fig.add_subplot(gs[0, 0:3])  # First 3 columns
ax1.plot(eia.index, eia['Shasta'], 'b-', 
         linewidth=2, label="EIA Gen")
ax1.plot(cvp_gen.index, cvp_gen['Shasta'], 'r-', 
         linewidth=2, label="CDEC Estimated Gen")
ax1.set_ylabel("Hydropower Generation (GWh)", fontsize=24)
ax1.set_xlabel("Month", fontsize=24)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_title(f"Monthly Hydropower Generation at Shasta", 
              fontsize=28)
ax1.grid(True)

# Place legend in the upper right corner of the left plot
ax1.legend(loc='upper right', fontsize=18, frameon=True)

# Scatter plot
ax2 = fig.add_subplot(gs[0, 3])  # Last column
ax2.scatter(eia['Shasta'].dropna(), cvp_gen['Shasta'].dropna(),  
           color='green', alpha=0.7, s=30)

# Add 1:1 line
max_val = max(cvp_gen['Shasta'].max(), eia['Shasta'].max())
min_val = min(cvp_gen['Shasta'].min(), eia['Shasta'].min())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

# Add correlation text to scatter plot
ax2.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')

# Set scatter plot labels and appearance
ax2.set_xlabel("EIA Gen (GWh)", fontsize=24)
ax2.set_ylabel("CDEC Gen (GWh)", fontsize=24)  # Fixed typo: Gwh → GWh
ax2.tick_params(axis='both', labelsize=18)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()