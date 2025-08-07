# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:44:56 2025

@author: amonkar

Code for the hydropower generation modules. This includes comparing the EIA Gen
vs estimates from CDEC input data. 
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




# %% Main Plot

# Set font sizes
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 24

#Remove San_Luis and Oneill
eia = eia.iloc[:, :-2]
cvp_gen = cvp_gen.iloc[:, :-2]

# Create a 5x2 subplot grid
fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
axes = axes.flatten()  # Flatten to make iteration easier

# Loop through each column and create the subplot
for i, column in enumerate(eia.columns):
    # Plot data for current column
    axes[i].plot(eia.index, eia[column], color='red')
    axes[i].plot(cvp_gen.index, cvp_gen[column], color='blue')
    
    # Set title and grid with larger font
    axes[i].set_title(column, fontsize=TITLE_SIZE, fontweight='bold')
    axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Only add y-label on the left side
    if i % 2 == 0:
        axes[i].set_ylabel('Generation (GWh)', fontsize=AXIS_LABEL_SIZE)
    
    # Increase tick label sizes
    axes[i].tick_params(axis='both', labelsize=TICK_SIZE)

# Adjust spacing between subplots
plt.tight_layout()

# Add more space at the bottom for the legend and time label
plt.subplots_adjust(bottom=0.12)

# Add the time label ABOVE the legend position
fig.text(0.5, 0.09, 'Time', ha='center', fontsize=AXIS_LABEL_SIZE, fontweight='bold')

# Create a single legend at the very bottom with larger font
fig.legend(['EIA Generation', 'CDEC Estimated Gen'], 
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.03),
           ncol=2,
           fontsize=LEGEND_SIZE)

# Show the plot
plt.show()


for i, column in enumerate(eia.columns):
    print(eia.columns[i])
    
    rmse = np.sqrt(((eia[column] - cvp_gen[column]) ** 2).mean())
    #print(f"RMSE: {rmse:.0f}")
    
    correlation = eia[column].corr(cvp_gen[column])**2
    print(f"Correlation: {correlation:.2f}")