# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:28:11 2025

@author: amonkar

Script to plot the results for the monthly uniform release assumptiosn
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
import matplotlib.dates as mdates

# Import necessary libraries for custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

#Extract custom functions
from WAPA_Runs.post_processing.functions.get_CALFEWS_results import get_results_sensitivity_number_outside_model
from WAPA_Runs.post_processing.functions.get_comparison_plots import get_comparison_plots
from WAPA_Runs.post_processing.functions.get_CVP_hydro_gen import get_CVP_hydro_gen


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

# %% SHASTA

#Water Year (WY 2017
input_subset = input_data[input_data.index > pd.Timestamp("2010-10-01")]
input_subset = input_subset[input_subset.index < pd.Timestamp("2011-10-01")]
datDaily_subset = datDaily[datDaily.index > pd.Timestamp("2010-10-01")]
datDaily_subset = datDaily_subset[datDaily_subset.index < pd.Timestamp("2011-10-01")]


#%% Plotting the releases 

# First calculate monthly averages
monthly_avg = input_subset.resample('M').mean()

# Create a daily series with the monthly average repeated for each day
monthly_avg_daily = pd.DataFrame(index=input_subset.index)
monthly_avg_daily = input_subset.groupby(pd.Grouper(freq='M')).transform('mean')

# Option 1: Water year month starts (Oct 2010 - Sep 2011)
# For water year 2011, start from Oct 2010
month_starts = []
# October 2010 through December 2010
for month in range(10, 13):
    month_starts.append(pd.Timestamp(f'2010-{month:02d}-15'))
# January 2011 through September 2011
for month in range(1, 10):
    month_starts.append(pd.Timestamp(f'2011-{month:02d}-15'))
    
# Create month labels in water year order
month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

# Import matplotlib.dates for formatting


# Option 1: Water year month starts (Oct 2010 - Sep 2011)
# For water year 2011, start from Oct 2010
month_lines = []
# October 2010 through December 2010
for month in range(10, 13):
    month_lines.append(pd.Timestamp(f'2010-{month:02d}-01'))
# January 2011 through September 2011
for month in range(1, 10):
    month_lines.append(pd.Timestamp(f'2011-{month:02d}-01'))


#%% SHASTA
# Combined plot with subplots
plt.figure(figsize=(27, 18))  # Taller figure to accommodate both plots

# Shasta plot (top)
ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, position 1
plt.fill_between(input_subset.index, 0, input_subset['SHA_otf']*cfs_tafd, alpha=0.5, color='lightblue')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['SHA_otf']*cfs_tafd, alpha=0.5, color='yellow')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['shasta_R'], alpha=0.0, color='red')
plt.axhline(y=34.909, color='black', linestyle='-.', linewidth=4)
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['SHA_otf']*cfs_tafd, 'y-', linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', linewidth=0)
plt.ylabel("Daily Releases (TAF)", fontsize=32)
plt.xlabel("Month", fontsize=32)
plt.yticks(fontsize=28)
ax1.set_xticks(month_starts)
ax1.set_xticklabels(month_labels, fontsize=28)

plt.text(0.02, 0.95, 'A', transform=plt.gca().transAxes, 
         fontsize=36, fontweight='bold', ha='right', va='top')
diff = (input_subset['SHA_otf'] * cfs_tafd).clip(upper=34.909) - (monthly_avg_daily['SHA_otf']*cfs_tafd).clip(upper=34.909)
rmse = np.sqrt(np.mean(diff**2))
plt.text(0.12, 0.85, f'RMSE={rmse:.1f} TAF', transform=plt.gca().transAxes, 
         fontsize=28, ha='right', va='top')

for month_start in month_lines:
    plt.axvline(x=month_start, color='gray', linestyle='--', alpha=0.2, linewidth=3)

# Add the legend with custom handles
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.5, label='CDEC (Daily)'),
    Patch(facecolor='yellow', alpha=0.5, label='CDEC (Monthly)'),
    Patch(facecolor='red', alpha=0.25, label='CALFEWS'),
    Line2D([0], [0], color='black', linestyle='-.', linewidth=4, label='Penstock Capacity')
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=36, framealpha = 1.0, facecolor = "white")

# Shasta plot (bottom)
ax2 = plt.subplot(2, 1, 2)
plt.fill_between(input_subset.index, 0, input_subset['SHA_otf']*cfs_tafd, alpha=0.5, color='lightblue')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['SHA_otf']*cfs_tafd, alpha=0.0, color='yellow')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['shasta_R'], alpha=0.25, color='red')
plt.axhline(y=34.909, color='black', linestyle='-.', linewidth=4)
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['SHA_otf']*cfs_tafd, 'y-', linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', linewidth=2)
plt.ylabel("Daily Releases (TAF)", fontsize=32)
plt.xlabel("Month", fontsize=32)
plt.yticks(fontsize=28)
ax2.set_xticks(month_starts)
ax2.set_xticklabels(month_labels, fontsize=28)

plt.text(0.02, 0.95, 'B', transform=plt.gca().transAxes, 
         fontsize=36, fontweight='bold', ha='right', va='top')
diff = (input_subset['SHA_otf'] * cfs_tafd).clip(upper=34.909) - (datDaily_subset['shasta_R']).clip(upper=34.909)
rmse = np.sqrt(np.mean(diff**2))
plt.text(0.12, 0.85, f'RMSE={rmse:.1f} TAF', transform=plt.gca().transAxes, 
         fontsize=28, ha='right', va='top')

for month_start in month_lines:
    plt.axvline(x=month_start, color='gray', linestyle='--', alpha=0.2, linewidth=3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Adjust bottom to make room for the legend
#plt.savefig("Combined_Releases_WY2017.png", dpi=300, bbox_inches="tight")
plt.show()




#%% FOLSOM
# Combined plot with subplots
plt.figure(figsize=(27, 18))  # Taller figure to accommodate both plots

# Folsom plot (top)
ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, position 1
plt.fill_between(input_subset.index, 0, input_subset['FOL_otf']*cfs_tafd, alpha=0.5, color='lightblue')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['FOL_otf']*cfs_tafd, alpha=0.5, color='yellow')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['folsom_R'], alpha=0.0, color='red')
plt.axhline(y=13.676, color='black', linestyle='-.', linewidth=4)
plt.plot(input_subset.index, input_subset['FOL_otf']*cfs_tafd, 'b-', linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['FOL_otf']*cfs_tafd, 'y-', linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_R'], 'r-', linewidth=0)
plt.ylabel("Daily Releases (TAF)", fontsize=32)
plt.xlabel("Month", fontsize=32)
plt.yticks(fontsize=28)
ax1.set_xticks(month_starts)
ax1.set_xticklabels(month_labels, fontsize=28)
plt.text(0.02, 0.95, 'A', transform=plt.gca().transAxes, 
         fontsize=36, fontweight='bold', ha='right', va='top')
diff = (input_subset['FOL_otf'] * cfs_tafd).clip(upper=13.676) - (monthly_avg_daily['FOL_otf']*cfs_tafd).clip(upper=34.909)
rmse = np.sqrt(np.mean(diff**2))
plt.text(0.12, 0.85, f'RMSE={rmse:.1f} TAF', transform=plt.gca().transAxes, 
         fontsize=28, ha='right', va='top')
for month_start in month_lines:
    plt.axvline(x=month_start, color='gray', linestyle='--', alpha=0.2, linewidth=3)

# Add the legend with custom handles
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.5, label='CDEC (Daily)'),
    Patch(facecolor='yellow', alpha=0.5, label='CDEC (Monthly)'),
    Patch(facecolor='red', alpha=0.25, label='CALFEWS'),
    Line2D([0], [0], color='black', linestyle='-.', linewidth=4, label='Penstock Capacity')
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=36, framealpha = 1.0, facecolor = "white")

# Folsom plot (bottom)
ax2 = plt.subplot(2, 1, 2)
plt.subplot(2, 1, 2)
plt.fill_between(input_subset.index, 0, input_subset['FOL_otf']*cfs_tafd, alpha=0.5, color='lightblue')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['FOL_otf']*cfs_tafd, alpha=0.0, color='yellow')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['folsom_R'], alpha=0.25, color='red')
plt.axhline(y=13.676, color='black', linestyle='-.', linewidth=4)
plt.plot(input_subset.index, input_subset['FOL_otf']*cfs_tafd, 'b-', linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['FOL_otf']*cfs_tafd, 'y-', linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_R'], 'r-', linewidth=2)
plt.ylabel("Daily Releases (TAF)", fontsize=32)
plt.xlabel("Month", fontsize=32)
plt.yticks(fontsize=28)
ax2.set_xticks(month_starts)
ax2.set_xticklabels(month_labels, fontsize=28)
plt.text(0.02, 0.95, 'B', transform=plt.gca().transAxes, 
         fontsize=36, fontweight='bold', ha='right', va='top')
diff = (input_subset['FOL_otf'] * cfs_tafd).clip(upper=13.676) - (datDaily_subset['folsom_R']).clip(upper=34.909)
rmse = np.sqrt(np.mean(diff**2))
plt.text(0.12, 0.85, f'RMSE={rmse:.1f} TAF', transform=plt.gca().transAxes, 
         fontsize=28, ha='right', va='top')
for month_start in month_lines:
    plt.axvline(x=month_start, color='gray', linestyle='--', alpha=0.2, linewidth=3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Adjust bottom to make room for the legend
#plt.savefig("Combined_Releases_WY2017.png", dpi=300, bbox_inches="tight")
plt.show()

