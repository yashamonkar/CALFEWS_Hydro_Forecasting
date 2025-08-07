# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:28:11 2025

@author: amonkar

Script to show the results for the storage plots for the meeting. 
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

# %% SHASTA STORAGE

#Water Year (WY 2017
input_subset = input_data[input_data.index > pd.Timestamp("2016-09-30")]
input_subset = input_subset[input_subset.index < pd.Timestamp("2017-10-01")]
datDaily_subset = datDaily[datDaily.index > pd.Timestamp("2016-09-30")]
datDaily_subset = datDaily_subset[datDaily_subset.index < pd.Timestamp("2017-10-01")]

plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 0)
plt.scatter(pd.Timestamp("2016-10-01"), 2803, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Storage Levels \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()

plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.scatter(pd.Timestamp("2016-10-01"), 2803, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Storage Levels \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.scatter(pd.Timestamp("2016-10-01"), 2803, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Storage Levels \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()

#%% FOLSOM PLOT

#Water Year (WY 2017
plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_storage']/1000, 'b-', label = "CDEC Storage", linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_S'], 'r-', label = "CALFEWS Storage", linewidth = 0)
plt.scatter(pd.Timestamp("2016-10-01"), 304.127, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Storage Levels \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()

plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_storage']/1000, 'b-', label = "CDEC Storage", linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.scatter(pd.Timestamp("2016-10-01"), 304.127, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Storage Levels \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.scatter(pd.Timestamp("2016-10-01"), 304.127, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Storage Levels \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


#%% Plotting the releases 

# First calculate monthly averages
monthly_avg = input_subset.resample('M').mean()

# Create a daily series with the monthly average repeated for each day
monthly_avg_daily = pd.DataFrame(index=input_subset.index)
monthly_avg_daily = input_subset.groupby(pd.Grouper(freq='M')).transform('mean')




#%% Shasta Releases

plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', label = "CALFEWS Releases", linewidth =0)
plt.plot(datDaily_subset.index, monthly_avg_daily['SHA_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =0)
plt.fill_between(input_subset.index, 0, input_subset['SHA_otf']*cfs_tafd, alpha=0.5, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['shasta_R'], alpha=0, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['SHA_otf']*cfs_tafd, alpha=0, color='yellow')
plt.axhline(y=39, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['SHA_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', label = "CALFEWS Releases", linewidth =0)
plt.fill_between(input_subset.index, 0, input_subset['SHA_otf']*cfs_tafd, alpha=0, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['shasta_R'], alpha=0, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['SHA_otf']*cfs_tafd, alpha=0.5, color='yellow')
plt.axhline(y=39, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()



plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['SHA_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', label = "CALFEWS Releases", linewidth =2)
plt.fill_between(input_subset.index, 0, input_subset['SHA_otf']*cfs_tafd, alpha=0, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['shasta_R'], alpha=0.5, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['SHA_otf']*cfs_tafd, alpha=0, color='yellow')
plt.axhline(y=39, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['SHA_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', label = "CALFEWS Releases", linewidth =2)
plt.fill_between(input_subset.index, 0, input_subset['SHA_otf']*cfs_tafd, alpha=0, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['shasta_R'], alpha=0.0, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['SHA_otf']*cfs_tafd, alpha=0, color='yellow')
plt.axhline(y=39, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()




#%% Shasta Releases

plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['FOL_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =0)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_R'], 'r-', label = "CALFEWS Releases", linewidth =0)
plt.fill_between(input_subset.index, 0, input_subset['FOL_otf']*cfs_tafd, alpha=0.5, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['folsom_R'], alpha=0.0, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['FOL_otf']*cfs_tafd, alpha=0, color='yellow')
plt.axhline(y=13.676, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()



plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['FOL_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =2)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_R'], 'r-', label = "CALFEWS Releases", linewidth =0)
plt.fill_between(input_subset.index, 0, input_subset['FOL_otf']*cfs_tafd, alpha=0.0, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['folsom_R'], alpha=0.0, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['FOL_otf']*cfs_tafd, alpha=0.5, color='yellow')
plt.axhline(y=13.676, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()



plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['FOL_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =2)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_R'], 'r-', label = "CALFEWS Releases", linewidth =2)
plt.fill_between(input_subset.index, 0, input_subset['FOL_otf']*cfs_tafd, alpha=0.0, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['folsom_R'], alpha=0.5, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['FOL_otf']*cfs_tafd, alpha=0.0, color='yellow')
plt.axhline(y=13.676, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()




plt.figure(figsize = (15, 8))
plt.plot(input_subset.index, input_subset['FOL_otf']*cfs_tafd, 'b-', label = "CDEC Releases", linewidth=2)
plt.plot(datDaily_subset.index, monthly_avg_daily['FOL_otf']*cfs_tafd, 'y-', label = "Monthly Averaged Releases", linewidth =2)
plt.plot(datDaily_subset.index, datDaily_subset['folsom_R'], 'r-', label = "CALFEWS Releases", linewidth =2)
plt.fill_between(input_subset.index, 0, input_subset['FOL_otf']*cfs_tafd, alpha=0, color='lightblue')
plt.fill_between(datDaily_subset.index, 0, datDaily_subset['folsom_R'], alpha=0.0, color='red')
plt.fill_between(input_subset.index, 0, monthly_avg_daily['FOL_otf']*cfs_tafd, alpha=0, color='yellow')
plt.axhline(y=13.676, color='black', linestyle='--', linewidth=2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Folsom Daily Releases \n WY 2017", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 3, frameon = False)
plt.tight_layout()

