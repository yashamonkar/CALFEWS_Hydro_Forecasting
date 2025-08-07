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

#First Year (WY 1996)
#Subset to the current levels
input_subset = input_data[input_data.index < pd.Timestamp("1996-09-30")]
datDaily_subset = datDaily[datDaily.index < pd.Timestamp("1996-09-30")]

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 0)
plt.scatter(pd.Timestamp("1995-10-01"), 3129, color='blue', s=0)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()



plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 0)
plt.scatter(pd.Timestamp("1995-10-01"), 3129, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=0)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.scatter(pd.Timestamp("1995-10-01"), 3129, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.scatter(pd.Timestamp("1996-10-01"), 3092, color='blue', s=50)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()



#Second Year (WY 1997)
#Subset to the current levels
datDaily_subset = datDaily[datDaily.index < pd.Timestamp("1997-09-30")]


plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.scatter(pd.Timestamp("1996-10-01"), 3092, color='blue', s=50)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()

input_subset = input_data[input_data.index < pd.Timestamp("1997-09-30")]

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', linewidth=0)
plt.plot(input_subset.index, input_subset['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()

#Compute the correlation
correlation = stats.pearsonr(input_data['SHA_storage']/1000, 
                           datDaily['shasta_S'])[0]

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily.index, datDaily['shasta_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.06, 0.1), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title("Shasta Storage Levels", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.grid('True')
plt.tight_layout()


###Folsom plot
correlation = stats.pearsonr(input_data['FOL_storage']/1000, 
                           datDaily['folsom_S'])[0]

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['FOL_storage']/1000, 'b-', label = "CDEC Storage", linewidth=2)
plt.plot(datDaily.index, datDaily['folsom_S'], 'r-', label = "CALFEWS Storage", linewidth = 2)
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"Folsom Storage Levels \n R = {round(correlation, 2)}", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


#%% 

#First Year (WY 1996)
#Subset to the current levels
input_subset = input_data[input_data.index < pd.Timestamp("1996-09-30")]
datDaily_subset = datDaily[datDaily.index < pd.Timestamp("1996-09-30")]

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_otf']*cfs_tafd, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_R'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Release Values", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', label = "CALFEWS Release Values", linewidth =0)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Daily Releases \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_otf']*cfs_tafd, 'b-', linewidth=0)
plt.plot(datDaily.index, datDaily['shasta_R'], 'r-', linewidth = 0)
plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Release Values", linewidth=2)
plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', label = "CALFEWS Release Values", linewidth =2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("Shasta Daily Releases \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


correlation = stats.pearsonr(input_data['SHA_otf']*cfs_tafd, 
                           datDaily['shasta_R'])[0]

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Release Values", linewidth=2)
plt.plot(datDaily.index, datDaily['shasta_R'], 'r-', label = "CALFEWS Release Values", linewidth = 2)
#plt.plot(input_subset.index, input_subset['SHA_otf']*cfs_tafd, 'b-', label = "CDEC Release Values", linewidth=2)
#plt.plot(datDaily_subset.index, datDaily_subset['shasta_R'], 'r-', label = "CALFEWS Release Values", linewidth =0)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"Shasta Daily Releases \n Daily R = {round(correlation, 2)}", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()




###Shasta Monthly plot
correlation = stats.pearsonr(cfs_tafd*input_data['SHA_otf'].resample('ME').sum(), 
                           datDaily['shasta_R'].resample('ME').sum())[0]

cdec_monthly = cfs_tafd*input_data['SHA_otf'].resample('ME').sum()
calfews_monthly = datDaily['shasta_R'].resample('ME').sum()


plt.figure(figsize = (15, 8))
plt.plot(cdec_monthly.index, cdec_monthly, 'b-', label = "CDEC Release Values", linewidth=2)
plt.plot(calfews_monthly.index, calfews_monthly, 'r-', label = "CALFEWS Release Values", linewidth = 2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"Shasta Daily Releases \n Monthly R = {round(correlation, 2)}", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()




###Folsom Monthly plot
correlation = stats.pearsonr(cfs_tafd*input_data['FOL_otf'].resample('ME').sum(), 
                           datDaily['folsom_R'].resample('ME').sum())[0]

cdec_monthly = cfs_tafd*input_data['FOL_otf'].resample('ME').sum()
calfews_monthly = datDaily['folsom_R'].resample('ME').sum()


plt.figure(figsize = (15, 8))
plt.plot(cdec_monthly.index, cdec_monthly, 'b-', label = "CDEC Release Values", linewidth=2)
plt.plot(calfews_monthly.index, calfews_monthly, 'r-', label = "CALFEWS Release Values", linewidth = 2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"Folsom Daily Releases \n Monthly R = {round(correlation, 2)}", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


###Trinity Monthly plot
correlation = stats.pearsonr(cfs_tafd*input_data['TRT_otf'].resample('ME').sum(), 
                           datDaily['trinity_R'].resample('ME').sum())[0]

cdec_monthly = cfs_tafd*input_data['TRT_otf'].resample('ME').sum()
calfews_monthly = datDaily['trinity_R'].resample('ME').sum()


plt.figure(figsize = (15, 8))
plt.plot(cdec_monthly.index, cdec_monthly, 'b-', label = "CDEC Release Values", linewidth=2)
plt.plot(calfews_monthly.index, calfews_monthly, 'r-', label = "CALFEWS Release Values", linewidth = 2)
plt.ylabel("Releases (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"Trinity Daily Releases \n Monthly R = {round(correlation, 2)}", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()