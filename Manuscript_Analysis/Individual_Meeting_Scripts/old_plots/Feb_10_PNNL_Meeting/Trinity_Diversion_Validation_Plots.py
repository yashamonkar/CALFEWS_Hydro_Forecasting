# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:37:25 2025

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


#%% Trinity Diversion Storage Validation plots 
# First compute the correlation
correlation = np.corrcoef(input_data['TRT_storage']/1000, datDaily['trinity_S'])[0,1]

plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['TRT_storage']/1000, 'b-', 
         linewidth=0, label = "CDEC Storage")
plt.plot(datDaily.index, datDaily['trinity_S'], 'r-', 
         linewidth=2, label = "CALFEWS Storage")
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title(f"Trinity Reservoir Storage Levels \n",  fontsize=28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=24, ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(input_data.index, input_data['TRT_storage']/1000, 'b-', 
         linewidth=2, label = "CDEC Storage")
plt.plot(datDaily.index, datDaily['trinity_S'], 'r-', 
         linewidth=2, label = "CALFEWS Storage")
plt.ylabel("Storage (TAF)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title(f"Trinity Reservoir Storage Levels \nPearson Correlation: {correlation:.2f}",  fontsize=28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=24, ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()



#%% Trinity Diversion Releases Validation plots 
# First compute the correlation
calfews_release = datDaily['trinity_R'].resample('MS').sum()
cdec_releases = cfs_tafd*input_data['TRT_otf'].resample('MS').sum()

correlation = np.corrcoef(cdec_releases, calfews_release)[0,1]

plt.figure(figsize = (15, 8))
plt.plot(cdec_releases.index, cdec_releases, 'b-', 
         linewidth=0, label = "CDEC Storage")
plt.plot(calfews_release.index, calfews_release, 'r-', 
         linewidth=2, label = "CALFEWS Storage")
plt.ylabel("Release (TAF)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title(f"Trinity Reservoir Monthly Releases \n",  fontsize=28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=24, ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(cdec_releases.index, cdec_releases, 'b-', 
         linewidth=2, label = "CDEC Storage")
plt.plot(calfews_release.index, calfews_release, 'r-', 
         linewidth=2, label = "CALFEWS Storage")
plt.ylabel("Release (TAF)", fontsize = 24)
plt.xlabel("Month", fontsize=24)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.title(f"Trinity Reservoir Monthly Releases \nPearson Correlation: {correlation:.2f}",  fontsize=28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=24, ncol = 3, frameon = False)
plt.grid(True)
plt.tight_layout()





