# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:26:35 2025

@author: amonkar

Code to analyze outputs from the annual simulation runs. 

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


#Read the lewiston reports (to get true diversions)
lewiston = pd.read_csv("C:/Users/amonkar/Documents/CALFEWS_Preliminary/data/Lewiston_Daily_Operations.csv", index_col=0)
lewiston.index = pd.DatetimeIndex(lewiston.index)
lewiston = lewiston[lewiston.index < pd.Timestamp('2023-10-01')]
lewiston['Diversion'] = lewiston['Diversion']*cfs_tafd


#Read the WAPA generation dataset
wapa = pd.read_csv('Yash/WAPA/WAPA_Daily_Gen.csv', index_col=0)
wapa.index = pd.to_datetime(wapa.index)

#Read the WAPA generation dataset
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)
eia = eia/1000
eia.index = pd.to_datetime(eia.index)


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
    yearly_data = yearly_data[['shasta_S', 'shasta_R', 'oroville_S', 'oroville_R', 'trinity_S', \
                               'trinity_R', 'folsom_S', 'folsom_R', 'newmelones_S', 'newmelones_R', \
                                   'sanluisstate_S', 'sanluisfederal_S', 'trinity_diversions', 'delta_TRP_pump']]
    
    # Append to our list
    all_data.append(yearly_data)
    
# Concatenate all DataFrames into one
datDaily = pd.concat(all_data,)
datDaily.head()

#%% Compare the results 

#Set the labels
labels = ['Shasta', 'Oroville', 'Trinity', 'Folsom', 'New Melones', 'San Luis State', \
          'San Luis Federal', 'San Luis']

###-------------------------Storage------------------------------------------#
calfews_storage = datDaily[['shasta_S', 'oroville_S','trinity_S', 'folsom_S',\
                            'newmelones_S', 'sanluisstate_S', 'sanluisfederal_S']]
calfews_storage['San Luis'] = calfews_storage['sanluisstate_S'] + calfews_storage['sanluisfederal_S']
calfews_storage.columns = labels

true_storage = input_data[['SHA_storage', 'ORO_storage', 'TRT_storage', \
                           'FOL_storage', 'NML_storage', \
                           'SLS_storage', 'SLF_storage', 'SL_storage']]
true_storage.columns = labels
true_storage = true_storage/1000

get_comparison_plots(model_output = calfews_storage, 
                     data_input = true_storage,
                     labels = labels, 
                     y_axis_label = "Storage", 
                     units = 'TAF',
                     plot_legends = ['CALFEWS Storage', 'CDEC Storage'],
                     monthly_method='average')



###-------------------------Releases------------------------------------------#
#Set the labels
labels = ['Shasta', 'Oroville', 'Trinity', 'Folsom', 'New Melones', 'San Luis']

    
calfews_releases = datDaily[['shasta_R', 'oroville_R','trinity_R', 'folsom_R',\
                            'newmelones_R']]
calfews_releases['San Luis'] =  calfews_storage['San Luis'].diff().mul(-1).fillna(0)
calfews_releases.loc[(calfews_releases.index.month == 10) & 
                     (calfews_releases.index.day == 1), 'San Luis'] = 0 #Changed to zero to account for the difference
calfews_releases.columns = labels

true_releases = input_data[['SHA_otf', 'ORO_otf', 'TRT_otf', \
                           'FOL_otf', 'NML_otf']]
true_releases = true_releases*cfs_tafd
true_releases['San Luis'] =  true_storage['San Luis'].diff().mul(-1).fillna(0)
true_releases.columns = labels

get_comparison_plots(model_output = calfews_releases, 
                     data_input = true_releases,
                     labels = labels, 
                     y_axis_label = "Releases", 
                     units = 'TAF',
                     plot_legends = ['CALFEWS Releases', 'CDEC Releases'],
                     monthly_method='sum')

#-----------------------------------------------------------------------------#
#Subset to WY2017
#calfews_releases = calfews_releases[calfews_releases.index > pd.Timestamp('2017-01-01')]
#calfews_releases = calfews_releases[calfews_releases.index < pd.Timestamp('2017-05-01')]

#true_releases = true_releases[true_releases.index > pd.Timestamp('2017-01-01')]
#true_releases = true_releases[true_releases.index < pd.Timestamp('2017-05-01')]

#calfews_releases = calfews_releases[['Shasta', 'Folsom']]
#true_releases = true_releases[['Shasta', 'Folsom']]






#%% Compute the total CVP Hydropower Generation


#-------------------------------CALFEWS---------------------------------------#
#Set up the storage inputs for the get_CVP_hydro_gen
cvp_storage_input = calfews_storage[['Shasta', 'Trinity', 'Folsom', \
                                     'New Melones', 'San Luis']]

#Set up the releases inputs for the get_CVP_hydro_gen
cvp_release_input = calfews_releases[['Shasta', 'Trinity', 'Folsom', 'New Melones']]
cvp_release_input.loc[:,'Diversions'] = datDaily['trinity_diversions']
cvp_release_input.loc[:, 'San Luis'] = cvp_storage_input['San Luis'].diff().mul(-1).fillna(0)
cvp_release_input.loc[(cvp_release_input.index.month == 10) & 
                     (cvp_release_input.index.day == 1), 'San Luis'] = 0 #Changed to zero to account for the difference

calfews_cvp_gen = get_CVP_hydro_gen(cvp_storage_input, cvp_release_input)


#-------------------------------True Flows------------------------------------#
#Set up the storage inputs for the get_CVP_hydro_gen
cvp_storage_input = true_storage[['Shasta', 'Trinity', 'Folsom', \
                                     'New Melones', 'San Luis']]
#Subset the 2001-10-01 (Trinity Diversions not available before that )
cvp_storage_input = cvp_storage_input[cvp_storage_input.index > pd.Timestamp('2000-09-30')]
    
#Set up the releases inputs for the get_CVP_hydro_gen
cvp_release_input = true_releases[['Shasta', 'Trinity', 'Folsom', 'New Melones']]
cvp_release_input.loc[:, 'San Luis'] = 0.001*input_data['SL_storage'].diff().mul(-1).fillna(0)

#Convert San Luis Flows to daily.
def distribute_monthly_values(df):
    # Get monthly values (on 1st of each month)
    monthly_values = df.loc[df.index.day == 1, 'San Luis']
    
    # For each month
    for date in monthly_values.index:
        # Get days in that month
        days_in_month = date.days_in_month
        # Calculate daily value
        daily_value = monthly_values[date] / days_in_month
        # Apply to all days in that month
        month_mask = (df.index.year == date.year) & (df.index.month == date.month)
        df.loc[month_mask, 'San Luis'] = daily_value
        
    return df

#cvp_release_input = distribute_monthly_values(cvp_release_input)

#Add the diversions
cvp_release_input = cvp_release_input[cvp_release_input.index > pd.Timestamp('2000-09-30')]
cvp_release_input.loc[:,'Diversions'] = lewiston['Diversion']

estimated_cvp_gen = get_CVP_hydro_gen(cvp_storage_input, cvp_release_input)


#Compare the values
labels = estimated_cvp_gen.columns

#Curtail the CALFEWS run to after 2000-09-30
calfews_cvp_gen_sub = calfews_cvp_gen[calfews_cvp_gen.index > pd.Timestamp('2000-09-30')]

#Convert the Nanss to zero
calfews_cvp_gen_sub = calfews_cvp_gen_sub.replace([np.inf, -np.inf, np.nan], 0)

get_comparison_plots(model_output = calfews_cvp_gen_sub, 
                     data_input = estimated_cvp_gen,
                     labels = labels, 
                     y_axis_label = "Generation", 
                     units = 'GWh',
                     plot_legends = ['CALFEWS Gen', 'CDEC Gen'],
                     monthly_method='sum')

# %% WAPA Generation

#Compare calfews generation to WAPA generation
wapa_gen = wapa[['Shasta', 'Trinity', 'Judge F Carr', 'Spring Creek', \
                 'Keswick', 'Folsom', 'Nimbus', 'New Melones']]
wapa_gen = wapa_gen/10**3
    
calfews_cvp_gen_sub = calfews_cvp_gen[['Shasta', 'Trinity', 'Carr', \
                                       'Spring_Creek', 'Keswick', 'Folsom', \
                                           'Nimbus', 'New_Melones']]

#Subset to the wapa_gen dates
calfews_cvp_gen_sub = calfews_cvp_gen_sub[calfews_cvp_gen_sub.index > pd.Timestamp('2016-09-30')]
wapa_gen = wapa_gen[wapa_gen.index > pd.Timestamp('2016-09-30')]

# Get common indices
common_indices = wapa_gen.index.intersection(calfews_cvp_gen_sub.index)
wapa_gen = wapa_gen.loc[common_indices]
calfews_cvp_gen_sub = calfews_cvp_gen_sub.loc[common_indices]
wapa_gen.columns = calfews_cvp_gen_sub.columns

wapa_gen['CVP_Gen'] = wapa_gen.sum(axis=1)
calfews_cvp_gen_sub['CVP_Gen'] = calfews_cvp_gen_sub.sum(axis=1)
calfews_cvp_gen_sub = calfews_cvp_gen_sub.replace([np.inf, -np.inf, np.nan], 0)
    
get_comparison_plots(model_output = calfews_cvp_gen_sub, 
                     data_input = wapa_gen,
                     labels = wapa_gen.columns, 
                     y_axis_label = "Generation", 
                     units = 'GWh',
                     plot_legends = ['CALFEWS Gen', 'WAPA Gen (w/o SL)'],
                     monthly_method='sum')    


#------------------------------Subset plots-----------------------------------#
calfews_cvp_gen_sub =  calfews_cvp_gen_sub[['Shasta', 'New_Melones', 'Folsom', \
                                            'Trinity', 'CVP_Gen']]
wapa_gen =  wapa_gen[['Shasta', 'New_Melones', 'Folsom', 'Trinity', 'CVP_Gen']]

get_comparison_plots(model_output = calfews_cvp_gen_sub, 
                     data_input = wapa_gen,
                     labels = wapa_gen.columns, 
                     y_axis_label = "Generation", 
                     units = 'GWh',
                     plot_legends = ['CALFEWS Gen', 'WAPA Gen (w/o SL)'],
                     monthly_method='sum')    
    

# %% EIA Generation

#Convert the CALFEWS gen to monthly
calfews_cvp_gen_monthly = calfews_cvp_gen.resample('ME').sum()
calfews_cvp_gen_monthly = calfews_cvp_gen_monthly[calfews_cvp_gen_monthly.index > pd.Timestamp('2002-12-31')]

#EIA Generation
eia_monthly = eia[['Shasta', 'Trinity', 'Judge F Carr', 'Spring Creek', 'Keswick', \
                   'Folsom', 'Nimbus', 'New Melones', 'W R Gianelli', 'ONeill',]]
eia_monthly['CVP_Gen'] = eia_monthly.sum(axis=1)
eia_monthly.columns = calfews_cvp_gen_monthly.columns
eia_monthly = eia_monthly[eia_monthly.index < pd.Timestamp('2023-10-01')]
    
get_comparison_plots(model_output = calfews_cvp_gen_monthly, 
                     data_input = eia_monthly,
                     labels = eia_monthly.columns, 
                     y_axis_label = "Generation", 
                     units = 'GWh',
                     plot_legends = ['CALFEWS Gen', 'EIA Gen (with SL)'],
                     monthly_method='sum')  

#------------------------------Subset plots-----------------------------------#
calfews_cvp_gen_monthly_sub =  calfews_cvp_gen_monthly[['Shasta', 'San_Luis', \
                                                    'New_Melones', 'Folsom', \
                                                        'Trinity', 'CVP_Gen']]

eia_monthly_sub = eia_monthly[['Shasta', 'San_Luis', 'New_Melones', 'Folsom', \
                           'Trinity', 'CVP_Gen']]

get_comparison_plots(model_output = calfews_cvp_gen_monthly_sub, 
                     data_input = eia_monthly_sub,
                     labels = eia_monthly_sub.columns, 
                     y_axis_label = "Generation", 
                     units = 'GWh',
                     plot_legends = ['CALFEWS Gen', 'EIA Gen (with SL)'],
                     monthly_method='sum') 






















