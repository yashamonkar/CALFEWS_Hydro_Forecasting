# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:39:34 2025

@author: amonkar

Plots for the operational forecasts starting Oct-1st
1. CALFEWS
2. CALFEWS with operational. 
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS_Hydro_Forecasting'
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
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec
from scipy import stats

#Hyper-parameters
cfs_tafd = 2.29568411*10**-5 * 86400 / 1000


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)

#Read the WAPA generation dataset
eia = pd.read_csv('Manuscript_Plots/data/EIA_Monthy_Gen.csv', index_col=0)
eia = eia/1000 #Convert to GWh
eia.index = pd.to_datetime(eia.index)
eia = eia.drop(['W R Gianelli', 'ONeill'], axis=1)
eia['CVP_Gen'] = eia.sum(axis=1)
eia = eia[eia.index < pd.Timestamp("2023-10-01")]
eia = eia[eia.index > pd.Timestamp("2003-09-01")]

#Read the Sacramento Water Year Types
wy_types = pd.read_csv('Manuscript_Plots/data/cdec-water-year-type.csv', index_col=0)
wy_types['WYT'][wy_types.index == 2017] = 'EW'
wy_types['WYT'][wy_types.index == 2006] = 'EW'


#%% Compute the exceedance probabilities

#Initialize the empty dataframes
p10 = []
p90 = []
p50 = []
p75 = []
p25 = []
p95 = []

for year in range(1996,2024):
    
    print(year)
    
    # Initialize an empty list to store DataFrames
    est_gen = []
    
    #Second loop
    for all_years in range(1996,2024):
        cvp_gen = pd.read_csv(f"Annual_Ensembles/results/{year}/{all_years}/CVP_Generation.csv", index_col =0)
        
        # Append to our list
        est_gen.append(cvp_gen)
    
    #Exit the loop and compute the exceedances for that year
    scenario_matrix = np.array([data['CVP_Gen'] for data in est_gen])
    p10_sub = np.percentile(scenario_matrix, 10, axis=0)
    p90_sub = np.percentile(scenario_matrix, 80, axis=0)
    p50_sub = np.percentile(scenario_matrix, 50, axis=0)
    p75_sub = np.percentile(scenario_matrix, 75, axis=0)
    p25_sub = np.percentile(scenario_matrix, 25, axis=0)
    p95_sub = np.percentile(scenario_matrix, 95, axis=0)
    
    
    #Save the exceedances in a bigger loop
    p10.append(p10_sub)
    p90.append(p90_sub)
    p50.append(p50_sub)
    p75.append(p75_sub)
    p25.append(p25_sub)
    p95.append(p95_sub)
    
    
    
p10_array = np.array(p10).flatten()
p90_array = np.array(p90).flatten()
p50_array = np.array(p50).flatten()
p75_array = np.array(p75).flatten()
p25_array = np.array(p25).flatten()
p95_array = np.array(p95).flatten()

# Create the DataFrame with the three percentile columns
exceedances = pd.DataFrame({
    'p10': p10_array,
    'p50': p50_array,
    'p90': p90_array,
    'p75': p75_array,
    'p25': p25_array,
    'p95': p95_array
})
    
#Set the index for the exceedance data frame
total_months = len(exceedances)
start_date = '1995-10-01'
date_index = pd.date_range(start=start_date, periods=total_months, freq='MS')
exceedances.index = date_index    
exceedances = exceedances[exceedances.index > pd.Timestamp("2003-09-01")]


def get_water_year(date):
    """
    Convert calendar date to water year.
    Water year runs from Oct 1 to Sep 30.
    """
    return date.to_series().apply(lambda x: x.year + 1 if x.month >= 10 else x.year)

#Compute the water year type
exceedances['WY'] = get_water_year(exceedances.index)
eia['WY'] = get_water_year(exceedances.index)

#Compute the annual CVP GEN
annual_eia = eia.groupby('WY')['CVP_Gen'].sum()
annual_p50 = exceedances.groupby('WY')['p50'].sum()

#Compute the exceedance probabilities using the known Water Year Type
exceedances = exceedances.merge(wy_types[['WYT']], left_on='WY', right_index=True, how='left') 
exceedances['WYT'] = exceedances['WYT'].str.strip()

#Adjust the values based on the WYT on April 1st. 
exceedances['month'] = exceedances.index.month
exceedances['adjusted_gen_v2'] = exceedances['p50']
apr_sep_mask = exceedances['month'].isin([4, 5, 6, 7, 8, 9])


critical_mask = apr_sep_mask & exceedances['WYT'].isin(['C'])
exceedances.loc[critical_mask, 'adjusted_gen_v2'] = exceedances.loc[critical_mask, 'p10']

dry_mask = apr_sep_mask & exceedances['WYT'].isin(['D'])
exceedances.loc[dry_mask, 'adjusted_gen_v2'] = exceedances.loc[dry_mask, 'p25']

below_normal_mask = apr_sep_mask & exceedances['WYT'].isin(['BN'])
exceedances.loc[below_normal_mask, 'adjusted_gen_v2'] = exceedances.loc[below_normal_mask, 'p25']

wet_mask = apr_sep_mask & exceedances['WYT'].isin(['W'])
exceedances.loc[wet_mask, 'adjusted_gen_v2'] = exceedances.loc[wet_mask, 'p90']

wet_mask = apr_sep_mask & exceedances['WYT'].isin(['EW'])
exceedances.loc[wet_mask, 'adjusted_gen_v2'] = exceedances.loc[wet_mask, 'p95']

# Clean up temporary month column if desired
exceedances = exceedances.drop('month', axis=1)

#%%




#%% Panel plot for the 5 years of data. 

# Selected water years
selected_years = [2014, 2020, 2012, 2011]
year_types = ['Critical', 'Dry', 'Below Normal', 'Wet']

# Create figure with subplots - 3 rows, 2 columns
fig = plt.figure(figsize=(22.5, 15))

# Define the subplot positions for the 2x2 + 1 center layout
# Using gridspec for better control of the bottom centered plot
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 4, figure=fig)
subplot_positions = [
    gs[0, 0:2],  # Top left
    gs[0, 2:4],  # Top right
    gs[1, 0:2],  # Middle left
    gs[1, 2:4],  # Middle right
]

# Store handles and labels for the legend (we'll get them from the first subplot)
legend_handles = None
legend_labels = None

# Create subplots for each water year
for i, year in enumerate(selected_years):
    # Filter data for the specific water year
    eia_wy = eia[eia['WY'] == year]
    exceedances_wy = exceedances[exceedances['WY'] == year]
    
    # Create subplot
    ax = fig.add_subplot(subplot_positions[i])
    
    # Add April vertical lines
    april_dates = exceedances_wy.index[exceedances_wy.index.month == 4]
    for apr_date in april_dates:
        ax.axvline(x=apr_date, color='gray', linestyle='--', alpha=1, linewidth=3)
    
    # Plot the data (same as original)
    h2 = ax.plot(exceedances_wy.index, exceedances_wy['p50'], 
                color='blue', linestyle='--', alpha=0.0, linewidth=3, label='p50')[0]
    h1 = ax.fill_between(exceedances_wy.index, exceedances_wy['p10'], exceedances_wy['p90'], 
                        alpha=0.2, color='blue', label='p10-p90 Range')
    h3 = ax.plot(eia_wy.index, eia_wy['CVP_Gen'], 
                color='red', linewidth=3, label='EIA Gen')[0]
    h4 = ax.plot(exceedances_wy.index, exceedances_wy['adjusted_gen_v2'], 
                color='blue',  linewidth=3, label='CALFEWS Forecast')[0]
    
    # Store legend handles and labels from first subplot
    if i == 0:
        legend_handles = [h1, h3, h4]
        legend_labels = ['p10-p90 Range', 'EIA', 'CALFEWS']
    
    # Add October vertical lines
    october_dates = exceedances_wy.index[exceedances_wy.index.month == 10]
    for oct_date in october_dates:
        ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0, linewidth=3)
    
    
    # Formatting
    if i > 1:
        ax.set_xlabel('Month', fontsize=32)
    else:
        ax.set_xlabel(' ', fontsize=32)
    if i%2 ==0:
        ax.set_ylabel('Hydropower (GWh)', fontsize=28)
    else:
        ax.set_ylabel(' ', fontsize=28)
    ax.set_ylim(0, exceedances['p90'].max() * 1.1)  # Use global max for consistency
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, alpha=0.5)
    
    # Set custom month labels for x-axis
    month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    if len(exceedances_wy) > 0:
        # Get the monthly tick positions
        monthly_ticks = [exceedances_wy.index[exceedances_wy.index.month == month][0] 
                        for month in [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
                        if len(exceedances_wy.index[exceedances_wy.index.month == month]) > 0]
        ax.set_xticks(monthly_ticks)
        ax.set_xticklabels(month_labels[:len(monthly_ticks)], fontsize=24)
    
    # Add year label with type in top-left corner
    ax.text(0.02, 0.92, f'{year_types[i]} (WY {year})', transform=ax.transAxes, fontsize=28, 
            verticalalignment='top', horizontalalignment='left', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))

# Add a single legend at the bottom of the entire figure
fig.legend(legend_handles, legend_labels, loc='lower center', ncol=4, fontsize=28, 
           bbox_to_anchor=(0.5, -0.02))

# Adjust layout - leave more space at bottom for legend
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.12)  # Make room for legend at bottom
plt.show()



#%% Supplemental Material Plots for adjusted CALFEWS for all years

fig, ax = plt.subplots(figsize=(18, 9))
ax.fill_between(exceedances.index, exceedances['p10'], exceedances['p90'], 
                alpha=0.4, color='blue', label='p10-p90 Range')
ax.plot(exceedances.index, exceedances['p50'], 
        color='blue', linestyle='--', linewidth=2, label='p50')
ax.plot(exceedances.index, exceedances['adjusted_gen_v2'], 
        color='blue', linewidth=2, label='CALFEWS Forecast')
ax.plot(eia.index, eia['CVP_Gen'], 
        color='red', linewidth=2, label='EIA Gen')
october_dates = exceedances.index[exceedances.index.month == 10]
for oct_date in october_dates:
    ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Month', fontsize=24)
ax.set_ylabel('Hydropower (GWh)', fontsize=24)
ax.set_ylim(0, exceedances['p90'].max() * 1.1)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.grid(True, alpha=0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
           ncol=4, fontsize=24, frameon=True)
plt.tight_layout()
plt.show() 

