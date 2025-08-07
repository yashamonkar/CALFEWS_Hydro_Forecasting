# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:12:41 2025

@author: amonkar

Code to compare the annual Oct-1st forecast from CALFEWS and statistics

Statistical Forecast:
    1. Monthly
    2. Includes the Oct-1st storage
    3. Dummy parameters for the month
    4. The option only uses p10 and p90 for statistical values.
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
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LinearRegression #Regression
from sklearn.metrics import r2_score #Regression
import seaborn as sns #Regression
from sklearn.model_selection import train_test_split
from scipy.stats import t
import matplotlib.gridspec as gridspec

#Hyper-parameters
cfs_tafd = 2.29568411*10**-5 * 86400 / 1000


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)

#Read the WAPA generation dataset
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)
eia = eia/1000 #Convert to GWh
eia.index = pd.to_datetime(eia.index)
eia = eia.drop(['W R Gianelli', 'ONeill'], axis=1)
eia['CVP_Gen'] = eia.sum(axis=1)
eia = eia[eia.index < pd.Timestamp("2023-10-01")]
eia = eia[eia.index > pd.Timestamp("2003-09-01")]

#Read the Sacramento Water Year Types
wy_types = pd.read_csv('Annual_Ensembles/cdec-water-year-type.csv', index_col=0)
wy_types['WYT'][wy_types.index == 2017] = 'EW'
wy_types['WYT'][wy_types.index == 2006] = 'EW'

#%% -------------------------CALFEWS Forecast---------------------------------#

# Compute the exceedance probabilities

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


#Compute the exceedance probabilities using the known Water Year Type
exceedances = exceedances.merge(wy_types[['WYT']], left_on='WY', right_index=True, how='left') 
exceedances['WYT'] = exceedances['WYT'].str.strip()

#Adjust the values based on the WYT on April 1st. 
exceedances['month'] = exceedances.index.month
exceedances['adjusted_gen_v2'] = exceedances['p50']
apr_sep_mask = exceedances['month'].isin([4, 5, 6, 7, 8, 9])

# Critical: use p10
critical_mask = apr_sep_mask & exceedances['WYT'].isin(['C'])
exceedances.loc[critical_mask, 'adjusted_gen_v2'] = exceedances.loc[critical_mask, 'p10']

# Dry: use p25
dry_mask = apr_sep_mask & exceedances['WYT'].isin(['D'])
exceedances.loc[dry_mask, 'adjusted_gen_v2'] = exceedances.loc[dry_mask, 'p25']

# Above Normal: use p75
above_normal_mask = apr_sep_mask & exceedances['WYT'].isin(['BN'])
exceedances.loc[above_normal_mask, 'adjusted_gen_v2'] = exceedances.loc[above_normal_mask, 'p25']

# Wet: use p90
wet_mask = apr_sep_mask & exceedances['WYT'].isin(['W'])
exceedances.loc[wet_mask, 'adjusted_gen_v2'] = exceedances.loc[wet_mask, 'p90']


# Extremely Wet: use p95
wet_mask = apr_sep_mask & exceedances['WYT'].isin(['EW'])
exceedances.loc[wet_mask, 'adjusted_gen_v2'] = exceedances.loc[wet_mask, 'p95']


# Below Normal uses p50 (already set as default)

# Clean up temporary month column if desired
exceedances = exceedances.drop('month', axis=1)

#Save the result as CALFEWS forecast
eia['CALFEWS_Gen'] = exceedances['adjusted_gen_v2']

# %% -----------Regression/Statistical Ensemble Methodology-------------------#

# Initial Storage Conditions
input_storage = input_data.filter(regex='_storage$')
input_storage = input_storage[['SHA_storage', 'FOL_storage','NML_storage', 'SL_storage', 'TRT_storage']]
input_storage['Total_Storage'] = input_storage.sum(axis=1)/1000
input_storage['WY'] = get_water_year(input_storage.index)
initial_storage = input_storage[ (input_storage.index.month == 10) & (input_storage.index.day == 1)]
initial_storage = initial_storage[['Total_Storage', 'WY']]
initial_storage = initial_storage[initial_storage['WY'] > 2003]
initial_storage = np.repeat(initial_storage['Total_Storage'], 12)
initial_storage.index = eia.index

#Incoming FNF
input_fnf = input_data.filter(regex='_fnf$')
input_fnf = input_fnf[['SHA_fnf', 'FOL_fnf','NML_fnf', 'TRT_fnf']]
input_fnf['Total_Inflow'] = input_fnf.sum(axis=1)/1000
input_fnf = input_fnf.drop(['SHA_fnf', 'FOL_fnf','NML_fnf', 'TRT_fnf'], axis=1)
input_fnf = input_fnf.resample('ME').sum()
input_fnf['WY'] = get_water_year(input_fnf.index)
all_fnf = input_fnf.groupby('WY')['Total_Inflow'].cumsum() 
input_fnf = input_fnf[input_fnf['WY'] > 2003]
input_fnf = input_fnf.groupby('WY')['Total_Inflow'].cumsum() #Cumulative Storage

# Create the DataFrame
data = pd.DataFrame({
    'Initial_Storage': initial_storage.values,
    'Cumsum_fnf': input_fnf.values,
    'Month': eia.index.month,
    'CVP_Gen': eia['CVP_Gen']
})

# Convert Month to categorical and create dummy variables
month_dummies = pd.get_dummies(data['Month'], prefix='Month')

# Combine all features (drop one month dummy to avoid multicollinearity)
X_features = pd.concat([
    data[['Initial_Storage', 'Cumsum_fnf']], 
    month_dummies.iloc[:, :-1]  # Drop last month dummy
], axis=1)


X = X_features.values
y = data['CVP_Gen'].values


# Split the data into train and test sets randomly
#X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, np.arange(len(X)), test_size=0.5)

#Block splitting
split_point = int(len(X) * 0.5)  
X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]
train_idx = np.arange(split_point)
test_idx = np.arange(split_point, len(X))

# Train the model on training data only
model = LinearRegression()
model.fit(X_train, y_train)
slope = model.coef_[0]
intercept = model.intercept_

# Prediction. 
'''
For each year starting 2003, keep the initial storage (repeat 28 times)
Use the all_fnf values to complete the regression. Compute the percentiles
'''

def get_water_year(date):
    """
    Convert calendar date to water year.
    Water year runs from Oct 1 to Sep 30.
    """
    return date.to_series().apply(lambda x: x.year + 1 if x.month >= 10 else x.year)

#Initialize the empty dataframes
p10 = []
p90 = []
p50 = []
p75 = []
p25 = []
p95 = []


for year in range(2004, 2024):
    print(year)
    
    cur_storage = pd.DataFrame({'Initial_Storage': initial_storage})
    cur_storage['WY'] = get_water_year(cur_storage.index)
    cur_storage = cur_storage[cur_storage['WY'] == year]
    
    
    #Value of all predictons
    all_predictions = []

    #Start with the regressions
    for yr in range(1996, 2024):
        
        #Extract the year's cumulative fnf
        cur_fnf = pd.DataFrame({'Cumsum_fnf': all_fnf})
        cur_fnf['WY'] = get_water_year(cur_fnf.index)
        cur_fnf = cur_fnf[cur_fnf['WY'] == yr]
        
        
        #Create the new predictor matrix
        X_new = pd.concat([
            pd.Series(cur_storage['Initial_Storage'].values,index=cur_fnf.index, name='Initial_Storage'),
            cur_fnf['Cumsum_fnf'],  
            month_dummies.iloc[0:12, :-1].set_index(cur_fnf.index)
            ], axis=1)
        X_new = X_new.values
        
        #Predict hydropower generation and save the values
        y_new = model.predict(X_new)
        all_predictions.append(y_new)
    
    #Unlist all the predictions
    all_predictions = np.concatenate(all_predictions)
    all_predictions = pd.Series(all_predictions, index=all_fnf.index, name='Predictions')
    
    #Compute and save the predictios
    water_year_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    p10_sub = all_predictions.groupby(all_predictions.index.month).quantile(0.1)
    p10_sub = p10_sub.reindex(water_year_order)
    p10.append(p10_sub)

    p25_sub = all_predictions.groupby(all_predictions.index.month).quantile(0.1)
    p25_sub = p25_sub.reindex(water_year_order)
    p25.append(p25_sub)

    p50_sub = all_predictions.groupby(all_predictions.index.month).quantile(0.5)
    p50_sub = p50_sub.reindex(water_year_order)
    p50.append(p50_sub)

    p75_sub = all_predictions.groupby(all_predictions.index.month).quantile(0.9)
    p75_sub = p75_sub.reindex(water_year_order)
    p75.append(p75_sub)

    p90_sub = all_predictions.groupby(all_predictions.index.month).quantile(0.9)
    p90_sub = p90_sub.reindex(water_year_order)
    p90.append(p90_sub)
    
    p95_sub = all_predictions.groupby(all_predictions.index.month).quantile(0.9)
    p95_sub = p95_sub.reindex(water_year_order)
    p95.append(p95_sub)
   
#Unlist all the all the percentiles
p10 = np.concatenate(p10)
p25 = np.concatenate(p25)
p50 = np.concatenate(p50)
p75 = np.concatenate(p75)
p90 = np.concatenate(p90)
p95 = np.concatenate(p95)


# Set up the exceedance table
exceedances = pd.DataFrame({
    'p10': p10,
    'p25': p25,
    'p50': p50,
    'p75': p75,
    'p90': p90,
    'p95': p95
})
exceedances.index = eia.index  

#Compute the water year type
exceedances['WY'] = get_water_year(exceedances.index)
eia['WY'] = get_water_year(eia.index)

#Compute the exceedance probabilities using the known Water Year Type
exceedances = exceedances.merge(wy_types[['WYT']], left_on='WY', right_index=True, how='left') 
exceedances['WYT'] = exceedances['WYT'].str.strip()

#Adjust the values based on the WYT on April 1st. 
exceedances['month'] = exceedances.index.month
exceedances['Predicted'] = exceedances['p50']
apr_sep_mask = exceedances['month'].isin([4, 5, 6, 7, 8, 9])

# Critical: use p10
critical_mask = apr_sep_mask & exceedances['WYT'].isin(['C'])
exceedances.loc[critical_mask, 'Predicted'] = exceedances.loc[critical_mask, 'p10']

# Dry: use p25
dry_mask = apr_sep_mask & exceedances['WYT'].isin(['D'])
exceedances.loc[dry_mask, 'Predicted'] = exceedances.loc[dry_mask, 'p25']

# Above Normal: use p75
above_normal_mask = apr_sep_mask & exceedances['WYT'].isin(['AN'])
exceedances.loc[above_normal_mask, 'Predicted'] = exceedances.loc[above_normal_mask, 'p75']

# Wet: use p90
wet_mask = apr_sep_mask & exceedances['WYT'].isin(['W'])
exceedances.loc[wet_mask, 'Predicted'] = exceedances.loc[wet_mask, 'p90']

# Extremely Wet: use p90
wet_mask = apr_sep_mask & exceedances['WYT'].isin(['EW'])
exceedances.loc[wet_mask, 'Predicted'] = exceedances.loc[wet_mask, 'p95']

# Below Normal uses p50 (already set as default)

# Clean up temporary month column if desired
exceedances = exceedances.drop('month', axis=1)

#Save the predictions
eia['Stat_Gen'] = exceedances['Predicted']


# %% --------------Statistical Base Generation -------------------------------#
#Water Year Order
water_year_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Initialize the empty dataframes
p10 = input_fnf.groupby(input_fnf.index.month).quantile(0.1)
p10 = p10.reindex(water_year_order)

p25 = input_fnf.groupby(input_fnf.index.month).quantile(0.1)
p25 = p25.reindex(water_year_order)

p50 = input_fnf.groupby(input_fnf.index.month).quantile(0.5)
p50 = p50.reindex(water_year_order)

p75 = input_fnf.groupby(input_fnf.index.month).quantile(0.9)
p75 = p75.reindex(water_year_order)

p90 = input_fnf.groupby(input_fnf.index.month).quantile(0.9)
p90 = p90.reindex(water_year_order)


# Set up the exceedance table
exceedances = pd.DataFrame({
    'p10': np.tile(p10,20),
    'p50': np.tile(p25,20),
    'p90': np.tile(p50,20),
    'p75': np.tile(p75,20),
    'p25': np.tile(p90,20)
})
exceedances.index = eia.index  

#Compute the water year type
exceedances['WY'] = get_water_year(exceedances.index)

#Compute the exceedance probabilities using the known Water Year Type
exceedances = exceedances.merge(wy_types[['WYT']], left_on='WY', right_index=True, how='left') 
exceedances['WYT'] = exceedances['WYT'].str.strip()


#Adjust the values based on the WYT on April 1st. 
exceedances['month'] = exceedances.index.month
exceedances['Inflows'] = exceedances['p50']
apr_sep_mask = exceedances['month'].isin([4, 5, 6, 7, 8, 9])

# Critical: use p10
critical_mask = apr_sep_mask & exceedances['WYT'].isin(['C'])
exceedances.loc[critical_mask, 'Inflows'] = exceedances.loc[critical_mask, 'p10']

# Dry: use p25
dry_mask = apr_sep_mask & exceedances['WYT'].isin(['D'])
exceedances.loc[dry_mask, 'Inflows'] = exceedances.loc[dry_mask, 'p25']

# Above Normal: use p75
above_normal_mask = apr_sep_mask & exceedances['WYT'].isin(['AN'])
exceedances.loc[above_normal_mask, 'Inflows'] = exceedances.loc[above_normal_mask, 'p75']

# Wet: use p90
wet_mask = apr_sep_mask & exceedances['WYT'].isin(['W'])
exceedances.loc[wet_mask, 'Inflows'] = exceedances.loc[wet_mask, 'p90']

# Extremely Wet
wet_mask = apr_sep_mask & exceedances['WYT'].isin(['EW'])
exceedances.loc[wet_mask, 'Inflows'] = exceedances.loc[wet_mask, 'p90']

# Clean up temporary month column if desired
exceedances = exceedances.drop('month', axis=1)


#Create the new input predictor set
# Combine all features (drop one month dummy to avoid multicollinearity)
X_features = pd.concat([
    data[['Initial_Storage']],
    exceedances[['Inflows']],
    month_dummies.iloc[:, :-1]  # Drop last month dummy
], axis=1)


X = X_features.values

# Get predictions for all data points
y_pred = model.predict(X)

#Save the predictions
eia['Stat_Baseline_Gen'] = y_pred


#%% Monthly Time Series Model

fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(eia.index, eia['CALFEWS_Gen'], 
        color='brown', linewidth=2, label='CALFEWS')
ax.plot(eia.index, eia['CVP_Gen'], 
        color='green',  linewidth=2, label='EIA')
ax.plot(eia.index, eia['Stat_Gen'], 
        color='violet',  linewidth=2, label='Stat (Ensemble)')
october_dates = exceedances.index[exceedances.index.month == 10]
for oct_date in october_dates:
    ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Month', fontsize=20)
ax.set_ylabel('Monthly Hydropower (GWh)', fontsize=20)
ax.set_ylim(0, 900)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show() 

# Create annual data by resampling monthly data
eia_annual = eia.groupby('WY').sum()



# %%
# Create annual data by resampling monthly data

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Function to calculate R² and RMSE
def calculate_stats(x, y):
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        r_squared = r_value**2
        rmse = np.corrcoef(x_clean, y_clean)**2
        return r_squared, rmse, slope, intercept
    return 0, 0, 0, 0

# Split data into training (first half) and testing (second half)
def split_data(data):
    mid_point = len(data) // 2
    return data.iloc[:mid_point], data.iloc[mid_point:]

# Top-left: Monthly CVP vs CALFEWS
x1 = eia['CVP_Gen']
y1 = eia['CALFEWS_Gen']

# Split into training and testing
x1_train, x1_test = split_data(x1)
y1_train, y1_test = split_data(y1)

# Calculate stats for entire period and testing period
r2_1_all, rmse_1_all, _, _ = calculate_stats(x1, y1)
r2_1_test, rmse_1_test, _, _ = calculate_stats(x1_test, y1_test)

# Plot training points (smaller, more transparent)
axes[0, 0].scatter(x1_train, y1_train, alpha=0.75, color='blue', s=20, label='Training')
# Plot testing points (larger, more opaque)
axes[0, 0].scatter(x1_test, y1_test, alpha=1.0, color='darkblue', s=50, label='Testing')

# Add 1:1 line
min_val = min(x1.min(), y1.min())
max_val = max(x1.max(), y1.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
axes[0, 0].set_xlabel('Monthly CVP Generation (GWh)', fontsize=12)
axes[0, 0].set_ylabel('Monthly CALFEWS Generation (GWh)', fontsize=12)
axes[0, 0].set_title('Monthly: CVP vs CALFEWS', fontsize=14)
axes[0, 0].text(0.05, 0.95, f'R² (All) = {r2_1_all:.2f}\nR² (Test) = {r2_1_test:.2f}', 
                transform=axes[0, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc='lower right', fontsize=10)

# Top-right: Monthly CVP vs Statistical
x2 = eia['CVP_Gen']
y2 = eia['Stat_Gen']

# Split into training and testing
x2_train, x2_test = split_data(x2)
y2_train, y2_test = split_data(y2)

# Calculate stats for entire period and testing period
r2_2_all, rmse_2_all, _, _ = calculate_stats(x2, y2)
r2_2_test, rmse_2_test, _, _ = calculate_stats(x2_test, y2_test)

# Plot training points (smaller, more transparent)
axes[0, 1].scatter(x2_train, y2_train, alpha=0.75, color='red', s=20, label='Training')
# Plot testing points (larger, more opaque)
axes[0, 1].scatter(x2_test, y2_test, alpha=1.0, color='darkred', s=50, label='Testing')

min_val = min(x2.min(), y2.min())
max_val = max(x2.max(), y2.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
axes[0, 1].set_xlabel('Monthly CVP Generation (GWh)', fontsize=12)
axes[0, 1].set_ylabel('Monthly Statistical Generation (GWh)', fontsize=12)
axes[0, 1].set_title('Monthly: CVP vs Statistical', fontsize=14)
axes[0, 1].text(0.05, 0.95, f'R² (All) = {r2_2_all:.2f}\nR² (Test) = {r2_2_test:.2f}', 
                transform=axes[0, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(loc='lower right', fontsize=10)

# Bottom-left: Annual CVP vs CALFEWS
x3 = eia_annual['CVP_Gen']
y3 = eia_annual['CALFEWS_Gen']

# Split into training and testing
x3_train, x3_test = split_data(x3)
y3_train, y3_test = split_data(y3)

# Calculate stats for entire period and testing period
r2_3_all, rmse_3_all, _, _ = calculate_stats(x3, y3)
r2_3_test, rmse_3_test, _, _ = calculate_stats(x3_test, y3_test)

# Plot training points (smaller, more transparent)
axes[1, 0].scatter(x3_train, y3_train, alpha=0.75, color='blue', s=35, label='Training')
# Plot testing points (larger, more opaque)
axes[1, 0].scatter(x3_test, y3_test, alpha=1.0, color='darkblue', s=70, label='Testing')

min_val = min(x3.min(), y3.min())
max_val = max(x3.max(), y3.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
axes[1, 0].set_xlabel('Annual CVP Generation (GWh)', fontsize=12)
axes[1, 0].set_ylabel('Annual CALFEWS Generation (GWh)', fontsize=12)
axes[1, 0].set_title('Annual: CVP vs CALFEWS', fontsize=14)
axes[1, 0].text(0.05, 0.95, f'R² (All) = {r2_3_all:.2f}\nR² (Test) = {r2_3_test:.2f}', 
                transform=axes[1, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(loc='lower right', fontsize=10)

# Bottom-right: Annual CVP vs Statistical
x4 = eia_annual['CVP_Gen']
y4 = eia_annual['Stat_Gen']

# Split into training and testing
x4_train, x4_test = split_data(x4)
y4_train, y4_test = split_data(y4)

# Calculate stats for entire period and testing period
r2_4_all, rmse_4_all, _, _ = calculate_stats(x4, y4)
r2_4_test, rmse_4_test, _, _ = calculate_stats(x4_test, y4_test)

# Plot training points (smaller, more transparent)
axes[1, 1].scatter(x4_train, y4_train, alpha=0.75, color='red', s=35, label='Training')
# Plot testing points (larger, more opaque)
axes[1, 1].scatter(x4_test, y4_test, alpha=1.0, color='darkred', s=70, label='Testing')

min_val = min(x4.min(), y4.min())
max_val = max(x4.max(), y4.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
axes[1, 1].set_xlabel('Annual CVP Generation (GWh)', fontsize=12)
axes[1, 1].set_ylabel('Annual Statistical Generation (GWh)', fontsize=12)
axes[1, 1].set_title('Annual: CVP vs Statistical', fontsize=14)
axes[1, 1].text(0.05, 0.95, f'R² (All) = {r2_4_all:.2f}\nR² (Test) = {r2_4_test:.2f}', 
                transform=axes[1, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.show()

# %% With the shaded seasons


# Create 4x3 subplot for monthly and cumulative analysis
fig, axes = plt.subplots(4, 3, figsize=(20, 20))
# Define water year months in order (Oct-Sep)
months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
month_numbers = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Define seasonal colors (light/pastel colors for backgrounds)
seasonal_colors = {
    0: '#FFF0E6',  # Fall - light orange
    1: '#B3D9FF',  # Precip - slightly darker blue-green
    2: '#F0F8FF',  # Snowmelt - lighter shade of blue
    3: '#FFE5B4'   # Summer - light peach
}

seasonal_labels = {
    0: 'Fall',
    1: 'Precip',
    2: 'Snowmelt', 
    3: 'Summer'
}

# Function to calculate R² and RMSE
def calculate_stats(x, y):
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        r_squared = r_value**2
        rmse = np.sqrt(np.mean((x_clean - y_clean)**2))
        return r_squared, rmse
    return 0, 0

# Create a copy of eia data and add month column
eia_monthly = eia.copy()
eia_monthly['Month'] = eia_monthly.index.month

# Group by water year and month for easier processing
eia_grouped = eia_monthly.groupby(['WY', 'Month']).agg({
    'CVP_Gen': 'sum',
    'CALFEWS_Gen': 'sum'
}).reset_index()

# For each subplot (month)
for i, (month, month_num) in enumerate(zip(months, month_numbers)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # Set background color for the subplot
    ax.set_facecolor(seasonal_colors[row])
    
    # For the first month (Oct), just show that month's data
    if i == 0:
        month_data = eia_grouped[eia_grouped['Month'] == month_num]
        x_data = month_data['CVP_Gen']
        y_data = month_data['CALFEWS_Gen']
        title_suffix = f'{month}'
    else:
        # For subsequent months, show cumulative from Oct to current month
        cumulative_data = []
        
        for wy in eia_grouped['WY'].unique():
            wy_data = eia_grouped[eia_grouped['WY'] == wy]
            
            # Get months from Oct (10) to current month, handling year boundary
            if month_num >= 10:  # Oct, Nov, Dec
                months_to_include = list(range(10, month_num + 1))
            else:  # Jan-Sep (next calendar year)
                months_to_include = list(range(10, 13)) + list(range(1, month_num + 1))
            
            cum_cvp = wy_data[wy_data['Month'].isin(months_to_include)]['CVP_Gen'].sum()
            cum_calfews = wy_data[wy_data['Month'].isin(months_to_include)]['CALFEWS_Gen'].sum()
            
            if cum_cvp > 0 and cum_calfews > 0:  # Only include if both have data
                cumulative_data.append([wy, cum_cvp, cum_calfews])
        
        if cumulative_data:
            cum_df = pd.DataFrame(cumulative_data, columns=['WY', 'CVP_Gen', 'CALFEWS_Gen'])
            x_data = cum_df['CVP_Gen']
            y_data = cum_df['CALFEWS_Gen']
        else:
            x_data = pd.Series([])
            y_data = pd.Series([])
    
    # Create scatter plot
    if len(x_data) > 0:
        ax.scatter(x_data, y_data, alpha=0.8, color='blue', s=100, zorder=3)
        
        # Add 1:1 line
        min_val = min(x_data.min(), y_data.min())
        max_val = max(x_data.max(), y_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.7, zorder=2)
        
        # Calculate and display statistics
        r2, rmse = calculate_stats(x_data, y_data)
        ax.text(0.05, 0.95, f'{month}\nR² = {r2:.2f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=24,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=4)
        
    
    # Formatting
    ax.grid(True, alpha=0.3, zorder=1)
    ax.tick_params(axis='both', labelsize=16)
    
    # Set axis labels only for edge subplots
    if row == 3:  # Bottom row
        ax.set_xlabel('EIA Gen (GWh)', fontsize=24)
    else:
        ax.set_xlabel(' ', fontsize=24)
    if col == 0:  # Left column
        ax.set_ylabel('CALFEWS Gen (GWh)', fontsize=24)
    else:
        ax.set_ylabel(' ', fontsize=24)

# Create legend
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=seasonal_colors[i], 
                                edgecolor='black', alpha=0.5, label=seasonal_labels[i]) 
                  for i in range(4)]

# Add legend at the bottom center
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
          ncol=4, fontsize=28, frameon=True, fancybox=True, shadow=True)

# Adjust layout to make room for legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()



# %%
# Create annual data by resampling monthly data

eia = eia[eia['WY'] > 2013]
eia_annual = eia_annual[eia_annual.index > 2013]
#eia_annual = eia_annual[eia_annual.index != 2017]
#eia_annual['CALFEWS_Gen'].iloc[3] = 4726.195429

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Function to calculate R² and RMSE
def calculate_stats(x, y):
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        r_squared = r_value**2
        corr_coef = np.corrcoef(x_clean, y_clean)**2
        r_squared = corr_coef[0,1]
        rmse = np.sqrt(np.mean((x_clean - y_clean)**2))
        return r_squared, rmse, slope, intercept
    return 0, 0, 0, 0

# Top-left: Monthly CVP vs CALFEWS
x1 = eia['CVP_Gen']
y1 = eia['CALFEWS_Gen']
r2_1, rmse_1, slope_1, intercept_1 = calculate_stats(x1, y1)

axes[0, 0].scatter(x1, y1, alpha=0.6, color='blue', s=50, label='CALFEWS')
# Add 1:1 line
min_val = min(x1.min(), y1.min())
max_val = max(x1.max(), y1.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
axes[0, 0].set_xlabel(' ', fontsize=24)
axes[0, 0].set_ylabel('Forecast Generation (GWh)', fontsize=24)
axes[0, 0].text(0.05, 0.95, f'R² = {r2_1:.2f}\nRMSE = {rmse_1:.0f} GWh', fontsize=20,
                transform=axes[0, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white'))
axes[0, 0].tick_params(axis='both', labelsize=18)
axes[0, 0].grid(True, alpha=0.3)

# Top-right: Monthly CVP vs Statistical
x2 = eia['CVP_Gen']
y2 = eia['Stat_Gen']
r2_2, rmse_2, slope_2, intercept_2 = calculate_stats(x2, y2)

axes[0, 1].scatter(x2, y2, alpha=0.6, color='red', s=50, label='Statistical')
min_val = min(x2.min(), y2.min())
max_val = max(x2.max(), y2.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
axes[0, 1].set_xlabel(' ', fontsize=24)
axes[0, 1].set_ylabel(' ', fontsize=24)
axes[0, 1].text(0.05, 0.95, f'R² = {r2_2:.2f}\nRMSE = {rmse_2:.0f} GWh', fontsize=20,
                transform=axes[0, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='both', labelsize=18)

# Bottom-left: Annual CVP vs CALFEWS
x3 = eia_annual['CVP_Gen']
y3 = eia_annual['CALFEWS_Gen']
r2_3, rmse_3, slope_3, intercept_3 = calculate_stats(x3, y3)

axes[1, 0].scatter(x3, y3, alpha=0.7, color='blue', s=100, label='CALFEWS')
for i, year in enumerate(y3.index):
    axes[1, 0].annotate(str(year), 
               (x3.iloc[i], y3.iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=14)
min_val = min(x3.min(), y3.min())
max_val = max(x3.max(), y3.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
axes[1, 0].set_xlabel('EIA Generation (GWh)', fontsize=24)
axes[1, 0].set_ylabel('Forecast Generation (GWh)', fontsize=24)
axes[1, 0].text(0.5, 0.15, f'R² = {r2_3:.2f}\nRMSE = {rmse_3:.0f} GWh', 
                transform=axes[1, 0].transAxes, verticalalignment='top', fontsize=20,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 0].tick_params(axis='both', labelsize=18)
axes[1, 0].grid(True, alpha=0.3)

# Bottom-right: Annual CVP vs Statistical
x4 = eia_annual['CVP_Gen']
y4 = eia_annual['Stat_Gen']
r2_4, rmse_4, slope_4, intercept_4 = calculate_stats(x4, y4)

axes[1, 1].scatter(x4, y4, alpha=0.7, color='red', s=100, label='Statistical')
for i, year in enumerate(y4.index):
    axes[1, 1].annotate(str(year), 
               (x4.iloc[i], y4.iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=14)
min_val = min(x4.min(), y4.min())
max_val = max(x4.max(), y4.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
axes[1, 1].set_xlabel('EIA Generation (GWh)', fontsize=24)
axes[1, 1].set_ylabel(' ', fontsize=24)
axes[1, 1].text(0.5, 0.15, f'R² = {r2_4:.2f}\nRMSE = {rmse_4:.0f} GWh', fontsize=20,
                transform=axes[1, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 1].tick_params(axis='both', labelsize=18)
axes[1, 1].grid(True, alpha=0.3)

# Create a single legend at the bottom of the figure
handles, labels = [], []
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Remove duplicates while preserving order
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

fig.legend(unique_handles, unique_labels, 
          loc='lower center', 
          bbox_to_anchor=(0.5, -0.02), 
          ncol=2,
          markerscale=2,
          fontsize=24)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for the legend
plt.show()




#%%

# Create the figure and GridSpec layout
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1])

# Top row
ax1 = fig.add_subplot(gs[0, 0:2])  # Top left: CALFEWS
ax2 = fig.add_subplot(gs[0, 2:4])  # Top right: Statistical (Baseline)
# Bottom row
ax3 = fig.add_subplot(gs[1, 1:3])  # Bottom center: Statistical (Ensemble)

# Function remains the same
def calculate_stats(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) > 0:
        r_squared = np.corrcoef(x_clean, y_clean)[0, 1] ** 2
        rmse = np.sqrt(np.mean((x_clean - y_clean) ** 2))
        slope, intercept, *_ = stats.linregress(x_clean, y_clean)
        return r_squared, rmse, slope, intercept
    return 0, 0, 0, 0



# Statistical Baseline
x5 = eia_annual['CVP_Gen']
y5 = eia_annual['Stat_Baseline_Gen']
r2_5, rmse_5, *_ = calculate_stats(x5, y5)
ax1.scatter(x5, y5, alpha=0.7, color='green', s=100)
for i, year in enumerate(y5.index):
    ax1.annotate(str(year), (x5.iloc[i], y5.iloc[i]), xytext=(5, 5), textcoords='offset points', fontsize=14)
min_val, max_val = min(x5.min(), y5.min()), max(x5.max(), y5.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax1.set_xlabel('EIA  (GWh)', fontsize=24)
ax1.set_ylabel('Forecast (GWh)', fontsize=24)
ax1.text(0.5, 0.17, f'r² = {r2_5:.2f}\nRMSE = {rmse_5:.0f} GWh', transform=ax1.transAxes, verticalalignment='top', fontsize=20,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.tick_params(axis='both', labelsize=18)
ax1.grid(True, alpha=0.3)


# Bottom Center: Statistical Ensemble
x4 = eia_annual['CVP_Gen']
y4 = eia_annual['Stat_Gen']
r2_4, rmse_4, *_ = calculate_stats(x4, y4)
ax2.scatter(x4, y4, alpha=0.7, color='red', s=100)
for i, year in enumerate(y4.index):
    ax2.annotate(str(year), (x4.iloc[i], y4.iloc[i]), xytext=(5, 5), textcoords='offset points', fontsize=14)
min_val, max_val = min(x4.min(), y4.min()), max(x4.max(), y4.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax2.set_xlabel('EIA (GWh)', fontsize=24)
ax2.set_ylabel(' ', fontsize=24)
ax2.text(0.5, 0.17, f'r² = {r2_4:.2f}\nRMSE = {rmse_4:.0f} GWh', 
         transform=ax2.transAxes, verticalalignment='top', fontsize=20,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.tick_params(axis='both', labelsize=18)
ax2.grid(True, alpha=0.3)




# Top Left: CALFEWS
x3 = eia_annual['CVP_Gen']
y3 = eia_annual['CALFEWS_Gen']
r2_3, rmse_3, *_ = calculate_stats(x3, y3)
ax3.scatter(x3, y3, alpha=0.7, color='blue', s=100)
for i, year in enumerate(y3.index):
    ax3.annotate(str(year), (x3.iloc[i], y3.iloc[i]), xytext=(5, 5), textcoords='offset points', fontsize=14)
min_val, max_val = min(x3.min(), y3.min()), max(x3.max(), y3.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax3.set_xlabel('EIA (GWh)', fontsize=24)
ax3.set_ylabel('Forecast (GWh)', fontsize=24)
ax3.text(0.5, 0.17, f'r² = {r2_3:.2f}\nRMSE = {rmse_3:.0f} GWh', 
         transform=ax3.transAxes, verticalalignment='top', fontsize=20,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax3.tick_params(axis='both', labelsize=18)
ax3.grid(True, alpha=0.3)




# Legend (in ax1)
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='CALFEWS')
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Statistical (Ensemble)')
green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Statistical (Baseline)')
ax3.legend(handles=[green_patch, red_patch, blue_patch], loc='upper left', fontsize=18, markerscale=1.5)

plt.tight_layout()
plt.show()
