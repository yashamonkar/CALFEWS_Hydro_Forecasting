# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 09:52:17 2025

@author: amonkar

Code to run the baseline forecast using only Oct-1st storage and cumulative FNF

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
from datetime import datetime
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec

from sklearn.linear_model import LinearRegression #Regression
from sklearn.metrics import r2_score #Regression
import seaborn as sns #Regression
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import t

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


#Read the Sacramento Water Year Types
wy_types = pd.read_csv('Annual_Ensembles/cdec-water-year-type.csv', index_col=0)


#%% Generation based on Water Year Type. 
 

def get_water_year(date):
    """
    Convert calendar date to water year.
    Water year runs from Oct 1 to Sep 30.
    """
    return date.to_series().apply(lambda x: x.year + 1 if x.month >= 10 else x.year)

eia['WY'] = get_water_year(eia.index)




#%% Initial Storage Conditions
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
input_fnf = input_fnf[['SHA_fnf', 'FOL_fnf','NML_fnf']]
input_fnf['Total_Inflow'] = input_fnf.sum(axis=1)/1000
input_fnf = input_fnf.drop(['SHA_fnf', 'FOL_fnf','NML_fnf'], axis=1)
input_fnf = input_fnf.resample('ME').sum()
input_fnf['WY'] = get_water_year(input_fnf.index)
input_fnf = input_fnf[input_fnf['WY'] > 2003]
input_fnf = input_fnf.groupby('WY')['Total_Inflow'].cumsum() #Cumulative Storage





# %% Linear Regression

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
#X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, np.arange(len(X)), test_size=0.7)

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

# Get predictions for all data points
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
eia['Predicted'] = y_pred
y_pred = 0


# %% Version I --- Baseline Forecast -- Entire Period




###-----------Time Series of the Monthly Time Step--------------------------###
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(eia.index, eia['Predicted'], 
        color='blue', linewidth=2, label='Statistical Forecast')
ax.plot(eia.index, eia['CVP_Gen'], 
        color='red', linewidth=2, label='EIA')
october_dates = eia.index[eia.index.month == 10]
for oct_date in october_dates:
    ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Month', fontsize=20)
#ax.set_xlim(pd.Timestamp('2012-10-01'), exceedances.index.max())
ax.set_ylabel('Monthly Hydropower (GWh)', fontsize=20)
ax.set_ylim(0, 900)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show() 

#Annual Baseline Comp
annual_eia = eia.groupby('WY')['CVP_Gen'].sum()
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
ax.set_xlabel('Annual CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Baseline Forecast (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(1500,7500)
ax.set_ylim(1500,7500)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#----Scatter plot of the monthly time step ------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(eia['CVP_Gen'], eia['Predicted'], 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(eia['CVP_Gen'], eia['Predicted'])
line_x = np.linspace(eia['CVP_Gen'], eia['Predicted'], 100)
line_y = slope * line_x + intercept
min_val = min(eia['CVP_Gen'].min(), eia['Predicted'].min())
max_val = max(eia['CVP_Gen'].max(), eia['Predicted'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((eia["CVP_Gen"] - eia["Predicted"])**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Monthly CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Monthly Baseline Generation (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,900)
ax.set_ylim(0,900)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()





# %% Version I --- Baseline Forecast -- Testing Period

eia = eia[eia['WY'] > 2013]


###-----------Time Series of the Monthly Time Step--------------------------###
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(eia.index, eia['Predicted'], 
        color='blue', linewidth=2, label='Statistical Forecast')
ax.plot(eia.index, eia['CVP_Gen'], 
        color='red', linewidth=2, label='EIA')
october_dates = eia.index[eia.index.month == 10]
for oct_date in october_dates:
    ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Month', fontsize=20)
#ax.set_xlim(pd.Timestamp('2012-10-01'), exceedances.index.max())
ax.set_ylabel('Monthly Hydropower (GWh)', fontsize=20)
ax.set_ylim(0, 900)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show() 

#Annual Baseline Comp
annual_eia = eia.groupby('WY')['CVP_Gen'].sum()
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
ax.set_xlabel('Annual CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Baseline Forecast (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(1500,7500)
ax.set_ylim(1500,7500)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#----Scatter plot of the monthly time step ------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(eia['CVP_Gen'], eia['Predicted'], 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(eia['CVP_Gen'], eia['Predicted'])
line_x = np.linspace(eia['CVP_Gen'], eia['Predicted'], 100)
line_y = slope * line_x + intercept
min_val = min(eia['CVP_Gen'].min(), eia['Predicted'].min())
max_val = max(eia['CVP_Gen'].max(), eia['Predicted'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((eia["CVP_Gen"] - eia["Predicted"])**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.set_xlabel('Monthly CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Monthly Baseline Generation (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,900)
ax.set_ylim(0,900)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



