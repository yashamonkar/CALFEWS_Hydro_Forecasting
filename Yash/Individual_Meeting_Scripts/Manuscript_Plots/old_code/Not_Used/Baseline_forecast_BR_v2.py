# -*- coding: utf-8 -*-
"""
Created on Sun May 18 16:36:16 2025

@author: amonkar

Code to run the baseline forecast using only Oct-1st storage and regression -- with incoming FNF
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
import shutil
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec

from sklearn.linear_model import LinearRegression #Regression
from sklearn.metrics import r2_score #Regression
import seaborn as sns #Regression
from sklearn.model_selection import train_test_split


#Hyper-parameters
cfs_tafd = 2.29568411*10**-5 * 86400 / 1000


# %% Read the input data files
input_data = pd.read_csv("calfews_src/data/input/annual_runs/cord-sim_realtime.csv", index_col=0)
input_data.index = pd.to_datetime(input_data.index)

#Read the WAPA generation dataset
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)
eia = eia/1000
eia.index = pd.to_datetime(eia.index)
#eia = eia.drop(['W R Gianelli', 'ONeill'], axis=1)
eia['CVP_Gen'] = eia.sum(axis=1)
eia = eia[eia.index < pd.Timestamp("2023-10-01")]
eia = eia[eia.index > pd.Timestamp("2003-09-01")]


#%% Generation based on Water Year Type. 
 

def get_water_year(date_index):
    """
    Calculate the water year for a pandas DatetimeIndex or Series.
        
    Args:
        date_index: A pandas DatetimeIndex or Series of dates
        
    Returns:
        Series: The water year for each date
    """
    # Extract year and month components
    years = date_index.year
    months = date_index.month
    
    # Calculate water year: if month >= 10, add 1 to the year
    water_years = years.copy()
    water_years = np.where(months >= 10, years + 1, years)
    
    return water_years

eia['WY'] = get_water_year(eia.index)
annual_cvp_gen = eia.groupby('WY')['CVP_Gen'].sum()



#%% Initial Storage Conditions
input_storage = input_data.filter(regex='_storage$')
input_storage = input_storage[['SHA_storage', 'TRT_storage', 'FOL_storage','NML_storage', 'SL_storage']]
input_storage['Total_Storage'] = input_storage.sum(axis=1)/1000
input_storage['WY'] = get_water_year(input_storage.index)
initial_storage = input_storage[ (input_storage.index.month == 10) & (input_storage.index.day == 1)]
initial_storage = initial_storage[['Total_Storage', 'WY']]
initial_storage = initial_storage[initial_storage['WY'] > 2003]
initial_storage = initial_storage.groupby('WY')['Total_Storage'].sum()


# Incoming FNF
input_fnf = input_data.filter(regex='_fnf$')
input_fnf = input_fnf[['SHA_fnf', 'TRT_fnf', 'FOL_fnf','NML_fnf']]
input_fnf['Total_FNF'] = input_fnf.sum(axis=1)/1000
input_fnf['WY'] = get_water_year(input_fnf.index)
input_fnf = input_fnf[['Total_FNF', 'WY']]
input_fnf = input_fnf[input_fnf['WY'] > 2003]
input_fnf = input_fnf.groupby('WY')['Total_FNF'].sum()


# %% Linear Regression

data = pd.DataFrame({
    'Initial_Storage': initial_storage,
    'Annual_CVP_Gen': annual_cvp_gen,
    'Input_FNF': input_fnf
})

# Extract X and y for the regression model
X = data[['Initial_Storage', 'Input_FNF']]  # Reshape for sklearn
y = data['Annual_CVP_Gen'].values


# Split the data into train and test sets randomly
#X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
#    X, y, np.arange(len(X)), test_size=0.33, random_state=10)

#Block splitting
split_point = int(len(X) * 0.5)  # First 67% for training
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

# Create the plot
plt.figure(figsize=(15, 8))
sns.set_style('whitegrid')
sns.scatterplot(x='Initial_Storage', y='Annual_CVP_Gen', data=data, s=100, alpha=0.7, 
                color='blue', label='Observed Data')
#sns.lineplot(x=X.flatten(), y=y_pred, color='red', linewidth=2, 
#             label=f'Regression Line: y = {slope:.4f}x + {intercept:.4f}')
sns.lineplot(x=X['Input_FNF'], y=y_pred, color='red', linewidth=2, linestyle='dashed',
             label=f'Fitted Regression Line')
plt.text(0.06, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes, 
         fontsize=20, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.xlabel('October 1st Storage (TAF)', fontsize=20)
plt.ylabel('Annual CVP Generation (GWh)', fontsize=20)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.tight_layout()
plt.legend(loc='lower right', \
           fontsize=22, ncol = 1, frameon = True)
plt.show()

# Print the regression statistics
print(f"Regression Equation: Annual_CVP_Gen = {slope:.4f} × Initial_Storage + {intercept:.4f}")
print(f"R-squared (Variance explained): {r2:.4f}")
print(f"Correlation coefficient: {np.sqrt(r2):.4f}")

#%% Compute the monthly generation fractions

eia['Month'] = eia.index.month
wy_totals = eia.groupby('WY')['CVP_Gen'].sum()
eia['CVP_Fraction'] = eia.apply(lambda row: row['CVP_Gen'] / wy_totals[row['WY']], axis=1)
eia_sub = eia[eia.index <  pd.Timestamp("2014-10-01")]
monthly_mean_fractions = eia_sub.groupby('Month')['CVP_Fraction'].mean()
water_year_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
monthly_mean_fractions = monthly_mean_fractions.reindex(water_year_order)

# %% Compute the predicted monthly values 
monthly_df = pd.DataFrame(index=range(len(y_pred)), columns=monthly_mean_fractions.index)

# Fill the DataFrame by multiplying each annual value by the monthly fractions
for i, annual_value in enumerate(y_pred):
    monthly_df.loc[i] = annual_value * monthly_mean_fractions

flattened_values = monthly_df.values.flatten(order='C')
eia['Baseline'] = flattened_values
eia['Baseline'] = pd.to_numeric(eia['Baseline'])



#%% Baseline forecast plot

eia = eia[eia.index > pd.Timestamp("2013-09-30")]


# First compute the correlation
correlation = np.corrcoef(
    eia['Baseline'].dropna(),
    eia['CVP_Gen'].dropna())[0,1]

# Create a figure with a special layout
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, 4)  # 4 columns total

# Main time series plot
ax1 = fig.add_subplot(gs[0, 0:3])  # First 3 columns
ax1.plot(eia.index, eia['CVP_Gen'], 'b-', 
         linewidth=2, label="EIA Gen")
ax1.plot(eia.index, eia['Baseline'] , 'r-', 
         linewidth=2, label="Baseline forecast")
ax1.set_ylabel("Hydropower Generation (GWh)", fontsize=24)
ax1.set_xlabel("Month", fontsize=24)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_title(f"Total CVP Gen - Baseline Forecast", 
              fontsize=28)
ax1.set_ylim(-100, 1100)
ax1.grid(True)

# Place legend in the upper right corner of the left plot
ax1.legend(loc='upper right', fontsize=18, frameon=True)

# Scatter plot
ax2 = fig.add_subplot(gs[0, 3])  # Last column
ax2.scatter(eia['CVP_Gen'].dropna(), eia['Baseline'].dropna(),  
           color='green', alpha=0.7, s=30)

# Add 1:1 line
max_val = max(eia['CVP_Gen'].max(), eia['Baseline'].max())
min_val = min(eia['CVP_Gen'].min(), eia['Baseline'].min())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

# Add correlation text to scatter plot
ax2.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')

# Set scatter plot labels and appearance
ax2.set_xlabel("EIA Gen (GWh)", fontsize=24)
ax2.set_ylabel("Baseline Gen (GWh)", fontsize=24)  # Fixed typo: Gwh → GWh
ax2.tick_params(axis='both', labelsize=18)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate MSE between actual and predicted values
mse = ((eia['Baseline'] - eia['CVP_Gen']) ** 2).mean()
print(f"Mean Squared Error: {mse:.2f}")

mae = (abs(eia['Baseline'] - eia['CVP_Gen'])).mean()
print(f"Mean Absolute Error: {mae:.2f}")