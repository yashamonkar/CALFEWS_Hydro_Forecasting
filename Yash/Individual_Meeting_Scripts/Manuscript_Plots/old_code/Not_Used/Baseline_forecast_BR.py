
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 16:36:16 2025

@author: amonkar

Code to run the baseline forecast using only Oct-1st storage and regression --- Base Resource 
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

#Read the base resource data. 
br_data = pd.read_csv("Yash/Individual_Meeting_Scripts/Manuscript_Plots/data/BR_Data.csv", index_col=0)
br_data.index = pd.to_datetime(br_data.index)
br_data = br_data[br_data.index < pd.Timestamp("2023-10-01")]
br_data = br_data[br_data.index > pd.Timestamp("2007-09-01")]
br_data.columns = [['Year', 'Month', 'BR']]
br_data = br_data.drop(['Year', 'Month'], axis=1)
br_data = br_data['BR']/1000

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

br_data['WY'] = get_water_year(br_data.index)
br_data.columns = [col[0] for col in br_data.columns]
annual_cvp_gen = br_data.groupby('WY')['BR'].sum()


#%% Initial Storage Conditions
input_storage = input_data.filter(regex='_storage$')
input_storage = input_storage[['SHA_storage', 'TRT_storage', 'FOL_storage','NML_storage', 'SL_storage']]
input_storage['Total_Storage'] = input_storage.sum(axis=1)/1000
input_storage['WY'] = get_water_year(input_storage.index)
initial_storage = input_storage[ (input_storage.index.month == 10) & (input_storage.index.day == 1)]
initial_storage = initial_storage[['Total_Storage', 'WY']]
initial_storage = initial_storage[initial_storage['WY'] > 2007]
initial_storage = initial_storage.groupby('WY')['Total_Storage'].sum()


# %% Linear Regression

data = pd.DataFrame({
    'Initial_Storage': initial_storage,
    'Annual_CVP_Gen': annual_cvp_gen
})

# Extract X and y for the regression model
X = data['Initial_Storage'].values.reshape(-1, 1)  # Reshape for sklearn
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
sns.lineplot(x=X.flatten(), y=y_pred, color='red', linewidth=2, linestyle='dashed',
             label=f'Fitted Regression Line')
plt.text(0.06, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes, 
         fontsize=20, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.xlabel('October 1st Storage (TAF)', fontsize=20)
plt.ylabel('Annual CVP Generation (GWh)', fontsize=20)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
for i, year in enumerate(data.index):
    plt.annotate(str(year), (X[i, 0], y[i]), 
                 xytext=(5, 5), textcoords='offset points', 
                 fontsize=18, alpha=0.7)
plt.tight_layout()
plt.legend(loc='lower right', \
           fontsize=22, ncol = 1, frameon = True)
plt.show()

# Print the regression statistics
print(f"Regression Equation: Annual_CVP_Gen = {slope:.4f} × Initial_Storage + {intercept:.4f}")
print(f"R-squared (Variance explained): {r2:.4f}")
print(f"Correlation coefficient: {np.sqrt(r2):.4f}")

#%% Compute the monthly generation fractions

br_data['Month'] = br_data.index.month
wy_totals = br_data.groupby('WY')['BR'].sum()
br_data['CVP_Fraction'] = br_data.apply(lambda row: row['BR'] / wy_totals[row['WY']], axis=1)
br_sub = br_data[br_data.index <  pd.Timestamp("2015-10-01")]
monthly_mean_fractions = br_sub.groupby('Month')['CVP_Fraction'].mean()
water_year_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
monthly_mean_fractions = monthly_mean_fractions.reindex(water_year_order)

# %% Compute the predicted monthly values 
monthly_df = pd.DataFrame(index=range(len(y_pred)), columns=monthly_mean_fractions.index)

# Fill the DataFrame by multiplying each annual value by the monthly fractions
for i, annual_value in enumerate(y_pred):
    monthly_df.loc[i] = annual_value * monthly_mean_fractions

flattened_values = monthly_df.values.flatten(order='C')
br_data['Baseline'] = flattened_values
br_data['Baseline'] = pd.to_numeric(br_data['Baseline'])



#%% Baseline forecast plot

# First compute the correlation
correlation = np.corrcoef(
    br_data['Baseline'].dropna(),
    br_data['BR'].dropna())[0,1]

# Create a figure with a special layout
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, 4)  # 4 columns total

# Main time series plot
ax1 = fig.add_subplot(gs[0, 0:3])  # First 3 columns
ax1.plot(br_data.index, br_data['BR'], 'b-', 
         linewidth=2, label="Historical Gen")
ax1.plot(br_data.index, br_data['Baseline'] , 'r-', 
         linewidth=2, label="Baseline forecast")
ax1.set_ylabel("Hydropower Generation (GWh)", fontsize=24)
ax1.set_xlabel("Month", fontsize=24)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_title(f"Total CVP Base Resource", 
              fontsize=28)
ax1.grid(True)

# Place legend in the upper right corner of the left plot
ax1.legend(loc='upper right', fontsize=18, frameon=True)

# Scatter plot
ax2 = fig.add_subplot(gs[0, 3])  # Last column
ax2.scatter(br_data['BR'].dropna(), br_data['Baseline'].dropna(),  
           color='green', alpha=0.7, s=30)

# Add 1:1 line
max_val = max(br_data['Baseline'] .max(), br_data['BR'].max())
min_val = min(br_data['Baseline'] .min(), br_data['BR'].min())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

# Add correlation text to scatter plot
ax2.annotate(f"r² = {correlation**2:.2f}", 
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=24, ha='left', va='top')

# Set scatter plot labels and appearance
ax2.set_xlabel("Historical Gen (GWh)", fontsize=24)
ax2.set_ylabel("CALFEWS Gen (GWh)", fontsize=24)  # Fixed typo: Gwh → GWh
ax2.tick_params(axis='both', labelsize=18)
ax2.set_title("Observed vs. Simulated", fontsize=28)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate MSE between actual and predicted values
mse = ((br_data['Baseline'] - br_data['BR']) ** 2).mean()
print(f"Mean Squared Error: {mse:.2f}")

mae = (abs(br_data['Baseline'] - br_data['BR'])).mean()
print(f"Mean Absolute Error: {mae:.2f}")