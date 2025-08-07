# -*- coding: utf-8 -*-
"""
Created on Sun May 18 16:36:16 2025

@author: amonkar

Code to run the baseline forecast using only Oct-1st storage and regression 
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
annual_eia = eia.groupby('WY')['CVP_Gen'].sum()



#%% Initial Storage Conditions
input_storage = input_data.filter(regex='_storage$')
input_storage = input_storage[['SHA_storage', 'FOL_storage','NML_storage', 'SL_storage']]
input_storage['Total_Storage'] = input_storage.sum(axis=1)/1000
input_storage['WY'] = get_water_year(input_storage.index)
initial_storage = input_storage[ (input_storage.index.month == 10) & (input_storage.index.day == 1)]
initial_storage = initial_storage[['Total_Storage', 'WY']]
initial_storage = initial_storage[initial_storage['WY'] > 2003]
initial_storage = initial_storage.groupby('WY')['Total_Storage'].sum()


#----Scatter plot of the annual time step ------------------------------------#

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(initial_storage.values, annual_eia.values, 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
for i, year in enumerate(initial_storage.index):
    ax.annotate(str(year), 
               (initial_storage.iloc[i], annual_eia.iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, alpha=0.8)
slope, intercept, r_value, p_value, std_err = stats.linregress(initial_storage.values, annual_eia.values)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Oct-1st Storage (TAF)', fontsize=24)
ax.set_ylabel('EIA Generation (GWh)', fontsize=24)
ax.set_title('Total CVP Gen', fontsize=28)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% Linear Regression

data = pd.DataFrame({
    'Initial_Storage': initial_storage,
    'Annual_CVP_Gen': annual_eia
})

# Extract X and y for the regression model
X = data['Initial_Storage'].values.reshape(-1, 1)  # Reshape for sklearn
y = data['Annual_CVP_Gen'].values


# Split the data into train and test sets randomly
#X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, np.arange(len(X)), test_size=0.5)

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
data['Pred'] = y_pred


# %% Regression plot with uncertainty

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(initial_storage.values, annual_eia.values, 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
for i, year in enumerate(initial_storage.index):
    ax.annotate(str(year), 
               (initial_storage.iloc[i], annual_eia.iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, alpha=0.8)

slope_stats, intercept_stats, r_value, p_value, std_err = stats.linregress(X_train.flatten(), y_train)
line_x = np.linspace(X.min(), X.max(), 100)
line_y = slope_stats * line_x + intercept_stats
ax.plot(line_x, line_y, 'r--', linewidth=2, label='Fitted Regression Line')
n = len(X_train)
dof = n - 2  # degrees of freedom
t_val = t.ppf(0.9, dof)  # 95% confidence interval
x_mean = np.mean(X_train)
sxx = np.sum((X_train - x_mean) ** 2)

# Calculate constant standard error for linear confidence bands
residuals = y_train - (slope_stats * X_train.flatten() + intercept_stats)
mse = np.sum(residuals ** 2) / (len(X_train.flatten()) - 2)
se_constant = np.sqrt(mse)  # Constant standard error

# Linear confidence intervals (constant width)
margin_of_error = t_val * se_constant
ci_upper = line_y + margin_of_error
ci_lower = line_y - margin_of_error
ax.fill_between(line_x, ci_lower, ci_upper, alpha=0.2, color='red', 
                label='80% Confidence Interval')
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
rmse = np.sqrt(np.mean((y_train - (slope_stats * X_train.flatten() + intercept_stats))**2))
ax.text(0.05, 0.9, f'RMSE = {rmse:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Oct-1st Storage (TAF)', fontsize=24)
ax.set_ylabel('EIA Generation (GWh)', fontsize=24)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()


# Compute the monthly generation fractions for the testing dataset

eia['Month'] = eia.index.month
wy_totals = eia.groupby('WY')['CVP_Gen'].sum()
eia['CVP_Fraction'] = eia.apply(lambda row: row['CVP_Gen'] / wy_totals[row['WY']], axis=1)
eia_sub = eia[eia.index <  pd.Timestamp("2014-10-01")]
monthly_mean_fractions = eia_sub.groupby('Month')['CVP_Fraction'].mean()
water_year_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
monthly_mean_fractions = monthly_mean_fractions.reindex(water_year_order)

#Add the Monthly Fraction Data
eia['CVP_Fraction'] = np.tile(monthly_mean_fractions, 20)

# %% Version I --- Baseline Forecast
monthly_df = pd.DataFrame(index=range(len(y_pred)), columns=monthly_mean_fractions.index)

# Fill the DataFrame by multiplying each annual value by the monthly fractions
for i, annual_value in enumerate(y_pred):
    monthly_df.loc[i] = annual_value * monthly_mean_fractions

flattened_values = monthly_df.values.flatten(order='C')
eia['Baseline'] = flattened_values
eia['Baseline'] = pd.to_numeric(eia['Baseline'])





###-----------Time Series of the Monthly Time Step--------------------------###
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(eia.index, eia['Baseline'], 
        color='blue', linestyle='--', linewidth=2, label='Baseline Forecast')
ax.plot(eia.index, eia['CVP_Gen'], 
        color='red', linestyle='--', linewidth=2, label='EIA')
october_dates = eia.index[eia.index.month == 10]
for oct_date in october_dates:
    ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Month', fontsize=20)
#ax.set_xlim(pd.Timestamp('2012-10-01'), exceedances.index.max())
ax.set_ylabel('Monthly Hydropower (GWh)', fontsize=20)
ax.set_title('Total CVP Generation (GWh)', fontsize=24)
ax.set_ylim(0, eia['Baseline'].max() * 1.1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show() 


#Annual Baseline Comp
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(data['Annual_CVP_Gen'].values, data['Pred'].values, 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
for i, year in enumerate(data.index):
    ax.annotate(str(year), 
               (data['Annual_CVP_Gen'].iloc[i], data['Pred'].iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, alpha=0.8)
slope, intercept, r_value, p_value, std_err = stats.linregress(data['Annual_CVP_Gen'].values, data['Pred'].values)
line_x = np.linspace(data['Annual_CVP_Gen'].min(), data['Annual_CVP_Gen'].max(), 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=2, label='Fitted Regression Line')
min_val = min(data['Annual_CVP_Gen'].min(), data['Pred'].min())
max_val = max(data['Annual_CVP_Gen'].max(), data['Pred'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((data["Annual_CVP_Gen"] - data["Pred"])**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Annual CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Baseline Forecast (GWh)', fontsize=24)
ax.set_title('Total CVP Gen', fontsize=28)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(min_val * 0.95, max_val * 1.05)
ax.set_ylim(min_val * 0.95, max_val * 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#----Scatter plot of the monthly time step ------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(eia['CVP_Gen'], eia['Baseline'], 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(eia['CVP_Gen'], eia['Baseline'])
line_x = np.linspace(eia['CVP_Gen'], eia['Baseline'], 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=2, label='Fitted Regression Line')
min_val = min(eia['CVP_Gen'].min(), eia['Baseline'].min())
max_val = max(eia['CVP_Gen'].max(), eia['Baseline'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((eia["CVP_Gen"] - eia["Baseline"])**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Monthly CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Monthly Baseline Generation (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(min_val * 0.95, max_val * 1.05)
ax.set_ylim(min_val * 0.95, max_val * 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% Version II --- Incorporate the Water Year Type in October


eia = eia.merge(wy_types[['WYT']], left_on='WY', right_index=True, how='left') 
eia['WYT'] = eia['WYT'].str.strip()

#Adjust the values based on the WYT on April 1st. 
eia['month'] = eia.index.month
eia['forecast_pct'] = 0.5 #Set the common to the median
apr_sep_mask = eia['month'].isin([4, 5, 6, 7, 8, 9])

#Critical: use p10
critical_mask = apr_sep_mask & eia['WYT'].isin(['C'])
eia.loc[critical_mask, 'forecast_pct'] = 0.1

# Dry use p25
dry_mask = apr_sep_mask & eia['WYT'].isin(['D'])
eia.loc[dry_mask, 'forecast_pct'] = 0.25

# Above Normal or Wet: use p75
above_normal_mask = apr_sep_mask & eia['WYT'].isin(['AN'])
eia.loc[above_normal_mask, 'forecast_pct'] = 0.75

#Wet: use p90
wet_mask = apr_sep_mask & eia['WYT'].isin(['W'])
eia.loc[wet_mask, 'forecast_pct'] = 0.9




#Add initial storage to the eia dataset
eia['initial_storage'] = np.repeat(initial_storage.values, 12)

# Get base predictions
base_predictions = model.predict(eia[['initial_storage']])

# Create predictions based on forecast_pct
predictions = np.zeros(len(eia))

for i, forecast_pct in enumerate(eia['forecast_pct']):
    if forecast_pct == 0.5:
        # Mean prediction
        predictions[i] = base_predictions[i]
    else:
        # Calculate confidence interval
        t_val = stats.t.ppf(forecast_pct, dof)
        margin_of_error = t_val * se_constant
        predictions[i] = base_predictions[i] + margin_of_error

# Add predictions to your dataframe
eia['predictions_v2'] = predictions*eia['CVP_Fraction'] 




###-----------Time Series of the Monthly Time Step--------------------------###
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(eia.index, eia['predictions_v2'], 
        color='blue', linestyle='--', linewidth=2, label='Updated Forecast')
ax.plot(eia.index, eia['CVP_Gen'], 
        color='red', linestyle='--', linewidth=2, label='EIA')
october_dates = eia.index[eia.index.month == 10]
for oct_date in october_dates:
    ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Month', fontsize=20)
#ax.set_xlim(pd.Timestamp('2012-10-01'), exceedances.index.max())
ax.set_ylabel('Monthly Hydropower (GWh)', fontsize=20)
ax.set_title('Total CVP Generation (GWh)', fontsize=24)
ax.set_ylim(0, eia['predictions_v2'].max() * 1.1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show() 


#Annual Baseline Comp

data['predictions_v2'] = eia.groupby('WY')['predictions_v2'].sum()

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(data['Annual_CVP_Gen'].values, data['predictions_v2'].values, 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
for i, year in enumerate(data.index):
    ax.annotate(str(year), 
               (data['Annual_CVP_Gen'].iloc[i], data['predictions_v2'].iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, alpha=0.8)
slope, intercept, r_value, p_value, std_err = stats.linregress(data['Annual_CVP_Gen'].values, data['predictions_v2'].values)
line_x = np.linspace(data['Annual_CVP_Gen'].min(), data['Annual_CVP_Gen'].max(), 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=2, label='Fitted Regression Line')
min_val = min(data['Annual_CVP_Gen'].min(), data['predictions_v2'].min())
max_val = max(data['Annual_CVP_Gen'].max(), data['predictions_v2'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((data["Annual_CVP_Gen"] - data["predictions_v2"])**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Annual CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Updated Forecast (GWh)', fontsize=24)
ax.set_title('Total CVP Gen', fontsize=28)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(min_val * 0.95, max_val * 1.05)
ax.set_ylim(min_val * 0.95, max_val * 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#----Scatter plot of the monthly time step ------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(eia['CVP_Gen'], eia['predictions_v2'], 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(eia['CVP_Gen'], eia['predictions_v2'])
line_x = np.linspace(eia['CVP_Gen'], eia['predictions_v2'], 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=2, label='Fitted Regression Line')
min_val = min(eia['CVP_Gen'].min(), eia['predictions_v2'].min())
max_val = max(eia['CVP_Gen'].max(), eia['predictions_v2'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((eia["CVP_Gen"] - eia["predictions_v2"])**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Monthly CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Updated Baseline Generation (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(min_val * 0.95, max_val * 1.05)
ax.set_ylim(min_val * 0.95, max_val * 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% Testing Period 

eia = eia[eia['WY'] > 2013]
data = data[data.index > 2013]

###-----------Time Series of the Monthly Time Step--------------------------###
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(eia.index, eia['predictions_v2'], 
        color='blue', linestyle='--', linewidth=2, label='Updated Forecast')
ax.plot(eia.index, eia['CVP_Gen'], 
        color='red', linestyle='--', linewidth=2, label='EIA')
october_dates = eia.index[eia.index.month == 10]
for oct_date in october_dates:
    ax.axvline(x=oct_date, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Month', fontsize=20)
#ax.set_xlim(pd.Timestamp('2012-10-01'), exceedances.index.max())
ax.set_ylabel('Monthly Hydropower (GWh)', fontsize=20)
ax.set_title('Total CVP Generation (GWh)', fontsize=24)
ax.set_ylim(0, 1000)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=24, frameon=True)
plt.tight_layout()
plt.show() 


#Annual Baseline Comp

data['predictions_v2'] = eia.groupby('WY')['predictions_v2'].sum()

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(data['Annual_CVP_Gen'].values, data['predictions_v2'].values, 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
for i, year in enumerate(data.index):
    ax.annotate(str(year), 
               (data['Annual_CVP_Gen'].iloc[i], data['predictions_v2'].iloc[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=14, alpha=0.8)
slope, intercept, r_value, p_value, std_err = stats.linregress(data['Annual_CVP_Gen'].values, data['predictions_v2'].values)
line_x = np.linspace(data['Annual_CVP_Gen'].min(), data['Annual_CVP_Gen'].max(), 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=1, label='Fitted Regression Line', alpha = 0)
min_val = min(data['Annual_CVP_Gen'].min(), data['predictions_v2'].min())
max_val = max(data['Annual_CVP_Gen'].max(), data['predictions_v2'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.6, 0.03, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=18,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.text(0.6, 0.1, f'RMSE = {np.sqrt(np.mean((data["Annual_CVP_Gen"] - data["predictions_v2"])**2)):.0f} GWh', 
        transform=ax.transAxes, fontsize=18,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.text(0.05, 0.95, '(B) Regression Approach', 
        transform=ax.transAxes, fontsize=18)
ax.set_xlabel('Annual Generation (GWh)', fontsize=22)
ax.set_ylabel('Forecasted Generation (GWh)', fontsize=22)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(1750, 6000)
ax.set_ylim(1750, 6000)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#----Scatter plot of the monthly time step ------------------------------------#
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(eia['CVP_Gen'], eia['predictions_v2'], 
                    color='blue', s=60, alpha=0.7, label='Observed Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(eia['CVP_Gen'], eia['predictions_v2'])
line_x = np.linspace(eia['CVP_Gen'], eia['predictions_v2'], 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=2, label='Fitted Regression Line')
min_val = min(eia['CVP_Gen'].min(), eia['predictions_v2'].min())
max_val = max(eia['CVP_Gen'].max(), eia['predictions_v2'].max())
ax.plot([min_val, max_val], [min_val, max_val], 
        'k-', linewidth=1, alpha=0.4)
ax.text(0.05, 0.95, f'R² = {r_value**2:.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(0.05, 0.9, f'RMSE = {np.sqrt(np.mean((eia["CVP_Gen"] - eia["predictions_v2"])**2)):.2f}', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Monthly CVP Generation (GWh)', fontsize=24)
ax.set_ylabel('Updated Baseline Generation (GWh)', fontsize=24)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 900)
ax.set_ylim(0, 900)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()










































