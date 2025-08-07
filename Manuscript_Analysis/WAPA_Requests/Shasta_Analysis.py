# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:47:38 2024

@author: Yash Vijay Amonkar

Code to analyze the validity of the Uniform Monthly Release Assumption. 
This code is for Shasta only. 

Input Data Sources:- 
1. Daily Releases - CDEC 
2. Penstock Capacity - 
3. Historic Water Year types - https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST

"""

# %% Initial Loading

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS\Yash\Misc_Data'
os.chdir(working_directory)

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#Daily Release Data
Shasta_Outflow = pd.read_excel('Shasta_Outflow.xlsx')
Shasta_Outflow = Shasta_Outflow[['OBS DATE', 'VALUE']]
Shasta_Outflow['VALUE'] = pd.to_numeric(Shasta_Outflow['VALUE'].str.replace(',', ''))
Shasta_Outflow['OBS DATE'] = pd.to_datetime(Shasta_Outflow['OBS DATE'])
Shasta_Outflow.set_index('OBS DATE', inplace=True)


#Historic Water Year Types 
wy_types = pd.read_csv("Water_Year_Types.csv", index_col = 0)
wy_types.index = pd.to_datetime(wy_types.index)
wy_types = wy_types[wy_types.index > '1996-09-30']
wy_types.index = wy_types.index + pd.offsets.MonthEnd(0) #Align index to the end of the month. 

#Global Hyperparameters
shasta_penstock = 17600 #CFS
cfs_to_afd = 1.983  #Unit conversion CFS to TAFD. 

# %% Analysis

### -------------------------Daily Time Step ---------------------------------#
#Spill Analysis
shasta_daily_spill = (Shasta_Outflow['VALUE']-shasta_penstock).clip(lower=0)
shasta_daily_spill = shasta_daily_spill*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day
shasta_daily_spill = shasta_daily_spill.resample('M').sum() #Aggregate to monthly

#Flow through the penstock
shasta_daily_penstock = Shasta_Outflow['VALUE'].clip(upper=shasta_penstock)
shasta_daily_penstock = shasta_daily_penstock*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day
shasta_daily_penstock = shasta_daily_penstock.resample('M').sum() #Aggregate to monthly

### -------------------------Monthly Time Step ---------------------------------#
#Aggregate total releases to the monthly time-step
shasta_monthly_flow = pd.DataFrame({
    'Flow': Shasta_Outflow['VALUE'].resample('M').sum(), #Total Monthly Releases
    'DPM': Shasta_Outflow['VALUE'].resample('M').count() #Days per month
})

#Spill Analysis
shasta_monthly_spill = (shasta_monthly_flow['Flow']-shasta_monthly_flow['DPM']*shasta_penstock).clip(lower=0)
shasta_monthly_spill = shasta_monthly_spill*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day

#Flow through the penstock
shasta_monthly_penstock = shasta_monthly_flow['Flow'].clip(upper = shasta_monthly_flow['DPM']*shasta_penstock)
shasta_monthly_penstock = shasta_monthly_penstock*cfs_to_afd/1000 #Convert to Thousand Acre-feet/day


# %% Metric Calculations 

#------------------Metric I -- Underestimation of spill-----------------------#
spill = 100*(sum(shasta_daily_spill) - sum(shasta_monthly_spill))/sum(shasta_monthly_spill)
f'Underestimation of spill for Shasta is {round(spill,2)}%'

#--------------Metric II -- Overestimation of penstock flows------------------#
overestimate = 100*(sum(shasta_monthly_penstock) - sum(shasta_daily_penstock))/sum(shasta_daily_penstock)
f'Over-estimation of penstock flow for Shasta is {round(overestimate,2)}%'


#-------------Monthly Distribution of penstock flow overestimation------------#
penstock_overestimate = pd.DataFrame({
    'Daily': shasta_daily_penstock,
    'Monthly': shasta_monthly_penstock, 
    'WYT': wy_types['SAC_Index']
})

#Extract the month from the date. 
penstock_overestimate['Month'] = pd.to_datetime(penstock_overestimate.index)
penstock_overestimate['Month'] = penstock_overestimate['Month'].dt.month

#Penstock overestimates in Jan-Apr
filtered_penstock_overestimate = penstock_overestimate[penstock_overestimate['Month'].isin([1, 2, 3, 4])]
overestimate = 100*(filtered_penstock_overestimate['Monthly'].sum() - filtered_penstock_overestimate['Daily'].sum()) / filtered_penstock_overestimate['Daily'].sum()
f'Over-estimation of penstock flow for Shasta during Jan-Apr is {round(overestimate,2)}%'


#Function to calculate overestimate value
def calculate_penstock_overestimate(group):
    daily_sum = group['Daily'].sum()
    monthly_sum = group['Monthly'].sum()
    return round(100 * (monthly_sum - daily_sum) / daily_sum,2)


# Calculate for all WYTs (by Month)
all_wyt_by_month = penstock_overestimate.groupby('Month').apply(calculate_penstock_overestimate).reset_index(name='Calculation')
all_wyt_by_month['WYT'] = 'All Types'

# Calculate for all Months (by WYT)
all_months_by_wyt = penstock_overestimate.groupby('WYT').apply(calculate_penstock_overestimate).reset_index(name='Calculation')
all_months_by_wyt['Month'] = 'All Months'

# Calculate for all WYTs and all Months
all_wyt_all_months = pd.DataFrame({
    'WYT': ['All Types'],
    'Month': ['All Months'],
    'Calculation': [calculate_penstock_overestimate(penstock_overestimate)]
})

#Individual Months and Water Year Tyopes. 
months_by_wyt = penstock_overestimate.groupby(['WYT', 'Month']).apply(calculate_penstock_overestimate).reset_index(name='Calculation')




# %% Visual Analysis


# Spill Analysis Scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(shasta_daily_spill, shasta_monthly_spill)
max_val = max(shasta_monthly_spill.max(), shasta_daily_spill.max())
min_val = min(shasta_monthly_spill.min(), shasta_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Spill \n Shasta (WY 1997 - WY 2023) ', fontsize=20)
plt.xlabel('Daily Time Step Model  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Model \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


# Penstock Flow Scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(shasta_daily_penstock, shasta_monthly_penstock)
max_val = max(shasta_monthly_penstock.max(), shasta_daily_penstock.max())
min_val = min(shasta_monthly_penstock.min(), shasta_daily_penstock.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Penstock Flows \n Shasta (WY 1997 - WY 2023) ', fontsize=20)
plt.xlabel('Daily Time Step Model  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Model \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


#Monthly Counts 
spill_difference = shasta_daily_spill - shasta_monthly_spill
non_zero_spill = spill_difference[spill_difference != 0]
non_zero_spill = non_zero_spill.reset_index()
non_zero_spill['Month'] = non_zero_spill['OBS DATE'].dt.strftime('%b')  # Extract month as a string
monthly_non_zero_count = non_zero_spill.groupby('Month').count()
monthly_non_zero_count = monthly_non_zero_count.reindex(['Jan', 'Feb', 'Mar', 
                                                         'Apr', 'May', 'Jun', 
                                                         'Jul', 'Aug', 'Sep', 
                                                         'Oct', 'Nov', 'Dec'])

monthly_non_zero_count['OBS DATE'].plot(kind='bar', figsize=(10, 6))
plt.xlabel('Month', fontsize = 16)
plt.ylabel('Months with Spill Under-estimation \n Count Data', fontsize = 16)
plt.title(f'Shasta (WY 1997 - WY 2023)', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()



# %% 2017 Jan - April Visual Analysis

#Subset to needed timeframe
Shasta_Example = Shasta_Outflow*cfs_to_afd/1000 #Convert to TAF
Shasta_Example = Shasta_Example[Shasta_Example.index > '2016-12-31']
Shasta_Example = Shasta_Example[Shasta_Example.index < '2017-05-01']

#Compute the Uniform Monthly Release Assumption Values
monthly_avg = Shasta_Example.resample('MS')['VALUE'].mean()
monthly_avg_daily = monthly_avg.reindex(Shasta_Example.index, method='ffill')
Shasta_Example['Monthly_Avg'] = monthly_avg_daily


plt.figure(figsize=(10, 6))
plt.fill_between(Shasta_Example.index, Shasta_Example['Monthly_Avg'], 0, color='yellow', alpha=0.15)
plt.plot(Shasta_Example.index, Shasta_Example['Monthly_Avg'], linewidth=2, color='orange', label='Monthly Time Step')
plt.fill_between(Shasta_Example.index, Shasta_Example['VALUE'], 0, color='blue', alpha=0.3)
plt.plot(Shasta_Example.index, Shasta_Example['VALUE'], linewidth=2, color='blue' , label='Daily Time Step')
plt.title('Shasta Releases in 2017 (Wet Water Year)', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Daily Releases (TAF)', fontsize=16)
for month in pd.date_range(start='2017-01-01', end='2017-05-01', freq='MS'):
    plt.axvline(x=month, color='black', linewidth=1.15)
plt.axhline(y=shasta_penstock*cfs_to_afd/1000, color='red', linestyle='--', linewidth=2, label='Penstock Capacity')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=16)
plt.tight_layout() 
plt.show()

