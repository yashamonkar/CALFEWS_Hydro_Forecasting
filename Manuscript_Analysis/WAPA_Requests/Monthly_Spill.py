# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:57:41 2024

@author: amonkar
"""


# %% Initial Loading

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Input
flow_data = pd.read_csv("Yash/WAPA/Daily Gen Flow.csv", index_col = 0, 
                        skiprows=2)



# %%

#Global Hyperparameters
shasta_penstock = 17600 #CFS
folsom_penstock = 6900 #CFS
cfs_to_afd = 1.983  #CFS to Acre-ft

#Initial Data Wrangling
flow_data.index = pd.to_datetime(flow_data.index)
flow_data['Shasta'] = flow_data['Shasta'].str.replace(',', '')
flow_data['Folsom'] = flow_data['Folsom'].str.replace(',', '')
flow_data['Shasta'] = pd.to_numeric(flow_data['Shasta'], errors='coerce')
flow_data['Folsom'] = pd.to_numeric(flow_data['Folsom'], errors='coerce')
flow_data = flow_data[flow_data.index > '1996-09-30']

#Count Days of Spill
shasta_spill_count = flow_data['Shasta'][flow_data['Shasta'] > shasta_penstock].count()
folsom_spill_count = flow_data['Folsom'][flow_data['Folsom'] > folsom_penstock].count()


#%% -------------------------Shasta-------------------------------------------

#Daily Time Step
shasta_daily_spill = (flow_data['Shasta']-shasta_penstock).clip(lower=0)
shasta_daily_spill = shasta_daily_spill*cfs_to_afd
shasta_daily_spill = shasta_daily_spill.resample('M').sum()/1000

#Monthly Time Step
shasta_monthly_flow = pd.DataFrame({
    'Flow': flow_data['Shasta'].resample('M').sum(),
    'DPM': flow_data['Shasta'].resample('M').count()
})
shasta_monthly_spill = (shasta_monthly_flow['Flow']-shasta_monthly_flow['DPM']*shasta_penstock).clip(lower=0)
shasta_monthly_spill = shasta_monthly_spill*cfs_to_afd/1000


# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(shasta_monthly_spill, shasta_daily_spill)
max_val = max(shasta_monthly_spill.max(), shasta_daily_spill.max())
min_val = min(shasta_monthly_spill.min(), shasta_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Shasta (1996-2024) \n'
          f'Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Monthly Spill By Monthly Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Spill By Daily Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()



#Generate the plot
plt.figure(figsize=(10, 6))
shasta_monthly_spill.plot(color='black', label='Monthly Assumption')
plt.title(f'Shasta (1996-2024) \n'
          f'Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()



#Generate the plot
plt.figure(figsize=(10, 6))
shasta_monthly_spill.plot(color='black', label='Monthly Assumption')
shasta_daily_spill.plot(color='red', label='Daily Assumption')
plt.title(f'Shasta (1996-2024) \n'
          f'Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()


Shasta_Monthly_Releases = cfs_to_afd*flow_data['Shasta'].resample('M').sum()/1000

#Generate the plot
plt.figure(figsize=(10, 6))
shasta_monthly_spill.plot(color='black', label='Monthly Assumption')
shasta_daily_spill.plot(color='red', label='Daily Assumption')
Shasta_Monthly_Releases.plot(color='blue', label='Total Releases')
plt.title(f'Shasta (1996-2024) \n'
          f'Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()



#%% -------------------------Folsom-------------------------------------------

#Daily Time Step
folsom_daily_spill = (flow_data['Folsom']-folsom_penstock).clip(lower=0)
folsom_daily_spill = folsom_daily_spill*cfs_to_afd
folsom_daily_spill = folsom_daily_spill.resample('M').sum()/1000

#Monthly Time Step
folsom_monthly_flow = pd.DataFrame({
    'Flow': flow_data['Folsom'].resample('M').sum(),
    'DPM': flow_data['Folsom'].resample('M').count()
})
folsom_monthly_spill = (folsom_monthly_flow ['Flow']-folsom_monthly_flow ['DPM']*folsom_penstock).clip(lower=0)
folsom_monthly_spill  = folsom_monthly_spill*cfs_to_afd/1000


# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(folsom_monthly_spill , folsom_daily_spill )
max_val = max(folsom_monthly_spill.max(), folsom_daily_spill.max())
min_val = min(folsom_monthly_spill.min(), folsom_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Folsom (1996-2024) \n'
          f'Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Monthly Spill By Monthly Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Spill By Daily Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()




#Generate the plot
plt.figure(figsize=(10, 6))
folsom_monthly_spill.plot(color='black', label='Monthly Assumption')
plt.title(f'Folsom (1996-2024) \n'
          f'Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()


#Generate the plot
plt.figure(figsize=(10, 6))
folsom_monthly_spill.plot(color='black', label='Monthly Assumption')
folsom_daily_spill.plot(color='red', label='Daily Assumption')
plt.title(f'Folsom (1996-2024) \n'
          f'Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()



Folsom_Monthly_Releases = cfs_to_afd*flow_data['Folsom'].resample('M').sum()/1000



#Generate the plot
plt.figure(figsize=(10, 6))
folsom_monthly_spill.plot(color='black', label='Monthly Assumption')
folsom_daily_spill.plot(color='red', label='Daily Assumption')
Folsom_Monthly_Releases.plot(color='blue', label='Total Releases')
plt.title(f'Folsom (1996-2024) \n'
          f'Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/flow_data.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()




#%% Comparision with historic

#Historic Monthly Releases and Storage
historic = pd.read_csv('Yash/WAPA/Monthly_20240325.csv', index_col=0)
historic.index = pd.to_datetime(historic.index)
historic = historic[historic.index > '1996-09-30']
historic = historic.replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce')

shasta_historic = historic['SHA-QT']*1.983/1000
folsom_historic = historic['FOL-QT']*1.983/1000

#Curtail the flow_release data
flow_data = flow_data[flow_data.index < '2024-03-01']
monthly_flow_through_gen = flow_data.resample('M').sum()*cfs_to_afd/1000 #TAF



# %% Folsom

#Read the daily data -- FOLSOM OUTFLOW
Folsom_Outflow = pd.read_excel('Yash/Misc_Data/Folsom_Outflow.xlsx')
Folsom_Outflow = Folsom_Outflow[['OBS DATE', 'VALUE']]
Folsom_Outflow['VALUE'] = pd.to_numeric(Folsom_Outflow['VALUE'].str.replace(',', ''))
Folsom_Outflow['OBS DATE'] = pd.to_datetime(Folsom_Outflow['OBS DATE'])
Folsom_Outflow.set_index('OBS DATE', inplace=True)