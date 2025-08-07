# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:51:20 2024

@author: amonkar
"""

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
generator_flow = pd.read_csv("Yash/WAPA/Daily Gen Flow.csv", index_col = 0, 
                        skiprows=2)
generator_flow.index = pd.to_datetime(generator_flow.index)
generator_flow['Shasta'] = generator_flow['Shasta'].str.replace(',', '')
generator_flow['Folsom'] = generator_flow['Folsom'].str.replace(',', '')
generator_flow['Shasta'] = pd.to_numeric(generator_flow['Shasta'], errors='coerce')
generator_flow['Folsom'] = pd.to_numeric(generator_flow['Folsom'], errors='coerce')
generator_flow = generator_flow[generator_flow.index > '1996-09-30']
generator_flow = generator_flow[generator_flow.index < '2023-10-01']


#Daily Data -- Folsom
Folsom_Outflow = pd.read_excel('Yash/Misc_Data/Folsom_Outflow.xlsx')
Folsom_Outflow = Folsom_Outflow[['OBS DATE', 'VALUE']]
Folsom_Outflow['VALUE'] = pd.to_numeric(Folsom_Outflow['VALUE'].str.replace(',', ''))
Folsom_Outflow['OBS DATE'] = pd.to_datetime(Folsom_Outflow['OBS DATE'])
Folsom_Outflow.set_index('OBS DATE', inplace=True)



#Daily Data -- Shasta
Shasta_Outflow = pd.read_excel('Yash/Misc_Data/Shasta_Outflow.xlsx')
Shasta_Outflow = Shasta_Outflow[['OBS DATE', 'VALUE']]
Shasta_Outflow['VALUE'] = pd.to_numeric(Shasta_Outflow['VALUE'].str.replace(',', ''))
Shasta_Outflow['OBS DATE'] = pd.to_datetime(Shasta_Outflow['OBS DATE'])
Shasta_Outflow.set_index('OBS DATE', inplace=True)



# %%
#Global Hyperparameters
shasta_penstock = 17600 #CFS
folsom_penstock = 6900 #CFS
cfs_to_afd = 1.983  #CFS to Acre-ft


#Count Days of Spill
shasta_spill_count = Shasta_Outflow['VALUE'][Shasta_Outflow['VALUE'] > shasta_penstock].count()
folsom_spill_count = Folsom_Outflow['VALUE'][Folsom_Outflow['VALUE'] > folsom_penstock].count()



# %% Shasta

### Daily Assumption
shasta_daily_spill = (Shasta_Outflow['VALUE']-shasta_penstock).clip(lower=0)
shasta_daily_spill = shasta_daily_spill*cfs_to_afd
shasta_daily_spill = shasta_daily_spill.resample('M').sum()/1000


#Monthly Time Step
shasta_monthly_flow = pd.DataFrame({
    'Flow': Shasta_Outflow['VALUE'].resample('M').sum(),
    'DPM': Shasta_Outflow['VALUE'].resample('M').count()
})
shasta_monthly_spill = (shasta_monthly_flow['Flow']-shasta_monthly_flow['DPM']*shasta_penstock).clip(lower=0)
shasta_monthly_spill = shasta_monthly_spill*cfs_to_afd/1000


#Actual Spill
shasta_true_spill = (Shasta_Outflow['VALUE']-generator_flow['Shasta']).clip(lower=0)
shasta_true_spill = shasta_true_spill*cfs_to_afd
shasta_true_spill = shasta_true_spill.resample('M').sum()/1000


#Generate the plot
plt.figure(figsize=(10, 6))
shasta_true_spill.plot(color='purple', label='True Spill')
plt.title(f'Shasta (1996-2024) \n'
          f'Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/Shasta_Outflow.shape[0]:.2f} %)', fontsize=18)
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
shasta_true_spill.plot(color='purple', label='True Spill')
shasta_daily_spill.plot(color='cornflowerblue', label='Daily Assumption')
plt.title(f'Shasta (1996-2024) \n'
          f'Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/Shasta_Outflow.shape[0]:.2f} %)', fontsize=18)
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
shasta_true_spill.plot(color='purple', label='True Spill')
shasta_daily_spill.plot(color='cornflowerblue', label='Daily Assumption')
shasta_monthly_spill.plot(color='goldenrod', label='Monthly Assumption')
plt.title(f'Shasta (1996-2024) \n'
          f'Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/Shasta_Outflow.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()


# Create a figure with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First scatter plot
axs[0].scatter(shasta_true_spill, shasta_daily_spill)
max_val = max(shasta_monthly_spill.max(), shasta_daily_spill.max(), shasta_true_spill.max())
min_val = min(shasta_monthly_spill.min(), shasta_daily_spill.min(), shasta_true_spill.min())
axs[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
axs[0].set_title(' ')
axs[0].set_ylabel('Spill By Daily Assumption  \n (Thousand Acre-ft)', fontsize=16)
axs[0].set_xlabel('Actual Spill  \n (Thousand Acre-ft)', fontsize=16)
axs[0].grid(True)

# Second scatter plot
axs[1].scatter(shasta_true_spill, shasta_monthly_spill)
max_val2 = max(shasta_monthly_spill.max(), shasta_daily_spill.max(), shasta_true_spill.max())
min_val2 = min(shasta_monthly_spill.min(), shasta_daily_spill.min(), shasta_true_spill.min())
axs[1].plot([min_val2, max_val2], [min_val2, max_val2], color='red', linestyle='--')
axs[1].set_title(' ')
axs[1].set_ylabel('Spill By Monthly Assumption  \n (Thousand Acre-ft)', fontsize=12)
axs[1].set_xlabel('Actual Spill  \n (Thousand Acre-ft)', fontsize=16)
axs[1].grid(True)

# Add a common title
fig.suptitle(f'Shasta (1996-2024) \n Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/Shasta_Outflow.shape[0]:.2f} %)', fontsize=20)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show plot
plt.show()



# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(shasta_daily_spill, shasta_monthly_spill,)
max_val = max(shasta_monthly_spill.max(), shasta_daily_spill.max())
min_val = min(shasta_monthly_spill.min(), shasta_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Shasta (1996-2024) \n Count Days with Spill: {shasta_spill_count} ({100*shasta_spill_count/Shasta_Outflow.shape[0]:.2f} %)', fontsize=20)
plt.xlabel('Monthly Spill By Daily Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Spill By Monthly Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()

# %% Folsom

### Daily Assumption
folsom_daily_spill = (Folsom_Outflow['VALUE']-folsom_penstock).clip(lower=0)
folsom_daily_spill = folsom_daily_spill*cfs_to_afd
folsom_daily_spill = folsom_daily_spill.resample('M').sum()/1000


#Monthly Time Step
folsom_monthly_flow = pd.DataFrame({
    'Flow': Folsom_Outflow['VALUE'].resample('M').sum(),
    'DPM': Folsom_Outflow['VALUE'].resample('M').count()
})
folsom_monthly_spill = (folsom_monthly_flow['Flow']-folsom_monthly_flow['DPM']*folsom_penstock).clip(lower=0)
folsom_monthly_spill = folsom_monthly_spill*cfs_to_afd/1000


#Actual Spill
folsom_true_spill = (Folsom_Outflow['VALUE']-generator_flow['Folsom']).clip(lower=0)
folsom_true_spill = folsom_true_spill*cfs_to_afd
folsom_true_spill = folsom_true_spill.resample('M').sum()/1000


#Generate the plot
plt.figure(figsize=(10, 6))
folsom_true_spill.plot(color='purple', label='True Spill')
plt.title(f'Folsom (1996-2024) \n'
          f'Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/Folsom_Outflow.shape[0]:.2f} %)', fontsize=18)
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
folsom_true_spill.plot(color='purple', label='True Spill')
folsom_daily_spill.plot(color='cornflowerblue', label='Daily Assumption')
plt.title(f'Folsom (1996-2024) \n'
          f'Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/Folsom_Outflow.shape[0]:.2f} %)', fontsize=18)
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
folsom_true_spill.plot(color='purple', label='True Spill')
folsom_daily_spill.plot(color='cornflowerblue', label='Daily Assumption')
folsom_monthly_spill.plot(color='goldenrod', label='Monthly Assumption')
plt.title(f'Folsom (1996-2024) \n'
          f'Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/Folsom_Outflow.shape[0]:.2f} %)', fontsize=18)
plt.xlabel('Month',
           fontsize = 16)
plt.ylabel('Monthly Spill  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.legend(fontsize=18)  # Add a legend to distinguish the time series
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()


# Create a figure with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First scatter plot
axs[0].scatter(folsom_true_spill, folsom_daily_spill)
max_val = max(folsom_monthly_spill.max(), folsom_daily_spill.max(), folsom_true_spill.max())
min_val = min(folsom_monthly_spill.min(), folsom_daily_spill.min(), folsom_true_spill.min())
axs[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
axs[0].set_title(' ')
axs[0].set_ylabel('Spill By Daily Assumption  \n (Thousand Acre-ft)', fontsize=16)
axs[0].set_xlabel('Actual Spill  \n (Thousand Acre-ft)', fontsize=16)
axs[0].grid(True)

# Second scatter plot
axs[1].scatter(folsom_true_spill, folsom_monthly_spill)
max_val = max(folsom_monthly_spill.max(), folsom_daily_spill.max(), folsom_true_spill.max())
min_val = min(folsom_monthly_spill.min(), folsom_daily_spill.min(), folsom_true_spill.min())
axs[1].plot([min_val2, max_val2], [min_val2, max_val2], color='red', linestyle='--')
axs[1].set_title(' ')
axs[1].set_ylabel('Spill By Monthly Assumption  \n (Thousand Acre-ft)', fontsize=12)
axs[1].set_xlabel('Actual Spill  \n (Thousand Acre-ft)', fontsize=16)
axs[1].grid(True)

# Add a common title
fig.suptitle(f'Folsom (1996-2024) \n Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/Folsom_Outflow.shape[0]:.2f} %)', fontsize=20)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show plot
plt.show()



# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(folsom_daily_spill, folsom_monthly_spill)
max_val = max(folsom_monthly_spill.max(), folsom_daily_spill.max())
min_val = min(folsom_monthly_spill.min(), folsom_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Folsom (1996-2024) \n Count Days with Spill: {folsom_spill_count} ({100*folsom_spill_count/Folsom_Outflow.shape[0]:.2f} %)', fontsize=20)
plt.xlabel('Monthly Spill By Daily Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Spill By Monthly Time Step  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()



100*(sum(folsom_daily_spill) - sum(folsom_monthly_spill))/sum(folsom_daily_spill)

100*(sum(shasta_daily_spill) - sum(shasta_monthly_spill))/sum(shasta_daily_spill)

