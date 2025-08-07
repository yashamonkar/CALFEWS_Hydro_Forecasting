# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:09:42 2024

@author: amonkar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:47:38 2024

@author: Yash Vijay Amonkar

Code to analyze the validity of the Uniform Monthly Release Assumption. 
This code is for Shasta only. 
This code also analyzes the hydropower overestimation in $/MWh

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
import calendar


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
penstock_overestimate['Overestimate'] = penstock_overestimate['Monthly'] - penstock_overestimate['Daily']  

# %%

def get_shasta_generation(release, storage, discharge_capacity = 9999999):
    
    #Curtail Releases
    release = release.apply(lambda x: min(x, discharge_capacity))
    
    #Convert Releases to Tailwater Elevation
    def get_tailwater_elev_shasta(df):
        return 1 / ((0.0000000000000197908 * ((df * 1000 / 31 / 1.9834711) - 40479.9296) ** 2) + 0.00168)
    tailwater_elevation = get_tailwater_elev_shasta(release)
    
    #Covert Storage to Forebay Elevation 
    def get_forebay_elev_shasta(df):
        return (740.270709 + 
                (0.0002185318721 * (df * 1000)) - 
                (0.0000000001006141253 * (df * 1000) ** 2) + 
                (3.224005059E-17 * (df * 1000) ** 3) - 
                (5.470777842E-24 * (df * 1000) ** 4) + 
                (3.711432277E-31 * (df * 1000) ** 5))
    forebay_elevation = get_forebay_elev_shasta(storage)
    
    #Compute the Gross Head 
    gross_head = forebay_elevation-tailwater_elevation
    
    #Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 1.045 * ((0.83522 * df) + 30.5324)
    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF*release/10**3


#%% Storage 

#Daily Release Data
Shasta_Storage = pd.read_excel('Shasta_Storage.xlsx')
Shasta_Storage = Shasta_Storage[['OBS DATE', 'VALUE']]
Shasta_Storage['VALUE'] = pd.to_numeric(Shasta_Storage['VALUE'].str.replace(',', ''))
Shasta_Storage['OBS DATE'] = pd.to_datetime(Shasta_Storage['OBS DATE'])
Shasta_Storage.set_index('OBS DATE', inplace=True)


###Shasta Hyper-parameters
shasta_capacity = 676 #MW
shasta_discharge_capacity = 34.909 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf


#Estimate Generation in GWh
true_shasta_gen = get_shasta_generation(Shasta_Outflow['VALUE']*cfs_to_afd/1000, Shasta_Storage['VALUE']/1000, shasta_discharge_capacity)

#### Overestimated Production


# Assuming your dataframes are already created and named Shasta_Outflow and penstock_overestimate
Shasta_Outflow['Mod_Value'] = Shasta_Outflow['VALUE']
for date, row in penstock_overestimate.iterrows():
    
    # Get the year and month from the current date
    year = date.year
    month = date.month
    
    # Calculate number of days in that month
    days_in_month = calendar.monthrange(year, month)[1]
    
    # Calculate daily adjustment
    daily_adjustment = row['Overestimate'] / days_in_month
    print(1000*daily_adjustment/cfs_to_afd)
    
    # Create mask for the current month in Shasta_Outflow
    mask = (Shasta_Outflow.index.year == year) & (Shasta_Outflow.index.month == month)
    
    # Add the daily adjustment to all days in that month
    Shasta_Outflow.loc[mask, 'Mod_Value'] += 1000*daily_adjustment/cfs_to_afd


#Overestimated Hydropower
#Estimate Generation in GWh
est_shasta_gen = get_shasta_generation(Shasta_Outflow['Mod_Value']*cfs_to_afd/1000, Shasta_Storage['VALUE']/1000, shasta_discharge_capacity)






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


# Hydropower Overestimation Values 
plt.figure(figsize=(10, 6))
plt.scatter(true_shasta_gen.resample('M').sum(), est_shasta_gen.resample('M').sum())
max_val = max(true_shasta_gen.resample('M').sum().max(), est_shasta_gen.resample('M').sum().max())
min_val = min(true_shasta_gen.resample('M').sum().min(), est_shasta_gen.resample('M').sum().min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Hydropower Generation \n Shasta (WY 1997 - WY 2023) ', fontsize=20)
plt.xlabel('Daily Models  \n (GWh)',
           fontsize = 16)
plt.ylabel('Monthly Models \n (GWh)',
           fontsize = 16)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.show()


#Combine the values 
Hydropower_Generation = pd.DataFrame({'Daily_Model':true_shasta_gen.resample('M').sum(),
                                      'Monthly_Model':est_shasta_gen.resample('M').sum(),
                                      'Diff':est_shasta_gen.resample('M').sum() - true_shasta_gen.resample('M').sum()})
Hydropower_Generation['Diff'].mean()
Hydropower_Generation.to_csv("Shasta_Hydropower_Est.csv")
