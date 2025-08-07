# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:47:38 2024

@author: amonkar

Code to analyze the validity of the Uniform Monthly Release Assumption. 
For (i) Shasta, (ii) Oroville, (iii) Folsom, (iv) New Melones
Data are from CDEC. 10-01-1996 to 09-30-2023 (WY 1997 - WY 2023)
The daily water spill aggregated monthly are compared against the monthly water
spills based on the penstock capacity. 

"""


# %% Initial Loading

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS\Yash\Misc_Data'
os.chdir(working_directory)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#Daily Data -- Folsom
Folsom_Outflow = pd.read_excel('Folsom_Outflow.xlsx')
Folsom_Outflow = Folsom_Outflow[['OBS DATE', 'VALUE']]
Folsom_Outflow['VALUE'] = pd.to_numeric(Folsom_Outflow['VALUE'].str.replace(',', ''))
Folsom_Outflow['OBS DATE'] = pd.to_datetime(Folsom_Outflow['OBS DATE'])
Folsom_Outflow.set_index('OBS DATE', inplace=True)



#Daily Data -- Shasta
Shasta_Outflow = pd.read_excel('Shasta_Outflow.xlsx')
Shasta_Outflow = Shasta_Outflow[['OBS DATE', 'VALUE']]
Shasta_Outflow['VALUE'] = pd.to_numeric(Shasta_Outflow['VALUE'].str.replace(',', ''))
Shasta_Outflow['OBS DATE'] = pd.to_datetime(Shasta_Outflow['OBS DATE'])
Shasta_Outflow.set_index('OBS DATE', inplace=True)


#Daily Data -- Oroville
Oroville_Outflow = pd.read_excel('Oroville_Outflow.xlsx')
Oroville_Outflow = Oroville_Outflow[['OBS DATE', 'VALUE']]
Oroville_Outflow['VALUE'] = pd.to_numeric(Oroville_Outflow['VALUE'].str.replace(',', ''))
Oroville_Outflow['OBS DATE'] = pd.to_datetime(Oroville_Outflow['OBS DATE'])
Oroville_Outflow.set_index('OBS DATE', inplace=True)



#Daily Data -- New Melones
New_Melones_Outflow = pd.read_excel('New_Melones_Outflow.xlsx')
New_Melones_Outflow = New_Melones_Outflow[['OBS DATE', 'VALUE']]
New_Melones_Outflow['VALUE'] = pd.to_numeric(New_Melones_Outflow['VALUE'].str.replace(',', ''))
New_Melones_Outflow['OBS DATE'] = pd.to_datetime(New_Melones_Outflow['OBS DATE'])
New_Melones_Outflow.set_index('OBS DATE', inplace=True)


#Historic Water Year Types https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST
wy_types = pd.read_csv("Water_Year_Types.csv", index_col = 0)
wy_types.index = pd.to_datetime(wy_types.index)
wy_types = wy_types[wy_types.index > '1996-09-30']
wy_types.index = wy_types.index + pd.offsets.MonthEnd(0) #Align index to the end of the month. 


# %%
#Global Hyperparameters
shasta_penstock = 17600 #CFS
folsom_penstock = 6900 #CFS
oroville_penstock = 16950 #CFS https://web.archive.org/web/20120407054356/http://www.water.ca.gov/swp/facilities/Oroville/hyatt.cfm
new_melones_penstock = 8290 #CFS
cfs_to_afd = 1.983  #CFS to Acre-ft


#Subset the to needed period
end_date = '2023-10-01'
Oroville_Outflow = Oroville_Outflow[Oroville_Outflow.index < end_date]
New_Melones_Outflow = New_Melones_Outflow[New_Melones_Outflow.index < end_date]


#Count Days of Spill
shasta_spill_count = Shasta_Outflow['VALUE'][Shasta_Outflow['VALUE'] > shasta_penstock].count()
folsom_spill_count = Folsom_Outflow['VALUE'][Folsom_Outflow['VALUE'] > folsom_penstock].count()
oroville_spill_count = Oroville_Outflow['VALUE'][Oroville_Outflow['VALUE'] > oroville_penstock].count()
new_melones_spill_count = New_Melones_Outflow['VALUE'][New_Melones_Outflow['VALUE'] > new_melones_penstock].count()


# %% Shasta

### Daily Time Step Release Assumption
shasta_daily_spill = (Shasta_Outflow['VALUE']-shasta_penstock).clip(lower=0)
shasta_daily_spill = shasta_daily_spill*cfs_to_afd #Convert to Acre-feet/day
shasta_daily_spill = shasta_daily_spill.resample('M').sum()/1000 #Sum to monthly

shasta_daily_penstock = Shasta_Outflow['VALUE'].clip(upper=shasta_penstock)
shasta_daily_penstock = shasta_daily_penstock*cfs_to_afd #Convert to Acre-feet/day
shasta_daily_penstock = shasta_daily_penstock.resample('M').sum()/1000 #Sum to monthly

#Monthly Time Step Release Assumption
shasta_monthly_flow = pd.DataFrame({
    'Flow': Shasta_Outflow['VALUE'].resample('M').sum(),
    'DPM': Shasta_Outflow['VALUE'].resample('M').count() #Days per month
})
shasta_monthly_spill = (shasta_monthly_flow['Flow']-shasta_monthly_flow['DPM']*shasta_penstock).clip(lower=0)
shasta_monthly_spill = shasta_monthly_spill*cfs_to_afd/1000

shasta_monthly_penstock = shasta_monthly_flow['Flow'].clip(upper = shasta_monthly_flow['DPM']*shasta_penstock)
shasta_monthly_penstock = shasta_monthly_penstock*cfs_to_afd/1000

# Create the scatter plot -- Spill
plt.figure(figsize=(10, 6))
plt.scatter(shasta_daily_spill, shasta_monthly_spill)
max_val = max(shasta_monthly_spill.max(), shasta_daily_spill.max())
min_val = min(shasta_monthly_spill.min(), shasta_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Spills \n Shasta (WY 1997 - WY 2023) ', fontsize=20)
plt.xlabel('Daily Time Step Model  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Model \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


# Create the scatter plot -- Flow through penstock
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


#Metric I -- Underestimation of spill
f'Over-estimation for Shasta is {round(100*(sum(shasta_daily_spill) - sum(shasta_monthly_spill))/sum(shasta_monthly_spill),2)}%'

#Metric II -- Underestimation of spill
f'Over-estimation for Shasta is {round(100*(sum(shasta_monthly_penstock) - sum(shasta_daily_penstock))/sum(shasta_daily_penstock),2)}%'




###Penstock Overestimation
penstock_overestimate = pd.DataFrame({
    'Daily': shasta_daily_penstock,
    'Monthly': shasta_monthly_penstock, 
    'WYT': wy_types['SAC_Index']
})
penstock_overestimate['Month'] = pd.to_datetime(penstock_overestimate.index)
penstock_overestimate['Month'] = penstock_overestimate['Month'].dt.month

filtered_penstock_overestimate = penstock_overestimate[penstock_overestimate['Month'].isin([1, 2, 3, 4])]
round(100*(filtered_penstock_overestimate['Monthly'].sum() - filtered_penstock_overestimate['Daily'].sum()) / filtered_penstock_overestimate['Daily'].sum(),2)

monthly_penstock_overestimate = penstock_overestimate.groupby(['WYT', 'Month']).apply(
    lambda x: round(100*(x['Monthly'].sum() - x['Daily'].sum()) / x['Daily'].sum(),2)
).reset_index(name='Calculation')

def calculate_penstock_overestimate(group):
    daily_sum = group['Daily'].sum()
    monthly_sum = group['Monthly'].sum()
    return 100 * (monthly_sum - daily_sum) / daily_sum


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



Month = '2017-03-31'

Shasta_Example = Shasta_Outflow*cfs_to_afd/1000
Shasta_Example = Shasta_Example[Shasta_Example.index > '2016-12-31']
Shasta_Example = Shasta_Example[Shasta_Example.index < '2017-05-01']

#Compute the Uniform Monthly Release Assumption
monthly_avg = Shasta_Example.resample('MS')['VALUE'].mean()
monthly_avg_daily = monthly_avg.reindex(Shasta_Example.index, method='ffill')
Shasta_Example['Monthly_Avg'] = monthly_avg_daily




plt.figure(figsize=(10, 6))
plt.fill_between(Shasta_Example.index, Shasta_Example['Monthly_Avg'], 0, color='yellow', alpha=0)
plt.plot(Shasta_Example.index, Shasta_Example['Monthly_Avg'], linewidth=0, color='orange', label='Monthly Model')
plt.fill_between(Shasta_Example.index, Shasta_Example['VALUE'], 0, color='blue', alpha=0)
plt.plot(Shasta_Example.index, Shasta_Example['VALUE'], linewidth=0, color='blue' , label='Daily Model')
plt.title('Shasta Releases in 2017 (Wet Water Year)', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Daily Releases (TAF)', fontsize=16)
for month in pd.date_range(start='2017-01-01', end='2017-05-01', freq='MS'):
    plt.axvline(x=month, color='black', linewidth=1.15)
plt.axhline(y=shasta_penstock*cfs_to_afd/1000, color='red', linestyle='--', linewidth=2)
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=16)
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()


###Penstock Overestimation
penstock_overestimate = pd.DataFrame({
    'Daily': shasta_daily_penstock,
    'Monthly': shasta_monthly_penstock, 
    'WYT': wy_types['SAC_Index']
})
penstock_overestimate['Overestimate'] = 100*(penstock_overestimate['Monthly']-penstock_overestimate['Daily'])/penstock_overestimate['Daily']
penstock_overestimate['Difference'] = penstock_overestimate['Monthly']-penstock_overestimate['Daily']



# %% Folsom

### Daily Time Step Release Assumption
folsom_daily_spill = (Folsom_Outflow['VALUE']-folsom_penstock).clip(lower=0)
folsom_daily_spill = folsom_daily_spill*cfs_to_afd
folsom_daily_spill = folsom_daily_spill.resample('M').sum()/1000

folsom_daily_penstock = Folsom_Outflow['VALUE'].clip(upper=folsom_penstock)
folsom_daily_penstock = folsom_daily_penstock*cfs_to_afd #Convert to Acre-feet/day
folsom_daily_penstock = folsom_daily_penstock.resample('M').sum()/1000 #Sum to monthly


#Monthly Time Step Release Assumption
folsom_monthly_flow = pd.DataFrame({
    'Flow': Folsom_Outflow['VALUE'].resample('M').sum(),
    'DPM': Folsom_Outflow['VALUE'].resample('M').count()
})
folsom_monthly_spill = (folsom_monthly_flow['Flow']-folsom_monthly_flow['DPM']*folsom_penstock).clip(lower=0)
folsom_monthly_spill = folsom_monthly_spill*cfs_to_afd/1000

folsom_monthly_penstock = folsom_monthly_flow['Flow'].clip(upper = folsom_monthly_flow['DPM']*folsom_penstock)
folsom_monthly_penstock = folsom_monthly_penstock*cfs_to_afd/1000


# Create the scatter plot -- Spills
plt.figure(figsize=(10, 6))
plt.scatter(folsom_daily_spill, folsom_monthly_spill)
max_val = max(folsom_monthly_spill.max(), folsom_daily_spill.max())
min_val = min(folsom_monthly_spill.min(), folsom_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Spills \n Folsom (1996-2023) ', fontsize=20)
plt.xlabel('Daily Time Step Models  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Models \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()



# Create the scatter plot -- Flow through penstock
plt.figure(figsize=(10, 6))
plt.scatter(folsom_daily_penstock, folsom_monthly_penstock)
max_val = max(folsom_monthly_penstock.max(), folsom_daily_penstock.max())
min_val = min(folsom_monthly_penstock.min(), folsom_daily_penstock.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Penstock Flows \n Folsom (1996-2023) ', fontsize=20)
plt.xlabel('Daily Time Step Assumption  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Assumption \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


#Monthly Counts 
spill_difference = folsom_daily_spill - folsom_monthly_spill
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
plt.ylabel('Count Months with Spill Under-estimation', fontsize = 16)
plt.title(f'Folsom (1996-2023)', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


###Penstock Overestimation
penstock_overestimate = pd.DataFrame({
    'Daily': folsom_daily_penstock,
    'Monthly': folsom_monthly_penstock, 
    'WYT': wy_types['SAC_Index']
})
penstock_overestimate['Month'] = pd.to_datetime(penstock_overestimate.index)
penstock_overestimate['Month'] = penstock_overestimate['Month'].dt.month

filtered_penstock_overestimate = penstock_overestimate[penstock_overestimate['Month'].isin([1, 2, 3, 4])]
round(100*(filtered_penstock_overestimate['Monthly'].sum() - filtered_penstock_overestimate['Daily'].sum()) / filtered_penstock_overestimate['Daily'].sum(),2)

round(100*(penstock_overestimate['Monthly'].sum() - penstock_overestimate['Daily'].sum()) / penstock_overestimate['Daily'].sum(),2)


# %% Oroville

### Daily Time Step Release Assumption
oroville_daily_spill = (Oroville_Outflow['VALUE']-oroville_penstock).clip(lower=0)
oroville_daily_spill = oroville_daily_spill*cfs_to_afd
oroville_daily_spill = oroville_daily_spill.resample('M').sum()/1000

oroville_daily_penstock = Oroville_Outflow['VALUE'].clip(upper=oroville_penstock)
oroville_daily_penstock = oroville_daily_penstock*cfs_to_afd #Convert to Acre-feet/day
oroville_daily_penstock = oroville_daily_penstock.resample('M').sum()/1000 #Sum to monthly


#Monthly Time Step Release Assumption
oroville_monthly_flow = pd.DataFrame({
    'Flow': Oroville_Outflow['VALUE'].resample('M').sum(),
    'DPM': Oroville_Outflow['VALUE'].resample('M').count()
})
oroville_monthly_spill = (oroville_monthly_flow['Flow']-oroville_monthly_flow['DPM']*oroville_penstock).clip(lower=0)
oroville_monthly_spill = oroville_monthly_spill*cfs_to_afd/1000

oroville_monthly_penstock = oroville_monthly_flow['Flow'].clip(upper = oroville_monthly_flow['DPM']*oroville_penstock)
oroville_monthly_penstock = oroville_monthly_penstock*cfs_to_afd/1000


# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(oroville_daily_spill, oroville_monthly_spill)
max_val = max(oroville_monthly_spill.max(), oroville_daily_spill.max())
min_val = min(oroville_monthly_spill.min(), oroville_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Spills \n Oroville (1996-2023) ', fontsize=20)
plt.xlabel('Daily Time Step Models  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Models \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


# Create the scatter plot -- Flow through penstock
plt.figure(figsize=(10, 6))
plt.scatter(oroville_daily_penstock, oroville_monthly_penstock)
max_val = max(oroville_monthly_penstock.max(), oroville_daily_penstock.max())
min_val = min(oroville_monthly_penstock.min(), oroville_daily_penstock.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Penstock Flows \n Oroville (1996-2023) ', fontsize=20)
plt.xlabel('Daily Time Step Models  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Models \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


#Monthly Counts 
spill_difference = oroville_daily_spill - oroville_monthly_spill
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
plt.ylabel('Count Months with Spill Under-estimation', fontsize = 16)
plt.title(f'Oroville (1996-2023)', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


###Penstock Overestimation
penstock_overestimate = pd.DataFrame({
    'Daily': oroville_daily_penstock,
    'Monthly': oroville_monthly_penstock, 
    'WYT': wy_types['SAC_Index']
})
penstock_overestimate['Month'] = pd.to_datetime(penstock_overestimate.index)
penstock_overestimate['Month'] = penstock_overestimate['Month'].dt.month

filtered_penstock_overestimate = penstock_overestimate[penstock_overestimate['Month'].isin([1, 2, 3, 4])]
round(100*(filtered_penstock_overestimate['Monthly'].sum() - filtered_penstock_overestimate['Daily'].sum()) / filtered_penstock_overestimate['Daily'].sum(),2)

round(100*(penstock_overestimate['Monthly'].sum() - penstock_overestimate['Daily'].sum()) / penstock_overestimate['Daily'].sum(),2)



# %% New_Melones

### Daily Time Step Release Assumption
new_melones_daily_spill = (New_Melones_Outflow['VALUE']-new_melones_penstock).clip(lower=0)
new_melones_daily_spill = new_melones_daily_spill*cfs_to_afd
new_melones_daily_spill = new_melones_daily_spill.resample('M').sum()/1000

new_melones_daily_penstock = New_Melones_Outflow['VALUE'].clip(upper=new_melones_penstock)
new_melones_daily_penstock = new_melones_daily_penstock*cfs_to_afd #Convert to Acre-feet/day
new_melones_daily_penstock = new_melones_daily_penstock.resample('M').sum()/1000 #Sum to monthly


#Monthly Time Step Release Assumption
new_melones_monthly_flow = pd.DataFrame({
    'Flow': New_Melones_Outflow['VALUE'].resample('M').sum(),
    'DPM': New_Melones_Outflow['VALUE'].resample('M').count()
})
new_melones_monthly_spill = (new_melones_monthly_flow['Flow']-new_melones_monthly_flow['DPM']*new_melones_penstock).clip(lower=0)
new_melones_monthly_spill = new_melones_monthly_spill*cfs_to_afd/1000

new_melones_monthly_penstock = new_melones_monthly_flow['Flow'].clip(upper = new_melones_monthly_flow['DPM']*shasta_penstock)
new_melones_monthly_penstock = new_melones_monthly_penstock*cfs_to_afd/1000


# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(new_melones_daily_spill, new_melones_monthly_spill)
max_val = max(new_melones_monthly_spill.max(), new_melones_daily_spill.max())
min_val = min(new_melones_monthly_spill.min(), new_melones_daily_spill.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Spills \n New Melones (1996-2023) ', fontsize=20)
plt.xlabel('Daily Time Step Models  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Models \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


# Create the scatter plot -- Flow through penstock
plt.figure(figsize=(10, 6))
plt.scatter(new_melones_daily_penstock, new_melones_monthly_penstock)
max_val = max(new_melones_monthly_penstock.max(), new_melones_daily_penstock.max())
min_val = min(new_melones_monthly_penstock.min(), new_melones_daily_penstock.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title(f'Monthly Penstock Flows \n New Melones (1996-2023) ', fontsize=20)
plt.xlabel('Daily Time Step Assumption  \n (Thousand Acre-ft)',
           fontsize = 16)
plt.ylabel('Monthly Time Step Assumption \n (Thousand Acre-ft)',
           fontsize = 16)
plt.grid(True)
plt.show()


# %% Compute the Overestimation of the Spill
#This measure the overestimation in estimated spill. 
#Note: The percentages are not a function of the total reservoir flow. 

#Fractional Spill -- Metric I
#Shasta
f'Over-estimation for Shasta is {round(100*(sum(shasta_daily_spill) - sum(shasta_monthly_spill))/sum(shasta_monthly_spill),2)}%'

#Orovile
f'Over-estimation for Oroville is {round(100*(sum(oroville_daily_spill) - sum(oroville_monthly_spill))/sum(oroville_monthly_spill),2)}%'

#Folsom
f'Over-estimation for Folsom is {round(100*(sum(folsom_daily_spill) - sum(folsom_monthly_spill))/sum(folsom_monthly_spill),2)}%'



#Flow through the penstock -- Metric II
#Shasta
f'Over-estimation for Shasta is {round(100*(sum(shasta_monthly_penstock) - sum(shasta_daily_penstock))/sum(shasta_daily_penstock),2)}%'

#Orovile
f'Over-estimation for Oroville is {round(100*(sum(oroville_monthly_penstock) - sum(oroville_daily_penstock))/sum(oroville_daily_penstock),2)}%'

#Folsom
f'Over-estimation for Folsom is {round(100*(sum(folsom_monthly_penstock) - sum(folsom_daily_penstock))/sum(folsom_daily_penstock),2)}%'



#Flow through the penstock -- Metric II
#Shasta
monthly = shasta_monthly_penstock[shasta_monthly_penstock != shasta_daily_penstock]
daily = shasta_daily_penstock[shasta_monthly_penstock != shasta_daily_penstock]
f'Over-estimation for Shasta is {round(100*(sum(monthly) - sum(daily))/sum(daily),2)}%'

#Orovile
monthly = oroville_monthly_penstock[oroville_monthly_penstock != oroville_daily_penstock]
daily = oroville_daily_penstock[oroville_monthly_penstock != oroville_daily_penstock]
f'Over-estimation for Oroville is {round(100*(sum(monthly) - sum(daily))/sum(daily),2)}%'

#Folsom
monthly = folsom_monthly_penstock[folsom_monthly_penstock != folsom_daily_penstock]
daily = folsom_daily_penstock[folsom_monthly_penstock != folsom_daily_penstock]
f'Over-estimation for Folsom is {round(100*(sum(monthly) - sum(daily))/sum(daily),2)}%'




#Compute the Overestimation of the Spill as a fraction of the total release. 
#This measure the overestimation in estimated spill. 
#Note: The percentages are not a function of the total reservoir flow. 
#Shasta
f'Over-estimation for Shasta is {round(100*(sum(shasta_daily_spill) - sum(shasta_monthly_spill))/(Shasta_Outflow["VALUE"].sum()*cfs_to_afd*10**-3),2)}%'

#Orovile
f'Over-estimation for Oroville is {round(100*(sum(oroville_daily_spill) - sum(oroville_monthly_spill))/(Oroville_Outflow["VALUE"].sum()*cfs_to_afd*10**-3),2)}%'

#Folsom
f'Over-estimation for Folsom is {round(100*(sum(folsom_daily_spill) - sum(folsom_monthly_spill))/(Folsom_Outflow["VALUE"].sum()*cfs_to_afd*10**-3),2)}%'



# %% DURING WET WATER YEAR TYPES

#Overestimation of the spill
#Shasta
Overestimate = 100*(sum(shasta_daily_spill[wy_types['SAC_Index'] =='W']) - sum(shasta_monthly_spill[wy_types['SAC_Index'] =='W']))/sum(shasta_monthly_spill[wy_types['SAC_Index'] =='W'])
f'Over-estimation for Shasta during Wet Water Years is {round(Overestimate,2)}%'

#Orovile
Overestimate = 100*(sum(oroville_daily_spill[wy_types['SAC_Index'] =='W']) - sum(oroville_monthly_spill[wy_types['SAC_Index'] =='W']))/sum(oroville_monthly_spill[wy_types['SAC_Index'] =='W'])
f'Over-estimation for Oroville during Wet Water Years is {round(Overestimate,2)}%'

#Folsom
Overestimate = 100*(sum(folsom_daily_spill[wy_types['SAC_Index'] =='W']) - sum(folsom_monthly_spill[wy_types['SAC_Index'] =='W']))/sum(folsom_monthly_spill[wy_types['SAC_Index'] =='W'])
f'Over-estimation for Folsom during Wet Water Years is {round(Overestimate,2)}%'



#Overestimation as a fraction of total releases

#Shasta
Monthly_Flow = Shasta_Outflow['VALUE'].resample('M').sum()*cfs_to_afd*10**-3 #Convert to TAF
Overestimate = 100*(sum(shasta_daily_spill[wy_types['SAC_Index'] =='W']) - sum(shasta_monthly_spill[wy_types['SAC_Index'] =='W']))/sum(Monthly_Flow[wy_types['SAC_Index'] =='W'])
f'Over-estimation for Shasta during Wet Water Years is {round(Overestimate,2)}%'

#Orovile
Monthly_Flow = Oroville_Outflow['VALUE'].resample('M').sum()*cfs_to_afd*10**-3 #Convert to TAF
Overestimate = 100*(sum(oroville_daily_spill[wy_types['SAC_Index'] =='W']) - sum(oroville_monthly_spill[wy_types['SAC_Index'] =='W']))/sum(Monthly_Flow[wy_types['SAC_Index'] =='W'])
f'Over-estimation for Oroville during Wet Water Years is {round(Overestimate,2)}%'

#Folsom
Monthly_Flow = Folsom_Outflow['VALUE'].resample('M').sum()*cfs_to_afd*10**-3 #Convert to TAF
Overestimate = 100*(sum(folsom_daily_spill[wy_types['SAC_Index'] =='W']) - sum(folsom_monthly_spill[wy_types['SAC_Index'] =='W']))/sum(Monthly_Flow[wy_types['SAC_Index'] =='W'])
f'Over-estimation for Folsom during Wet Water Years is {round(Overestimate,2)}%'


# %% Overestimation in months with spill




#Shasta
Monthly_Flow = Shasta_Outflow['VALUE'].resample('M').sum()*cfs_to_afd*10**-3 #Convert to TAF
Monthly_Flow = Monthly_Flow[shasta_daily_spill > 0]
Overestimate = 100*(sum(shasta_daily_spill[shasta_daily_spill > 0]) - sum(shasta_monthly_spill[shasta_daily_spill > 0]))/sum(Monthly_Flow)


#Oroville
Monthly_Flow = Oroville_Outflow['VALUE'].resample('M').sum()*cfs_to_afd*10**-3 #Convert to TAF
Monthly_Flow = Monthly_Flow[oroville_daily_spill > 0]
Overestimate = 100*(sum(oroville_daily_spill[oroville_daily_spill > 0]) - sum(oroville_monthly_spill[oroville_daily_spill > 0]))/sum(Monthly_Flow)


#Folsom
Monthly_Flow = Folsom_Outflow['VALUE'].resample('M').sum()*cfs_to_afd*10**-3 #Convert to TAF
Monthly_Flow = Monthly_Flow[folsom_daily_spill > 0]
Overestimate = 100*(sum(folsom_daily_spill[folsom_daily_spill > 0]) - sum(folsom_monthly_spill[folsom_daily_spill > 0]))/sum(Monthly_Flow)









