# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:43:19 2024

@author: amonkar

#Code to analyze the Trinity, Lewiston and Whiskeytown Operating Reports. 
"""


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

#Set the working directory
# Using absolute path
os.chdir('C:/Users/amonkar/Documents/CALFEWS_Preliminary')


#Read data files
trinity = pd.read_csv("data/Trinity_Daily_Operations.csv", index_col=0)
lewiston = pd.read_csv("data/Lewiston_Daily_Operations.csv", index_col = 0)


#Hyperparameters
cfs_to_tafd = 1.98211*10**-3




# %%   TRINITY WATER YEAR TYPES

#Add the Water Year
trinity['WaterYear'] = trinity['Year'] + np.where(trinity['Month'].isin(['October', 'November', 'December']), 1, 0)


#Trinity Water Year Types (2001-2024)
#Note the 2024 WYT_ACTUAL is not yet finalized. Kept as wet for the moment. 
#Note: The water year values are lagged. Be careful. 
#WTY 2000 decides the carryover storage in Oct 1st 2001 (start of WYT 2001)
#wyt_forecast = ["N", "D", "N", "W", "W", "N", "EW", "D", "N", "D", "N", "W", "N", 
#                "D", "CD", "D", "W", "EW", "CD", "W", "CD", "CD", "CD", "W"]
wyt_actual_lagged = ["N", "D", "N", "W", "W", "W", "EW", "D", "D", "D", "W", 
                     "W", "N", "D", "CD", "D", "W", "EW", "CD", "W", "CD", "CD", "CD", "W"]



#Adding the water year types to the data. 
wyt_df = pd.DataFrame({
    'WaterYear': range(2001, 2025),
    'WaterYearType': wyt_actual_lagged
})

# Merge the water year types with the original dataframe
trinity = trinity.reset_index().merge(wyt_df, on='WaterYear', how='left').set_index('index')


#%% Carryover Analysis

####---------------------Carryover Analysis---------------------------------###
# Filter data for October 1st of each year
oct_first = trinity[
    (trinity['Month'] == "October") & 
    (trinity['Day'] == 1)
]

#All Years
plt.figure(figsize=(12, 6))
plt.plot(oct_first['WaterYear'], oct_first['Storage'], marker='o', linestyle='-', linewidth=2)
plt.title('Trinity Storage Values on October 1st (2000-2023)', fontsize=18)
plt.xlabel('Water Year (Start)', fontsize=18)
plt.ylabel('Storage (TAF)', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#With the Water Year Types
color_dict = {
    'CD': 'red',
    'D': 'orange',
    'N': 'yellow',
    'W': 'lightblue',
    'EW': 'blue',
}


plt.figure(figsize=(12, 9))
plt.plot(oct_first['WaterYear'], oct_first['Storage'], 
         linestyle='-', linewidth=2, color='gray', alpha=0.5)
for year_type in color_dict:
    mask = oct_first['WaterYearType'] == year_type
    plt.scatter(oct_first.loc[mask, 'WaterYear'], 
                oct_first.loc[mask, 'Storage'],
                c=color_dict[year_type], 
                label=year_type, 
                s=100)  
plt.title('Trinity Storage Values on October 1st (2000-2023) \n Previous Years WYT', fontsize=18)
plt.axhline(y=650, color='black', linestyle='--')
plt.axhline(y=1000, color='black', linestyle='--')
plt.axhline(y=1200, color='black', linestyle='--')
plt.axhline(y=1600, color='black', linestyle='--')
plt.xlabel('Water Year (Start)', fontsize=24)
plt.ylabel('Storage (TAF)', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(rotation=45, fontsize=20)
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=5, fontsize=18) 
plt.tight_layout(rect=[0, 0.1, 1, 1])  
plt.show()



# %% Lewiston Analysis Exports and ROD Flows

#Add the Water Year
lewiston['WaterYear'] = lewiston['Year'] + np.where(lewiston['Month'].isin(['October', 'November', 'December']), 1, 0)

#Aggregate flows
columns_to_sum = ['Power', 'Outlet', 'Spill', 'Diversion']
lewiston_annual = lewiston[columns_to_sum].groupby(lewiston['WaterYear']).sum()*cfs_to_tafd
lewiston_annual['ROD_flows'] = lewiston_annual['Power'] + lewiston_annual['Outlet'] + lewiston_annual['Spill']

#Aggregate the River Flows
wyt_actual = ["D", "N", "W", "W", "W", "EW", "D", "D", "D", "W", "W", "N", "D", 
              "CD", "D", "W", "EW", "CD", "W", "CD", "CD", "CD", "W", "W"]
legal_rod_flows = [369, 470, 453, 647, 647, 815, 453, 647, 453, 647, 701, 647, 
                  453, 369, 453, 701, 815, 369, 701,  369, 369, 369, 701, 701]


#Add the therotical data. 
lewiston_annual["WYT"] = wyt_actual
lewiston_annual['Legal_ROD_Flows'] = legal_rod_flows



### Analysis of the ROD Flows:
plt.figure(figsize=(12, 6))
x = np.arange(len(lewiston_annual))
width = 0.35
colors = [color_dict[wyt] for wyt in lewiston_annual['WYT']]
plt.bar(x - width/2, lewiston_annual['ROD_flows'], width, label='Flows to Trinity River', color='gray')
plt.bar(x + width/2, lewiston_annual['Legal_ROD_Flows'], width, label='Legal ROD 2001 Flows', color=colors)
plt.xlabel('Water Year', fontsize=18)
plt.ylabel('Flow (TAF)', fontsize=18)
plt.title('Legal ROD Flows vs Flows into Trinity River by Water Year' , fontsize=22)
plt.xticks(x, lewiston_annual.index, rotation=45)
plt.legend()
wyt_patches = [plt.Rectangle((0,0),1,1, color=color) for color in color_dict.values()]
plt.legend(handles=[
    plt.Rectangle((0,0),1,1, color='gray'),
    plt.Rectangle((0,0),1,1, color='white')
] + wyt_patches, 
    labels=['Flows into Trinty' 'Legal ROD 2001 Flow',] + list(color_dict.keys()),
    loc='upper right', fontsize=14)
plt.tight_layout()
plt.show()


#%%
#Analysis of the exports
#Multiply by 1.04 https://www.trrp.net/restoration/flows/summary/
trinity_annual = trinity['Inflow_CFS'].groupby(lewiston['WaterYear']).sum()*cfs_to_tafd*1.04

#Add FNF to Lewiston Annual
lewiston_annual['Annual_FNF'] = trinity_annual
lewiston_annual['Theortical_Diversion'] = lewiston_annual['Annual_FNF'] - lewiston_annual['Legal_ROD_Flows']


### Analysis of the ROD Flows:
plt.figure(figsize=(12, 6))
x = np.arange(len(lewiston_annual))
width = 0.35
colors = [color_dict[wyt] for wyt in lewiston_annual['WYT']]
plt.bar(x - width/2, lewiston_annual['Diversion'], width, label='CVP Diversions', color='gray')
plt.bar(x + width/2, lewiston_annual['Theortical_Diversion'], width, label='ROD Diversions', color=colors)
plt.xlabel('Water Year', fontsize=18)
plt.ylabel('Flow (TAF)', fontsize=18)
plt.title('CVP Diversions vs ROD Diversions by Water Year' , fontsize=22)
plt.xticks(x, lewiston_annual.index, rotation=45)
plt.legend()
wyt_patches = [plt.Rectangle((0,0),1,1, color=color) for color in color_dict.values()]
plt.legend(handles=[
    plt.Rectangle((0,0),1,1, color='gray'),
    plt.Rectangle((0,0),1,1, color='white')
] + wyt_patches, 
    labels=['CVP Diversions', 'ROD Diversions',] + list(color_dict.keys()),
    loc='upper right', ncol = 2, fontsize=14)
plt.tight_layout()
plt.show()


# %%

plt.figure(figsize=(12, 6))

# Create primary axis for bars
ax1 = plt.gca()
x = np.arange(len(lewiston_annual))
width = 0.35
colors = [color_dict[wyt] for wyt in lewiston_annual['WYT']]
ax1.bar(x - width/2, lewiston_annual['Diversion'], width, label='CVP Diversions', color='gray')
ax1.bar(x + width/2, lewiston_annual['Theortical_Diversion'], width, label='ROD Diversions', color=colors)
ax1.set_xlabel('Water Year', fontsize=18)
ax1.set_ylabel('Flow (TAF)', fontsize=18)

# Create secondary axis for storage line with lagged values
ax2 = ax1.twinx()
# Shift storage values by one year (lag)
lagged_storage = pd.Series(index=oct_first['WaterYear'][:-1], data=oct_first['Storage'][1:].values)
ax2.plot(x[:-1], lagged_storage, marker='o', linestyle='-', linewidth=2, color='blue', label='Carryover Storage')
ax2.set_ylabel('Storage (TAF)', fontsize=18, color='blue')
# Set y-axis limits for storage
ax2.set_ylim(0, 2200)

# Title
plt.title('CVP Diversions vs ROD Diversions\nCarryover Storage (EOY)', fontsize=22)

# X-axis settings
plt.xticks(x, lewiston_annual.index, rotation=45)

# Create combined legend
wyt_patches = [plt.Rectangle((0,0),1,1, color=color) for color in color_dict.values()]
lines = [plt.Line2D([0], [0], color='blue', linewidth=2, marker='o')]
all_handles = [
    plt.Rectangle((0,0),1,1, color='gray'),
    plt.Rectangle((0,0),1,1, color='white')
] + wyt_patches + lines
all_labels = ['CVP Diversions', 'ROD Diversions'] + list(color_dict.keys()) + ['Carryover Storage']

ax1.legend(handles=all_handles, 
          labels=all_labels,
          loc='upper right', 
          ncol=2, 
          fontsize=14)

plt.tight_layout()
plt.show()


#_____________________________________________________________________________#
#Compute Trinity Metrics
trinity['Total'] = trinity['Power'] + trinity['Outlet'] +trinity['Spill']
trinity['Total'] = trinity['Total']*cfs_to_tafd
trinity.index = pd.DatetimeIndex(trinity.index)
#trinity['Total'].resample('A-SEP').sum()


# Create figure and axis
plt.figure(figsize=(15, 8))
colors = LinearSegmentedColormap.from_list('custom_red_blue', ['#8B0000', '#00008B'])
lines = []
labels = []

def day_of_water_year(date):
    if date.month >= 10:
        return (date - pd.Timestamp(year=date.year, month=10, day=1)).days + 1
    else:
        return (date - pd.Timestamp(year=date.year-1, month=10, day=1)).days + 1

# Create a dictionary to store values for each day of water year
daily_values = {day: [] for day in range(1, 367)}

# First pass: collect values for each day of water year
for year in range(1995, 2023):
    mask = ((trinity.index.month >= 10) & (trinity.index.year == year)) | \
           ((trinity.index.month < 10) & (trinity.index.year == year + 1))
    water_year_data = trinity['Total'][mask]
    
    if not water_year_data.empty:
        for date, value in water_year_data.items():
            day = day_of_water_year(date)
            daily_values[day].append(value)

# Calculate median for each day
daily_medians = {day: np.median(values) for day, values in daily_values.items() if values}

# Plot each water year's data
for year in range(1995, 2023):
    mask = ((trinity.index.month >= 10) & (trinity.index.year == year)) | \
           ((trinity.index.month < 10) & (trinity.index.year == year + 1))
    water_year_data = trinity['Total'][mask]
    if not water_year_data.empty:
        x_values = [day_of_water_year(date) for date in water_year_data.index]
        color = colors((year - 1995) / 28)
        line, = plt.plot(x_values, water_year_data.values, 
                        color=color, linewidth=1.5, alpha=0.8)
        lines.append(line)
        labels.append(f'WY {year}')

# Add median line
median_x = list(daily_medians.keys())
median_y = list(daily_medians.values())
median_line, = plt.plot(median_x, median_y, 'r-', linewidth=2.5, label='Median')
lines.append(median_line)
labels.append('Median')

# Rest of your plotting code remains the same
plt.title('Trinity Releases (WY 1995-2014)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylim(0, 30)
plt.ylabel('Releases (TAF)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(lines, labels, title='Water Year', bbox_to_anchor=(1.05, 1), 
           loc='upper left', borderaxespad=0.)
month_positions = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
               'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
plt.xticks(month_positions, month_labels)
plt.tight_layout()
plt.show()




#Actual post ROD releases
trinity['Total'].resample('A-SEP').sum()[0:23]

#Total Releases
sum(trinity['Total'].resample('A-SEP').sum()[0:23])

#Monthly Distribution of releases
trinity_filtered = trinity['2001':'2023']
trinity_filtered.groupby(trinity_filtered.index.month)['Total'].sum().groupby(level=0).median()






