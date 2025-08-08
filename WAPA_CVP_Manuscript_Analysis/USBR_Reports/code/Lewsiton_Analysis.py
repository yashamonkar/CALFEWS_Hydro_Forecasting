# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:43:07 2024

@author: amonkar

#Code to understand Lewiston Outflows

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
lewiston = pd.read_csv("data/Lewiston_Daily_Operations.csv", index_col = 0)
lewiston.index = pd.DatetimeIndex(lewiston.index)

#Filter Lewiston to correct time periods
lewiston = lewiston[lewiston.index < pd.Timestamp('2023-10-01')]

#Hyperparameters
cfs_to_tafd = 1.98211*10**-3


#_____________________________________________________________________________#
#Compute the Restoration flows
lewiston['Restoration'] = lewiston['Power'] + lewiston['Outlet'] + lewiston['Spill'] 

#Convert to Acre-Feet
lewiston['Restoration'] = lewiston['Restoration']*cfs_to_tafd
lewiston['Diversion'] = lewiston['Diversion']*cfs_to_tafd


#%%
def plot_water_year_releases(data, title='Water Year Releases', y_limit=30):
    """
    Plot water year releases from a pandas Series with datetime index.
    
    Parameters:
    data (pd.Series): Time series data with datetime index
    title (str): Plot title
    y_limit (float): Y-axis limit
    """
    def day_of_water_year(date):
        if date.month >= 10:
            return (date - pd.Timestamp(year=date.year, month=10, day=1)).days + 1
        return (date - pd.Timestamp(year=date.year-1, month=10, day=1)).days + 1
    
    # Setup
    plt.figure(figsize=(15, 8))
    colors = LinearSegmentedColormap.from_list('custom_red_blue', ['#8B0000', '#00008B'])
    lines, labels = [], []
    
    # Calculate daily values and medians
    daily_values = {day: [] for day in range(1, 367)}
    start_year = data.index.year.min()
    end_year = data.index.year.max()
    
    for year in range(start_year, end_year + 1):
        mask = ((data.index.month >= 10) & (data.index.year == year)) | \
               ((data.index.month < 10) & (data.index.year == year + 1))
        water_year_data = data[mask]
        
        if not water_year_data.empty:
            for date, value in water_year_data.items():
                day = day_of_water_year(date)
                daily_values[day].append(value)
    
    daily_medians = {day: np.median(values) for day, values in daily_values.items() if values}
    
    # Plot each water year
    for year in range(start_year, end_year + 1):
        mask = ((data.index.month >= 10) & (data.index.year == year)) | \
               ((data.index.month < 10) & (data.index.year == year + 1))
        water_year_data = data[mask]
        
        if not water_year_data.empty:
            x_values = [day_of_water_year(date) for date in water_year_data.index]
            color = colors((year - start_year) / (end_year - start_year))
            line, = plt.plot(x_values, water_year_data.values, color=color, 
                           linewidth=1.5, alpha=0.8)
            lines.append(line)
            labels.append(f'WY {year}')
    
    # Plot median line
    median_line, = plt.plot(list(daily_medians.keys()), list(daily_medians.values()), 
                           'r-', linewidth=2.5, label='Median')
    lines.append(median_line)
    labels.append('Median')
    
    # Formatting
    plt.title(f'{title} Total = {data.sum()/1000:.2f} MAF', fontsize=18)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Releases (TAF)', fontsize=12)
    plt.ylim(0, y_limit)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(lines, labels, title='Water Year', bbox_to_anchor=(1.05, 1), 
              loc='upper left', borderaxespad=0.)
    
    month_positions = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
                   'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    plt.xticks(month_positions, month_labels)
    plt.tight_layout()
    
    return plt.gcf()


# %% Basin Visualization

#Total Diversion Flows
print(lewiston['Diversion'].resample('YE-SEP').sum())
sum(lewiston['Diversion'].resample('YE-SEP').sum())

#Total Restoration Flows
print(lewiston['Restoration'].resample('YE-SEP').sum())
sum(lewiston['Restoration'].resample('YE-SEP').sum())

#Total Outflows
lewiston['Total_Outflows'] = lewiston['Diversion'] + lewiston['Restoration']
print(lewiston['Total_Outflows'].resample('YE-SEP').sum())
sum(lewiston['Total_Outflows'].resample('YE-SEP').sum())


#Create the plots
plot_water_year_releases(lewiston['Restoration'], title='Restoration Flows')
plot_water_year_releases(lewiston['Diversion'], title='Diversion Flows')
plot_water_year_releases(lewiston['Total_Outflows'], title='Total OutFlows')


#%%
#Additional Analysis
monthly_diversions = lewiston['Diversion'].resample('ME').sum()
monthly_fractions = 100*monthly_diversions.groupby(monthly_diversions.index.month).sum()/monthly_diversions.sum()

#Volume diverted in Mar-Oct
monthly_fractions[3:10]/monthly_fractions[3:10].sum()