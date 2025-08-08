# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 20:46:49 2025

@author: amonkar

Figure 3 - Script to plot Trinity Restoration flow based on ROD 2001. 
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# Read the CSV file
df = pd.read_csv('Yash\Individual_Meeting_Scripts\Manuscript_Plots\Trinity_Flow_Schedule.csv') #Location to be changed.

# Set the 'Type' column as index
df.set_index('Type', inplace=True)

# Create a mapping for the flow types to their descriptions
flow_type_map = {
    'EW': 'Extremely Wet',
    'W': 'Wet',
    'N': 'Normal',
    'D': 'Dry',
    'DC': 'Critically Dry'
}

# Create a function to convert column names to datetime objects
def column_to_date(col_name):
    if 'Unnamed' in col_name:
        return None
    try:
        # Parse the date from format like "1-Oct"
        day, month = col_name.split('-')
        
        # Map month abbreviations to numbers
        month_map = {
            'Oct': 10, 'Nov': 11, 'Dec': 12,
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
            'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9
        }
        
        # Determine the correct year (spanning Oct to Sep of the next year)
        year = 2023 if month_map[month] >= 10 else 2024
        
        return datetime(year, month_map[month], int(day))
    except:
        return None

# Create a list of dates for the x-axis
dates = [column_to_date(col) for col in df.columns]
valid_dates = [d for d in dates if d is not None]

# Create a new DataFrame with dates as columns
date_df = pd.DataFrame()
for i, flow_type in enumerate(df.index):
    values = df.loc[flow_type].values[:len(valid_dates)]
    date_df[flow_type_map[flow_type]] = values

date_df.index = valid_dates

# Plot the data
plt.figure(figsize=(12, 8))

# Colors for the different flow types
colors = {
    'Extremely Wet': 'navy',
    'Wet': 'blue',
    'Normal': 'lightblue',
    'Dry': '#FF7F7F',
    'Critically Dry': 'red'
}

# Plot each flow type
for flow_type in date_df.columns:
    plt.plot(date_df.index, date_df[flow_type], label=flow_type, color=colors[flow_type], linewidth=2)

# Add vertical grid lines
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='-', alpha=0.3)

# Add horizontal grid lines with light gray color
plt.grid(True, axis='y', color='lightgray')

# Set labels and title
plt.xlabel('Day', fontsize=18)
plt.ylabel('Restoration Flow Releases (CFS)', fontsize=18)

# Format the x-axis to show month names
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, fontsize = 14)

# Set y-axis limits to match the reference image
plt.ylim(0, 15000)
plt.yticks(np.arange(0, 17500, 2500), fontsize = 14)

# Add annotations for peak flows
peak_flow = date_df['Extremely Wet'].max()
peak_date = date_df.index[date_df['Extremely Wet'].argmax()]
plt.annotate(f'Peak flow: {peak_flow} cfs for 5 days',
             xy=(peak_date, peak_flow),
             xytext=(peak_date + timedelta(days=10), peak_flow + 1000),
             arrowprops=dict(arrowstyle='->'), fontsize = 14)

# Add a legend
plt.legend(loc='upper left', ncol=1, fontsize = 18)

# Adjust layout to make room for axis labels
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)  # Make room for the legend

# Save the plot
#plt.savefig('trinity_flow_schedule_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()