# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:45:03 2024

@author: amonkar
"""


from PyPDF2 import PdfFileReader
from tabula.io import read_pdf
import camelot
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np

#Set the working directory
# Using absolute path
os.chdir('C:/Users/amonkar/Documents/CALFEWS_Preliminary')


# Initialize empty list to store dataframes
all_dfs = []

# Generate the file name patterns
dates = pd.date_range(start='2000-10-01', end='2024-10-01', freq='M')
file_patterns = [d.strftime('%m%y') for d in dates]
print(file_patterns)

# Define column orders for different time periods
columns_before_aug_2022 = ['Day', 'Elev', 'Storage', 'Storage_Change', 'Inflow_CFS',
                          'Power', 'Outlet', 'Spill', 'Diversion', 'Evap_cfs', 'Evap_inches']

columns_after_aug_2022 = ['Day', 'Elev', 'Storage', 'Storage_Change', 'Inflow_CFS',
                         'Power', 'Spill', 'Outlet', 'Diversion', 'Evap_cfs', 'Evap_inches']

#Start the loop
for file_pattern in file_patterns:
    filename = f"data/CVP_Reports/Lewiston/lewdop{file_pattern}.pdf"
    print(f"Processing {filename}")
    
    # Extract month and year from file pattern
    month = int(file_pattern[:2])
    year = int('20' + file_pattern[2:])
    current_date = pd.Timestamp(year=year, month=month, day=1)
    
    # Try reading with camelot first
    tables = camelot.read_pdf(filename, pages='1', flavor='stream')
    
    if len(tables) > 0:
        df = tables[0].df
    else:
        # Fallback to tabula if camelot fails
        print(f"Using Tabula for {filename}")
        df = read_pdf(filename, pages='1')[0]
    
    #Data Cleanup
    if (df.iloc[:,0] == 'TOTALS').any():
        df = df.iloc[:df[df.iloc[:,0] == 'TOTALS'].index[0]].reset_index(drop=True)
    
    # Inside the loop: Check for '1' and handle month days
    if (df.iloc[:,0] == '1').any():
        df = df.iloc[df[df.iloc[:,0] == '1'].index[0]:].reset_index(drop=True)
    else:
        print("Back-Calculating Days")
        # Get number of days in the current month
        days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
        # Keep only the last N rows where N is the number of days in the month
        df = df.tail(days_in_month).reset_index(drop=True)
        # Add day numbers (1 to number of days in month) in the first column
        df.iloc[:,0] = range(1, days_in_month + 1)
        #Convert to string for consistency
        df.iloc[:,0] = df.iloc[:,0].astype(str)
    
    df = df.loc[:, (df != '').any(axis=0)]
    
    # Choose appropriate column order based on date
    if current_date < pd.Timestamp('2022-08-01'):
        columns = columns_before_aug_2022
    else:
        columns = columns_after_aug_2022
    
    if len(df.columns) == len(columns):
        df.columns = columns
        df = df.reindex(columns=columns_before_aug_2022)  # Use pre-Aug 2022 order as standard
    else:
        print(f"Incorrect Number of Columns for {filename}")
    
    # Clean numeric values
    for col in columns_before_aug_2022:  # Use consistent column list for cleaning
        df[col] = df[col].str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create proper dates for the index
    dates = [pd.Timestamp(year=year, month=month, day=int(d)) for d in df['Day']]
    df.index = pd.DatetimeIndex(dates)
    
    # Add month info
    df['Month'] = dates[0].strftime('%B')  # Full month name
    df['Year'] = year
    
    # Append to list
    all_dfs.append(df)

# Combine all dataframes
final_df = pd.concat(all_dfs, ignore_index=False)  # Keep the date index
# Sort by index to ensure chronological order
final_df = final_df.sort_index()
#Save the file. 
final_df.to_csv('data/Lewiston_Daily_Operations.csv')

#_____________________________________________________________________________#
### Initial Visualization of the Diversions. 
fig = plt.figure(figsize=(18,6))
ax = plt.subplot(122)
ax.plot(final_df['Diversion'][0:365], label='Diversions', linestyle='solid')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Diversions (CFS/day)')


#Annual Water Year Diversions
plt.figure(figsize=(15, 8))
colors = LinearSegmentedColormap.from_list('custom_red_blue', ['#8B0000', '#00008B'])
lines = []
labels = []

def day_of_water_year(date):
    if date.month >= 10:
        return (date - pd.Timestamp(year=date.year, month=10, day=1)).days + 1
    else:
        return (date - pd.Timestamp(year=date.year-1, month=10, day=1)).days + 1

# Create arrays to store values for each day of water year
daily_values = [[] for _ in range(366)]

# First pass: collect all values for each day of water year
for year in range(2001, 2025):
    mask = ((final_df.index.month >= 10) & (final_df.index.year == year)) | \
           ((final_df.index.month < 10) & (final_df.index.year == year + 1))
    water_year_data = final_df['Diversion'][mask]
    
    if not water_year_data.empty:
        for date, value in water_year_data.items():
            day = day_of_water_year(date) - 1  # 0-based index
            daily_values[day].append(value)

# Calculate percentiles for each day
days = range(1, 367)
percentile_low = []
percentile_high = []
for day_values in daily_values:
    if day_values:
        percentile_low.append(np.percentile(day_values, 5))
        percentile_high.append(np.percentile(day_values, 95))
    else:
        percentile_low.append(np.nan)
        percentile_high.append(np.nan)

# Plot the confidence interval
plt.fill_between(days, percentile_low, percentile_high, 
                color='gray', alpha=0.9, label='90% Interval')

# Plot individual lines with increased transparency
for year in range(2001, 2025):
    mask = ((final_df.index.month >= 10) & (final_df.index.year == year)) | \
           ((final_df.index.month < 10) & (final_df.index.year == year + 1))
    water_year_data = final_df['Diversion'][mask]
    
    if not water_year_data.empty:
        x_values = [day_of_water_year(date) for date in water_year_data.index]
        color = colors((year - 2001) / 24)
        line, = plt.plot(x_values, water_year_data.values, 
                        color=color, linewidth=1, alpha=0.4)  # Reduced alpha and linewidth
        lines.append(line)
        labels.append(f'WY {year}')

plt.title('Trinity Diversions (WY 2001-2024)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Diversions (CFS)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(lines, labels, title='Water Year', bbox_to_anchor=(1.05, 1), 
           loc='upper left', borderaxespad=0.)

month_positions = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
               'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
plt.xticks(month_positions, month_labels)
plt.tight_layout()
plt.show()
