# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:43:15 2024

@author: amonkar
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)

#Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
### Load the data

calfews_new = pd.read_csv('cord-sim_realtime.csv', index_col=0)
calfews_old = pd.read_csv('calfews_src/data/input/calfews_src-data.csv', index_col=0)

#Convert index to datetime
calfews_new.index = pd.to_datetime(calfews_new.index)
calfews_old.index = pd.to_datetime(calfews_old.index)

# %%
def compare_column(df1, df2, column_name, start_date, end_date):
    
    # Subset to the specified column
    series1 = df1[column_name]
    series2 = df2[column_name]  
    
    # Filter date range
    series1 = series1.loc[(series1.index >= start_date) & (series1.index <= end_date)]
    series2 = series2.loc[(series2.index >= start_date) & (series2.index <= end_date)]
    
    # Combine the two series into a single DataFrame
    combined_df = pd.DataFrame({
        'calfews_src-data': series1,
        'cord-sim_realtime': series2
    })
    
    # Drop rows with NaN values
    combined_df.dropna(inplace=True)
    
    # Check if series are identical after dropping NaNs
    is_identical = combined_df['calfews_src-data'].equals(combined_df['cord-sim_realtime'])
    print(f"Are the series identical? {'Yes' if is_identical else 'No'}")
    
    # Calculate maximum difference
    max_diff = (combined_df['calfews_src-data'] - combined_df['cord-sim_realtime']).abs().max()
    print(f"Maximum difference between corresponding values: {max_diff}")
    
    # Calculate correlation
    correlation = combined_df['calfews_src-data'].corr(combined_df['cord-sim_realtime'])
    
    # Main time series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(combined_df.index, combined_df['calfews_src-data'], label='calfews_src-data')
    ax.plot(combined_df.index, combined_df['cord-sim_realtime'], label='cord-sim_realtime')
    ax.set_title(f'Comparison of {column_name} \n (Max Diff: {max_diff:.4f})', fontsize = 20)
    ax.set_xlabel('Date', fontsize = 20)
    ax.set_ylabel('Value', fontsize = 20)
    ax.legend(loc = 'upper left')
    
    # Add correlation text to the main plot
    ax.text(0.05, 0.85, f'Correlation: {correlation:.2f}', 
            transform=ax.transAxes, fontsize=15, 
            verticalalignment='top')

    # Create an inset scatter plot with a 45-degree line
    ax_inset = fig.add_axes([0.65, 0.35, 0.25, 0.35])  # [left, bottom, width, height]
    ax_inset.scatter(combined_df['calfews_src-data'], combined_df['cord-sim_realtime'], alpha=0.5)
    
    # Plot the 45-degree line
    min_val = min(combined_df.min().min(), combined_df.max().max())
    max_val = max(combined_df.min().min(), combined_df.max().max())
    ax_inset.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)

    # Add labels and a grid to the inset plot
    ax_inset.set_xlabel('calfews_src-data')
    ax_inset.set_ylabel('cord-sim_realtime')
    ax_inset.grid(True)
    
    plt.show()
    
    
# %%
# ORO, SHA, YRS, FOL, 
start_date = '1996-10-01'
end_date = '2016-09-30'
column_name = 'SHA_storage'
df1 = calfews_old
df2 = calfews_new

compare_column(calfews_old, calfews_new, column_name, start_date, end_date)
