# -*- coding: utf-8 -*-
"""
This script reads the hourly generation data provided by WAPA/Eric
The output is a text file which contains the daily production across the time period. 

@author: amonkar
"""


#Data Source
#https://www.eia.gov/electricity/data/eia923/

import pandas as pd
import os
import xlrd
import matplotlib.pyplot as plt

# Set the working directory
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS\Yash'
os.chdir(working_directory)


# %%


#Provide path to the Excel Sheet. 
file_path = 'Pi-Gen-DATA.xlsx'


# Initialize an empty list to store each year's daily_totals DataFrame
all_years_totals = []

# Loop through each year from 2015 to 2023
for year in range(2016, 2024):
    
    #Set-up the counter.
    print(year)
    
    # Read the sheet for the current year
    data = pd.read_excel(file_path, sheet_name=str(year))
    
    # Create a subset dataset
    data_subset = pd.DataFrame({
        'Time': data['LOCAL_DATETIME']
    })

    # Calculate the total generation for each location
    data_subset['Judge F Carr'] = data['CAR_1'] + data['CAR_2']
    data_subset['Folsom'] = data['FOL_1'] + data['FOL_2'] + data['FOL_3']
    data_subset['Keswick'] = data['KES_1'] + data['KES_2'] + data['KES_3']
    data_subset['Nimbus'] = data['NIM_1'] + data['NIM_2']
    data_subset['New Melones'] = data['NML_1'] + data['NML_2']
    data_subset['Spring Creek'] = data['SCR_1'] + data['SCR_2']
    data_subset['Shasta'] = data['SHA_1'] + data['SHA_2'] + data['SHA_3'] + data['SHA_4'] + data['SHA_5']
    data_subset['Trinity'] = data['TRN_1'] + data['TRN_2']

    # Convert 'Time' to datetime and set it as the index
    data_subset['Time'] = pd.to_datetime(data_subset['Time'])
    data_subset.set_index('Time', inplace=True)

    # Resample to get daily sums
    daily_totals = data_subset.resample('D').sum()

    # Append the result to the list
    all_years_totals.append(daily_totals)

# Concatenate all DataFrames into one
final_totals = pd.concat(all_years_totals)

#Plot a single generation 
plt.figure(figsize=(10, 5))  # Set the figure size for better visibility
plt.plot(final_totals.index, final_totals['Folsom'], label='Folsom')
plt.title('Daily Totals of Folsom Over Time')  # Title of the plot
plt.xlabel('Date')  # Label for the x-axis
plt.ylabel('Total')  # Label for the y-axis
plt.legend()  # Add a legend
plt.grid(True)  # Show grid lines for better readability


#Save to output
final_totals.to_csv('WAPA/WAPA_Daily_Gen.csv', index = True)




