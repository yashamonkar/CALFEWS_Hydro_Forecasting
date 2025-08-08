# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:31:45 2024

@author: amonkar
"""


#Data Source
#https://www.eia.gov/electricity/data/eia923/

import pandas as pd
import os
import xlrd

# Set the working directory
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)



# %% ----------------LOOP BETWEEN 2011 and 2023--------------------------------

#Column Names
subset_columns = ['YEAR', 'Plant Name', 'YEAR', 'Netgen\nJanuary', 'Netgen\nFebruary', 'Netgen\nMarch', 'Netgen\nApril', 'Netgen\nMay',  'Netgen\nJune', 'Netgen\nJuly', 'Netgen\nAugust',  'Netgen\nSeptember',  'Netgen\nOctober', 'Netgen\nNovember',  'Netgen\nDecember']                                                       
subset_columns_an = ['YEAR', 'Plant Name', 'YEAR', 'Netgen_Jan', 'Netgen_Feb', 'Netgen_Mar', 'Netgen_Apr', 'Netgen_May',  'Netgen_Jun', 'Netgen_Jul', 'Netgen_Aug',  'Netgen_Sep',  'Netgen_Oct', 'Netgen_Nov',  'Netgen_Dec']                                                       
subset_plants = ['Shasta', 'Folsom', 'New Melones', 'Keswick', 'Judge F Carr', 'Trinity', 'W R Gianelli', 'Nimbus', 'ONeill', 'Spring Creek']
#subset_plants = ['Shasta', 'Folsom', 'New Melones', 'Keswick', 'ONeill', 'Judge F Carr', 'Trinity']


# Initialize an empty DataFrame to hold the concatenated data
concatenated_df_a = pd.DataFrame()

for year in range(2011, 2024):
    #Set-up the counter.
    print(year)
    
    # Define the file path for the current year
    file_path = f'Yash/EIA/{year}.xlsx'
    
    # Read the Excel file, skipping the first 5 rows and using the 6th row as the header
    df = pd.read_excel(file_path, header=5)
    
    # Filter the DataFrame based on the 'Plant Name' column
    filtered_df = df[df['Plant Name'].isin(subset_plants)]
    
    # Check if the year is 2011 or 2013
    if year in [2011, 2013]:
        # Subset the filtered DataFrame to the specified columns for 2011 and 2013
        subset_df = filtered_df[subset_columns_an]
    else:
        # Subset the filtered DataFrame to the specified columns for other years
        subset_df = filtered_df[subset_columns]
    
    # Change the column names to subset_columns for both cases
    subset_df.columns = subset_columns
    
    # Concatenate the subset_df for the current year with the concatenated DataFrame
    concatenated_df_a = pd.concat([concatenated_df_a, subset_df], ignore_index=True)

# Display the concatenated DataFrame
concatenated_df_a = concatenated_df_a.iloc[:, 1:]
print(concatenated_df_a)






# %% ----------------LOOP BETWEEN 2011 and 2023--------------------------------

#Column Names
subset_columns_an = ['Year', 'Plant Name', 'Year', 'NETGEN_JAN', 'NETGEN_FEB', 'NETGEN_MAR', 'NETGEN_APR', 'NETGEN_MAY',  'NETGEN_JUN', 'NETGEN_JUL', 'NETGEN_AUG',  'NETGEN_SEP',  'NETGEN_OCT', 'NETGEN_NOV',  'NETGEN_DEC']                                                       


# Initialize an empty DataFrame to hold the concatenated data
concatenated_df_b = pd.DataFrame()

for year in range(2003, 2011):
    #Set-up the counter.
    print(year)
    
    # Define the file path for the current year
    file_path = f'Yash/EIA/{year}.xls'
    
    # Read the Excel file, skipping the first 5 rows and using the 6th row as the header
    df = pd.read_excel(file_path, header=7)
    
    # Filter the DataFrame based on the 'Plant Name' column
    filtered_df = df[df['Plant Name'].isin(subset_plants)]
    
    # Check if the year is 2011 or 2013
    if year in [2011, 2013]:
        # Subset the filtered DataFrame to the specified columns for 2011 and 2013
        subset_df = filtered_df[subset_columns_an]
    else:
        # Subset the filtered DataFrame to the specified columns for other years
        subset_df = filtered_df[subset_columns_an]
    
    # Change the column names to subset_columns for both cases
    subset_df.columns = subset_columns
    
    # Concatenate the subset_df for the current year with the concatenated DataFrame
    concatenated_df_b = pd.concat([concatenated_df_b, subset_df], ignore_index=True)

# Display the concatenated DataFrame
concatenated_df_b = concatenated_df_b.iloc[:, 1:]
print(concatenated_df_b)


# %%


# Concatenate both sections
concatenated_df = pd.concat([concatenated_df_b, concatenated_df_a], 
                            ignore_index=True)

#concatenated_df = concatenated_df.dropna(axis=0, how='any')

# We will create a new DataFrame for the time series
time_series_data = {
    'Year-Month': [],
    'Folsom': [],
    'Shasta': [],
    'New Melones': [], 
    'Keswick': [],
    'Judge F Carr': [],
    'Trinity': [],
    'W R Gianelli': [],
    'Nimbus': [],
    'ONeill': [],
    'Spring Creek': []
}

# Define the months to help create the 'Year-Month' labels
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Iterate through each unique year in the DataFrame
for year in concatenated_df['YEAR'].unique():
    # Filter the DataFrame for the current year
    year_df = concatenated_df[concatenated_df['YEAR'] == year]
    
    # For each month, add the corresponding generation values to the respective plant in the new DataFrame
    for month_idx, month in enumerate(months):
        # Create the 'Year-Month' label
        year_month = f"{year}-{month_idx + 1:02d}"
        time_series_data['Year-Month'].append(year_month)
        
        # For each plant, get the generation value and add it to the list
        for plant in subset_plants:
            # Find the row for the plant and the year
            plant_row = year_df[year_df['Plant Name'] == plant]
            
            # There should be only one row per plant per year, so we can safely access the value
            if not plant_row.empty:
                gen_value = plant_row.iloc[0][f'Netgen\n{month}']
                time_series_data[plant].append(gen_value)
            else:
                # If there's no data for that plant and year, we can append NaN or some placeholder
                time_series_data[plant.capitalize()].append(pd.NA)

# Create the final time series DataFrame
time_series_df = pd.DataFrame(time_series_data)

# Set 'Year-Month' as the index
time_series_df.set_index('Year-Month', inplace=True)

# Display the transformed DataFrame
print(time_series_df)

#Save to output
time_series_df.to_csv('Yash/EIA/EIA_Monthy_Gen.csv', index = True)


