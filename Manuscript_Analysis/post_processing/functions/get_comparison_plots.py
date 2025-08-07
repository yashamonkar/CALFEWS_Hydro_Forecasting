# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:32:43 2025

@author: amonkar

Code to plot the re
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_comparison_plots(model_output, data_input, labels, y_axis_label, units,\
                         plot_legends, monthly_method='sum'):
    """
    Create comparison plots between model output and input data, showing both daily and monthly analyses.
    
    Parameters:
    -----------
    model_output : pandas.DataFrame
        DataFrame containing model output with datetime index
    data_input : pandas.DataFrame
        DataFrame containing observed data with datetime index
    labels : list
        List of labels for each column
    y_axis_label : str
        Label for y-axis
    monthly_method : str
        Method for monthly aggregation ('sum' or 'average')
    """
    
    # Validate inputs
    if not all(model_output.columns == data_input.columns):
        raise ValueError("Model output and data input must have the same columns")
    if len(labels) != len(model_output.columns):
        raise ValueError("Number of labels must match number of columns")
    if monthly_method not in ['sum', 'average']:
        raise ValueError("monthly_method must be either 'sum' or 'average'")

    # Calculate number of rows needed for subplots
    n_cols = min(3, len(model_output.columns))
    n_rows = int(np.ceil(len(model_output.columns) / 3))

    # Create daily plots
    plt.figure(figsize=(25, 5*n_rows))
    fig = plt.figure(figsize=(25, 5*n_rows + 0.5))  
    for i, col in enumerate(model_output.columns):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Plot daily data
        plt.plot(model_output.index, model_output[col], 'r-')
        plt.plot(data_input.index, data_input[col], 'b-')
        
        # Calculate correlation
        correlation = stats.pearsonr(model_output[col].values, data_input[col].values)[0]
        
        # Set title and labels
        plt.title(f"{labels[i]} {y_axis_label}\nR = {round(correlation, 2)}", 
                  fontsize = 24)
        plt.ylabel(f"{y_axis_label} ({units})", fontsize = 22)
        plt.xlabel('Date', fontsize = 22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) 
    lines = fig.axes[0].get_lines()  
    fig.legend(lines, [plot_legends[0], plot_legends[1]], 
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, fontsize=24)
    
    # Create monthly plots
    # Convert to monthly data
    if monthly_method == 'sum':
        monthly_model = model_output.resample('M').sum()
        monthly_data = data_input.resample('M').sum()
    else:  # average
        monthly_model = model_output.resample('M').mean()
        monthly_data = data_input.resample('M').mean()
    
    plt.figure(figsize=(25, 5*n_rows))
    fig = plt.figure(figsize=(25, 5*n_rows + 1))  
    for i, col in enumerate(monthly_model.columns):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Plot monthly data
        plt.plot(monthly_model.index, monthly_model[col], 'r-')
        plt.plot(monthly_data.index, monthly_data[col], 'b-')
        
        # Calculate correlation for monthly data
        correlation = stats.pearsonr(monthly_model[col].values, 
                                   monthly_data[col].values)[0]
        
        # Set title and labels
        plt.title(f"{labels[i]} {y_axis_label} \n(Monthly) R = {round(correlation, 2)}", 
                  fontsize = 24)
        plt.ylabel(f"{y_axis_label} ({units})", fontsize = 22)
        plt.xlabel('Date', fontsize = 22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) 
    lines = fig.axes[0].get_lines()  
    fig.legend(lines, [plot_legends[0], plot_legends[1]], 
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, fontsize=24)
    
   
    
    # Display plots
    plt.show()



