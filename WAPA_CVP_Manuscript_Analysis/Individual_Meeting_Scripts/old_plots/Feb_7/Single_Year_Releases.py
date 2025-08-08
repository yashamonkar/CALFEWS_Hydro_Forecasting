import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# First, let's modify get_comparison_plots slightly to return the figures instead of showing them
def modified_get_comparison_plots(model_output, data_input, labels, y_axis_label, units,
                                plot_legends, monthly_method='sum'):
    # Copy the entire function but remove plt.show() at the end
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
    daily_fig = plt.figure(figsize=(25, 5*n_rows + 0.5))  
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
    lines = daily_fig.axes[0].get_lines()  
    daily_fig.legend(lines, [plot_legends[0], plot_legends[1]], 
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, fontsize=24)

    # Create monthly plots
    if monthly_method == 'sum':
        monthly_model = model_output.resample('M').sum()
        monthly_data = data_input.resample('M').sum()
    else:  # average
        monthly_model = model_output.resample('M').mean()
        monthly_data = data_input.resample('M').mean()
    
    plt.figure(figsize=(25, 5*n_rows))
    monthly_fig = plt.figure(figsize=(25, 5*n_rows + 1))  
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
    lines = monthly_fig.axes[0].get_lines()  
    monthly_fig.legend(lines, [plot_legends[0], plot_legends[1]], 
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, fontsize=24)
    
    return daily_fig, monthly_fig

# Create PDF file
with PdfPages('water_year_plots.pdf') as pdf:
    # Get the range of years
    start_year = calfews_releases.index[0].year
    end_year = calfews_releases.index[-1].year

    # Loop through each water year
    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}-{year+1}")
        
        # Define water year start and end
        start_date = f"{year}-10-01"
        end_date = f"{year+1}-09-30"
        
        # Filter data for this water year
        mask = (calfews_releases.index >= start_date) & (calfews_releases.index <= end_date)
        year_calfews = calfews_releases[mask]
        year_true = true_releases[mask]
        
        labels = [f'{col} WY {year+1}' for col in year_true.columns]
        
        # Create plots and get the figures
        daily_fig, monthly_fig = modified_get_comparison_plots(
            model_output=year_calfews,
            data_input=year_true,
            labels=labels,
            y_axis_label="Releases",
            units='TAF',
            plot_legends=['CALFEWS Releases', 'CDEC Releases'],
            monthly_method='sum'
        )
        
        # Save both figures
        pdf.savefig(daily_fig)
        pdf.savefig(monthly_fig)
        
        # Close figures
        plt.close(daily_fig)
        plt.close(monthly_fig)
        
        print(f"Saved plots for year {year}-{year+1}")

print("All water year plots have been saved to water_year_plots.pdf")