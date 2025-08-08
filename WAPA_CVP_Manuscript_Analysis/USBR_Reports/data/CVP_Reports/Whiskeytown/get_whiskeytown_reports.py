# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:47:51 2024

@author: amonkar

Code to download the CVP Reports for the Whiskeytown Reservoir.

Note: The code is only set-up right now to download to Oct-2000. 
Files exists upto Jan-1998 but need some cleaning up.  
"""

import pandas as pd
import requests
import os
from datetime import datetime
import time

#Set the working directory and relative path
os.chdir('C:/Users/amonkar/Documents/CALFEWS_Preliminary')
script_dir = ('data/CVP_Reports/Whiskeytown')


# Generate the file name patterns
dates = pd.date_range(start='2000-10-01', end='2024-10-01', freq='M')
file_patterns = [d.strftime('%m%y') for d in dates]
print(file_patterns)


#_____________________________________________________________________________#
###Deleting earlier files.
#Be Careful This deletes all earlier .pdf files

# List all files in directory
files = os.listdir(script_dir)

# Delete all files except .py files
for file in files:
    if not file.endswith('.py'):
        file_path = os.path.join(script_dir, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")



#_____________________________________________________________________________#
# Initialize counters for logging
successful_downloads = 0
failed_downloads = []

# Download files
for pattern in file_patterns:
    
    filename = f"whidop{pattern}.pdf"
    url = f"https://www.usbr.gov/mp/cvo/vungvari/{filename}"
    file_path = os.path.join(script_dir, filename)
    
    try:
        # Add a small delay to prevent overwhelming the server
        time.sleep(1)
        
        # Send GET request
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the file if request was successful
        with open(file_path, "wb") as file:
            file.write(response.content)
        
        print(f"Successfully downloaded: {filename}")
        successful_downloads += 1
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}: {str(e)}")
        failed_downloads.append((filename, str(e)))
        continue

# Print summary
print("\nDownload Summary:")
print(f"Total files attempted: {len(file_patterns)}")
print(f"Successfully downloaded: {successful_downloads}")
print(f"Failed downloads: {len(failed_downloads)}")

if failed_downloads:
    print("\nFailed Downloads Details:")
    for filename, error in failed_downloads:
        print(f"- {filename}: {error}")


