#______________________________________________________________________________#
#Beginning Storage -- Dry Conditions -- Beginning Oct 1st 2015
#Code to generate input files for each Water Year - Annual Run
#Main input is the comprehensive water year file

#Set working directory to file location
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

#Load the libraries
library(dplyr)

#Read the main CALFEWS input file
data <- read.csv("cord-sim_realtime.csv")

# Convert datetime to proper date format
data$datetime <- as.Date(data$datetime, format = "%m/%d/%Y")


data$water_year <- ifelse(as.numeric(format(data$datetime, "%m")) >= 10,
                          as.numeric(format(data$datetime, "%Y")) + 1,
                          as.numeric(format(data$datetime, "%Y")))

unique_years <- unique(data$water_year)
unique_years <- head(unique_years, -4) #Remove the last 

#Get the dateimte values
datetime_og <- data %>% filter(water_year %in% 2016:(2020))
write.csv(datetime_og, "5yr_sim_inflow.csv", row.names = FALSE)

datetime_vals <- datetime_og$datetime

for(t in 1:length(unique_years)){
  #print(unique_years[t])
  
  #Assign water year
  wy = unique_years[t]
  
  #Subset to that year
  wy_data <- data %>% filter(water_year %in% wy:(wy+4))
  wy_data$water_year <- NULL
  
  #Subset to rows which end in _fnf (This is for has_full_inputs = FALSE, in the base_inflows.json file)
  #fnf_cols <- grep("_fnf$", names(wy_data), value = TRUE)
  #wy_data$datetime <- NULL 
  #wy_data <- wy_data[, c("datetime", fnf_cols)]
  #wy_data <- wy_data[, c(fnf_cols)]
  
  if(nrow(wy_data) == 1826){
    last_row <- wy_data[nrow(wy_data),]
    wy_data <- rbind(wy_data, last_row, make.row.names = FALSE)
  }
  
  wy_data$datetime <- datetime_vals
  print(dim(wy_data)[1])
  
  #Create the output file name 
  output_file <- paste0("WY_", wy, ".csv")

  # Write the filtered data to a new CSV file
  write.csv(wy_data, output_file, row.names = FALSE)
}

