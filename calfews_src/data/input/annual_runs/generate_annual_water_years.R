#______________________________________________________________________________#
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

for(wy in 1:length(unique_years)){
  print(unique_years[wy])
  
  #Subset to that year
  wy_data <- data %>% filter(water_year == unique_years[wy])
  wy_data$water_year <- NULL
  
  #Subset to rows which end in _fnf (This is for has_full_inputs = FALSE, in the base_inflows.json file)
  #fnf_cols <- grep("_fnf$", names(wy_data), value = TRUE)
  #wy_data <- wy_data[, c("datetime", fnf_cols)]
  
  print(dim(wy_data))
  
  #Create the output file name 
  output_file <- paste0("WY_", unique_years[wy], ".csv")

  # Write the filtered data to a new CSV file
  write.csv(wy_data, output_file, row.names = FALSE)
}