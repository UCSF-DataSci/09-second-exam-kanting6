#!/bin/bash

# Step 1: Generate dirty data file
python3 generate_dirty_data.py

# Step 2: Remove comment lines and empty lines
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' > ms_data_cleaned.csv

# Step 3: Remove extra commas
sed 's/,,*/,/g' ms_data_cleaned.csv > ms_data_no_extra_commas.csv

# Step 4: Extract essential columns
cut -d',' -f1,2,3,4,5 ms_data_no_extra_commas.csv > ms_data_extracted.csv

# Step 5: Filter rows by walking speed between 2.0 and 8.0
awk -F',' 'NR==1 || ($5 >= 2.0 && $5 <= 8.0)' ms_data_extracted.csv > ms_data.csv

# Step 6: Create insurance type list
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst

# Step 7: Generate summary of the processed data
echo "Total visits:"
tail -n +2 ms_data.csv | wc -l
echo "First few records:"
head -n 5 ms_data.csv
