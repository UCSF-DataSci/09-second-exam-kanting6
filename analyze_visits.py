import pandas as pd
import numpy as np
import random

# Load the processed CSV file and the insurance types
ms_data = pd.read_csv('/Users/kankantingting/09-second-exam-kanting6/ms_data.csv')
insurance_types = pd.read_csv('/Users/kankantingting/09-second-exam-kanting6/insurance.lst', header=None, names=['insurance_type'])

# Convert visit_date to datetime
ms_data['visit_date'] = pd.to_datetime(ms_data['visit_date'])

# Sort data by patient_id and visit_date
ms_data = ms_data.sort_values(by=['patient_id', 'visit_date']).reset_index(drop=True)

# Step 1: Assign insurance type consistently per patient
# Create a mapping of each patient_id to a random insurance type
unique_patients = ms_data['patient_id'].unique()
insurance_choices = insurance_types['insurance_type'].tolist()
insurance_mapping = {pid: random.choice(insurance_choices) for pid in unique_patients}

# Add the insurance_type column to ms_data based on patient_id
ms_data['insurance_type'] = ms_data['patient_id'].map(insurance_mapping)

# Step 2: Generate visit costs based on insurance type
# Define base costs and variations for each insurance type
insurance_costs = {
    'Basic': 100,
    'Premium': 150,
    'Platinum': 200
}

# Add visit_cost column by applying base cost plus random variation
ms_data['visit_cost'] = ms_data['insurance_type'].apply(lambda x: insurance_costs.get(x, 100) + np.random.normal(0, 20))

# Step 3: Calculate summary statistics
# Mean walking speed by education level
mean_walking_speed_by_education = ms_data.groupby('education_level')['walking_speed'].mean()

# Mean costs by insurance type
mean_costs_by_insurance = ms_data.groupby('insurance_type')['visit_cost'].mean()

# Correlation between age and walking speed
age_speed_corr = ms_data[['age', 'walking_speed']].corr().iloc[0, 1]

# Handle any missing data
ms_data = ms_data.dropna()

# Step 4: Display summary statistics
print("Mean walking speed by education level:")
print(mean_walking_speed_by_education)

print("\nMean visit costs by insurance type:")
print(mean_costs_by_insurance)

print("\nCorrelation between age and walking speed:")
print(age_speed_corr)
