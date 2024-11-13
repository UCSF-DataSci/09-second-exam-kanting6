import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import scipy.stats as stats
from scipy.stats import linregress
from scipy.stats import f_oneway

from statsmodels.formula.api import mixedlm
import statsmodels.formula.api as smf
import statsmodels.api as sm

ms_data = pd.read_csv('/Users/kankantingting/09-second-exam-kanting6/ms_data.csv')
insurance_data = pd.read_csv('insurance.lst')
ms_data['insurance_type'] = insurance_data['insurance_type']

# 1. Analyze walking speed:
# Preprocessing
ms_data['education_level'] = ms_data['education_level'].astype('category')
ms_data['education_code'] = ms_data['education_level'].cat.codes  # Encode education level

# Step 1: Mixed-Effects Model
# Formula: walking_speed ~ age + education_code, grouping by patient_id
model = mixedlm("walking_speed ~ age + education_code", ms_data, groups=ms_data["patient_id"])
result = model.fit()

# Extract results
coefficients = result.params
conf_intervals = result.conf_int()

print("Mixed-Effects Model Results")
print("Coefficients:")
print(coefficients)
print("\nConfidence Intervals:")
print(conf_intervals)

# Step 2: Linear Trend Test with scipy.stats
# Test for the significant trend between age and walking speed
slope, intercept, r_value, p_value, std_err = linregress(ms_data['age'], ms_data['walking_speed'])

print("\nLinear Trend Analysis:")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Standard Error: {std_err:.4f}")

# Step 3: Residual Analysis
ms_data['predicted'] = result.fittedvalues
ms_data['residuals'] = ms_data['walking_speed'] - ms_data['predicted']

# Residuals vs Fitted Values
plt.figure(figsize=(12, 6))
sns.scatterplot(x=ms_data['predicted'], y=ms_data['residuals'], alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(12, 6))
sns.histplot(ms_data['residuals'], kde=True, bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# QQ-plot for residuals
sm.qqplot(ms_data['residuals'], line='s')
plt.title('QQ Plot of Residuals')
plt.show()

# Scatterplot of Walking Speed vs Age with Regression Line
plt.figure(figsize=(12, 6))
sns.regplot(x='age', y='walking_speed', data=ms_data, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
plt.title('Walking Speed vs Age with Regression Line')
plt.xlabel('Age')
plt.ylabel('Walking Speed')
plt.show()

#2. Simple analysis of insurance type effect

# Define the insurance costs
insurance_costs = {
    "Basic": 100,
    "Premium": 150,
    "Platinum": 200
}

# Load the dataset 
file_path = "insurance.lst"  
insurance_data = pd.read_csv(file_path, sep="\t") 

# Map insurance types to their respective costs
insurance_data['insurance_cost'] = insurance_data['insurance_type'].map(insurance_costs)

# Basic statistics for each insurance type
basic_stats = insurance_data.groupby('insurance_type')['insurance_cost'].agg(
    mean_cost=np.mean,
    median_cost=np.median,
    std_dev=np.std,
    count='count'
).reset_index()

print("Basic Statistics by Insurance Type:")
print(basic_stats)

# ANOVA to test the effect of insurance type on costs
groups = [insurance_data.loc[insurance_data['insurance_type'] == ins_type, 'insurance_cost']
          for ins_type in insurance_costs.keys()]
f_stat, p_value = f_oneway(*groups)

print("\nANOVA Results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4e}")

# Effect size: Eta squared (η²)
ss_between = sum(len(group) * (np.mean(group) - np.mean(insurance_data['insurance_cost']))**2 for group in groups)
ss_total = sum((insurance_data['insurance_cost'] - np.mean(insurance_data['insurance_cost']))**2)
eta_squared = ss_between / ss_total

print("\nEffect Size:")
print(f"Eta squared (η²): {eta_squared:.4f}")

plt.figure(figsize=(10, 6))
sns.boxplot(x='insurance_type', y='insurance_cost', data=insurance_data, palette="Set2")
plt.title("Insurance Costs by Type")
plt.xlabel("Insurance Type")
plt.ylabel("Insurance Cost")
plt.show()

# 2. Box plots and basic statistics of 'age' by 'insurance_type'
plt.figure(figsize=(10, 6))
sns.boxplot(data=ms_data, x='insurance_type', y='age')
plt.title("Age Distribution by Insurance Type")
plt.xlabel("Insurance Type")
plt.ylabel("Age")
plt.show()

# 3. Education age interaction effects on walking speed

# Encode education_level as a categorical variable
ms_data['education_level'] = ms_data['education_level'].astype('category')
ms_data['education_level_code'] = ms_data['education_level'].cat.codes

# Define the regression formula with interaction term for age and education level
formula = 'walking_speed ~ age * education_level_code'

# Fit the linear regression model
model = smf.ols(formula, data=ms_data).fit()

# Extract results
coefficients = model.params
conf_intervals = model.conf_int()
p_values = model.pvalues
summary = model.summary()

# Control for relevant confounders (assuming they exist in the dataset and adding them here)
# If, for example, 'gender' is a confounder, add it to the formula as follows:
# formula_confounded = 'walking_speed ~ age * education_level_code + gender'
# model_confounded = smf.ols(formula_confounded, data=data).fit()

# Conducting statistical tests 
age_walking_speed_corr, age_walking_speed_p = stats.pearsonr(data['age'], data['walking_speed'])

regression_results = {
    'coefficients': coefficients,
    'confidence_intervals': conf_intervals,
    'p_values': p_values,
    'summary': summary,
    'age_walking_speed_correlation': age_walking_speed_corr,
    'age_walking_speed_p_value': age_walking_speed_p
}



# Report key statistics and p-values
insurance_groups = [ms_data['age'][ms_data['insurance_type'] == ins] for ins in ms_data['insurance_type'].unique()]
anova_result = stats.f_oneway(*insurance_groups)

print("ANOVA Test for Age across Insurance Types:")
print(f"F-statistic: {anova_result.statistic}")
print(f"p-value: {anova_result.pvalue}")
