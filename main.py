# Step 1: File Upload and Data Loading
from google.colab import files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Upload the file manually
uploaded = files.upload()

# Load the CSV after uploading
data = pd.read_csv(list(uploaded.keys())[0])
print(data.head())
print(data.info())

# Basic Stats
print(data.describe())

# Correlation Matrix (only numeric columns)
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Value Counts for Categorical columns
print(data['city'].value_counts())
print(data['job'].value_counts())

# Age Grouping
data['age_group'] = pd.cut(data['age'], bins=[18, 30, 40, 50, 60, 70], labels=['18-30', '30-40', '40-50', '50-60', '60-70'])

# Income Binning
data['income_band'] = pd.cut(data['income'], bins=5, labels=['Low', 'Lower-Mid', 'Middle', 'Upper-Mid', 'High'])

# Income Distribution
sns.histplot(data['income'], kde=True)
plt.title('Income Distribution')
plt.show()

# Age vs Job Distribution
sns.countplot(x='age_group', hue='job', data=data)
plt.title('Job Distribution by Age Group')
plt.show()

# City Distribution
data['city'].value_counts().plot(kind='bar')
plt.title('City Distribution')
plt.show()