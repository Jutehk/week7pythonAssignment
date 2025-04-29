import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --- Task 1: Load and Explore the Dataset ---

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())
print("\n" + "="*50 + "\n")

# Explore the structure of the dataset
print("Dataset information:")
df.info()
print("\n" + "="*50 + "\n")

# Check for missing values
print("Missing values:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

# Clean the dataset (handling missing values if any)
# In the Iris dataset, there are no missing values, so no cleaning is strictly needed.
# However, for demonstration, if there were missing values, you might do something like:
# df_cleaned = df.dropna()  # To drop rows with any missing values
# OR
# df_cleaned = df.fillna(df.mean()) # To fill missing values with the mean

df_cleaned = df.copy() # Create a copy to work with, even though no cleaning needed for Iris

print("Cleaned dataset information:")
df_cleaned.info()
print("\n" + "="*50 + "\n")

# --- Task 2: Basic Data Analysis ---

# Compute basic statistics of numerical columns
print("Basic statistics of numerical columns:")
print(df_cleaned.describe())
print("\n" + "="*50 + "\n")

# Perform groupings on a categorical column ('target' representing species)
# and compute the mean of numerical columns for each group
mean_measurements_per_species = df_cleaned.groupby('target').mean()
print("Mean measurements per Iris species:")
print(mean_measurements_per_species)
print("\n" + "="*50 + "\n")

# Identify patterns or interesting findings
print("Interesting Findings:")
print("- Sepal length appears to be generally larger than sepal width across all species.")
print("- Setosa species tends to have smaller petal length and petal width compared to versicolor and virginica.")
print("- Virginica generally has the largest petal length and petal width.")
print("\n" + "="*50 + "\n")

# --- Task 3: Data Visualization ---

# 1. Line chart showing trends over time (Not directly applicable to the Iris dataset as it's not time-series data)
# For demonstration purposes, let's create a hypothetical time-series-like plot
# by sorting the DataFrame by sepal length and plotting it against the index.
plt.figure(figsize=(10, 6))
plt.plot(df_cleaned['sepal length (cm)'].sort_values().values, label='Sepal Length')
plt.title('Hypothetical Trend of Sepal Length')
plt.xlabel('Data Point Index (Sorted by Sepal Length)')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n(Line chart showing a hypothetical trend)\n")

# 2. Bar chart showing the comparison of a numerical value across categories
plt.figure(figsize=(8, 6))
sns.barplot(x=iris.target_names, y=mean_measurements_per_species['petal length (cm)'])
plt.title('Average Petal Length per Iris Species')
plt.xlabel('Iris Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()
print("\n(Bar chart comparing average petal length across species)\n")

# 3. Histogram of a numerical column to understand its distribution
plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()
print("\n(Histogram showing the distribution of sepal width)\n")

# 4. Scatter plot to visualize the relationship between two numerical columns
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df_cleaned, palette='viridis')
plt.title('Relationship between Sepal Length and Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species', labels=iris.target_names)
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n(Scatter plot showing the relationship between sepal and petal length)\n")

print("\n--- End of Analysis and Visualizations ---")