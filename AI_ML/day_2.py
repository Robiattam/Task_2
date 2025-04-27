
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('C:\\project_py\\Titanic-Dataset.csv')
print("Summary Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())
numeric_features = ['Age', 'Fare']

for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()

for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

selected_features = ['Survived', 'Pclass', 'Age', 'Fare', 'Sex']
sns.pairplot(data[selected_features].dropna(), hue='Survived', diag_kind='kde')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.show()


