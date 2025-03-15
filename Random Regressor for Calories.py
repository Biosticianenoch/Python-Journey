#!/usr/bin/env python
# coding: utf-8

# In[8]:


## Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# In[9]:


## Load the dataset
df = pd.read_csv("C:/Users/PC/OneDrive/Desktop/Data Science/Datasets/Datasets/calories.csv")


# In[10]:


## Inspect the few observations of the dataset
df.head()


# In[11]:


## Assess the structure of the dataset
df.info()


# In[13]:


## Check for duplicates
df.duplicated().sum()


# In[14]:


## Check for missing values
df.isnull().sum()


# In[16]:


## Summary statistics
df = df.drop(columns = ["User_ID"])
df.describe()


# In[17]:


## Perform correlation analysis
numeric_vars = df.select_dtypes(include = ["float64", "int64"])
correlation_matrix = numeric_vars.corr()
print(correlation_matrix)


# In[18]:


## Visualize your correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[19]:


## Label encoding
## Select categorical columns
categorical_cols = df.select_dtypes(include = ["object"]).columns
## Initialize the label encoder
label_encoder = LabelEncoder()
## Apply label encooding to selected columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[21]:


## Self-Correlation
plt.figure(figsize=(8,6))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, mask=mask)
plt.title("Correlation Heatmap")
plt.show()


# In[22]:


## Define Independent and Dependent Variables
X = df.drop(columns = ["Calories"])
y = df["Calories"] 


# In[23]:


## Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


## Train the Random Forest Model
## Create the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
## Train the model
rf_model.fit(X_train, y_train)


# In[25]:


## Make Predictions
y_pred = rf_model.predict(X_test)


# In[26]:


## Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared Score: {r2}")


# In[27]:


## Feature importance analysis
feature_importance = rf_model.feature_importances_

# Convert to DataFrame
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance in Random Forest")
plt.show()


# In[28]:


## Visualizing predictions
## Plot Actual vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()


# In[29]:


## Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color='red', linestyle='dashed')
plt.title("Residuals Distribution")
plt.show()


# In[30]:


## Safe the model
joblib.dump(rf_model, "random_forest_model.pkl")

