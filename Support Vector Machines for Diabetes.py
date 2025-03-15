#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# In[2]:


## Load the dataset
df = pd.read_csv("C:/Users/PC/OneDrive/Desktop/Data Science/Datasets/Datasets/Diabetes.csv")


# In[3]:


## Inspect the first few observations of the dataset
df.head(10)


# In[4]:


## Assess the structure of the dataset
df.info()


# In[5]:


## Check for missing values
df.isnull().sum()


# In[7]:


## Check for duplicates
df.duplicated().sum()


# In[8]:


## Check target class distribution
print(df["Outcome"].value_counts())


# In[9]:


# Visualize class distribution
sns.countplot(x=df['Outcome'])
plt.title("Class Distribution in Target Variable")
plt.xticks([0, 1], labels = ["Negative", "Positive"])
plt.show()


# In[10]:


## Summary statistics
df.describe()


# In[19]:


## Define Features and Target variable
X = df.drop(columns = ["Outcome"])
y = df["Outcome"] 


# In[20]:


## Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the shape
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# In[21]:


## Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[22]:


## Train the SVM Model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)


# In[23]:


## Make Predictions
y_pred = svm_model.predict(X_test_scaled)


# In[24]:


## Evaluate Model Performance
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[25]:


# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[28]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[30]:


## Feature importance
# Train an SVM with a linear kernel
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

# Get feature importance (absolute value of coefficients)
feature_importance = np.abs(svm_linear.coef_).mean(axis=0)

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance (Linear SVM)')
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Features')
plt.show()


# In[31]:


# Save the trained model
joblib.dump(best_svm, 'svm_model.pkl')

print("SVM model saved successfully!")

