#!/usr/bin/env python
# coding: utf-8

# In[31]:


## Import Required Libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from sklearn import tree   
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# In[32]:


## Load the Dataset
df = pd.read_csv("C:/Users/PC/OneDrive/Desktop/Data Science/Datasets/Datasets/heart.csv")


# In[33]:


## Inspect the first view observations of the dataset
df.head(10)


# In[34]:


## Assessing the structure of the dataset
df.info()


# In[35]:


## Checking for missing values
df.isnull().sum()


# In[36]:


## Check for duplicates
df.duplicated().sum()


# In[37]:


## Remove duplicates
df = df.drop_duplicates()


# In[38]:


## ## Check the unique entries of the outcome variable
print(df["target"].unique())


# In[39]:


## Check target class distribution
print(df["target"].value_counts())


# In[40]:


## Summary statistics
df.describe()


# In[41]:


# Visualize class distribution
sns.countplot(x=df['target'])
plt.title("Class Distribution in Target Variable")
plt.xticks([0, 1], labels = ["Negative", "Positive"])
plt.show()


# In[42]:


## Define Features and Target variable
X = df.drop(columns = ["target"])
y = df["target"] 


# In[43]:


## Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# In[44]:


## Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[45]:


## Train a Decision Tree Model
# Initialize the model
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
dt_model.fit(X_train, y_train)


# In[46]:


## Make Predictions on Test Data
# Predict
y_pred = dt_model.predict(X_test)

# Display predictions
print(y_pred)


# In[47]:


## Evaluate the Model Performance
# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[48]:


# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[49]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[50]:


## Feature Importance Plot
# Get feature importance scores
feature_importance = dt_model.feature_importances_
feature_names = X.columns  # Feature names
# Create a DataFrame for better visualization
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Sort features by importance
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Tree")
plt.show()


# In[51]:


# Save the model to a file
joblib.dump(dt_model, "decision_tree_model.pkl")

print("Model saved successfully!")

