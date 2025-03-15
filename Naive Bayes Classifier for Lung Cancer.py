#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler


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


## Check for duplicates
df.duplicated().sum()


# In[7]:


## Check for missing values
df.isnull().sum()


# In[8]:


## ## Check target class distribution
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


# In[11]:


## Define Features and Target variable
X = df.drop(columns = ["Outcome"])
y = df["Outcome"] 


# In[12]:


## Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the shape
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# In[14]:


## Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[15]:


## Train the Naïve Bayes Model
# Initialize the classifier
nb_classifier = GaussianNB()

# Train (fit) the model
nb_classifier.fit(X_train, y_train)


# In[16]:


## Make Predictions
y_pred = nb_classifier.predict(X_test)


# In[17]:


## Evaluate the Model
## Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[18]:


## Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[19]:


## Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




