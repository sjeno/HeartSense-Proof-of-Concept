#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # to load and manipulate data and for one hot encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt #to draw graphs
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree #to draw a classification tree
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.model_selection import cross_val_score #for cross validation
from sklearn.metrics import confusion_matrix #to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score


ud = pd.read_csv(r'C:\Users\User\Desktop\heart_disease_uci.csv')


# In[2]:


# Display first five rows
ud.head()


# In[3]:


ud.info()


# In[4]:


ud['num'].value_counts()


# In[5]:


ud.groupby('sex').mean()


# # Dealing with missing values

# In[6]:


import pandas as pd

# Define the mappings
sex_mapping = {'Male': 1, 'Female': 0}
cp_mapping = {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3}
fbs_mapping = {True: 1, False: 0}
restecg_mapping = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
exang_mapping = {True: 1, False: 0}
slope_mapping = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
thal_mapping = {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
num_mapping = {0: 'no heart disease', 1: 'mild heart disease', 2: 'moderate heart disease', 3: 'severe heart disease', 4: 'very severe heart disease'}

# Convert categorical columns to numerical representations
ud['sex'] = ud['sex'].map(sex_mapping).fillna(ud['sex'])
ud['cp'] = ud['cp'].map(cp_mapping).fillna(ud['cp'])
ud['fbs'] = ud['fbs'].map(fbs_mapping).fillna(ud['fbs'])
ud['restecg'] = ud['restecg'].map(restecg_mapping).fillna(ud['restecg'])
ud['exang'] = ud['exang'].map(exang_mapping).fillna(ud['exang'])
ud['slope'] = ud['slope'].map(slope_mapping).fillna(ud['slope'])
ud['thal'] = ud['thal'].map(thal_mapping).fillna(ud['thal'])

# Convert columns to numeric data type
ud = ud.apply(pd.to_numeric, errors='ignore')

# Remove 'dataset' and 'id' columns
ud = ud.drop(['dataset', 'id'], axis=1)

# Print the updated DataFrame
print(ud)


# In[7]:


# Display first five rows
ud.head()


# In[8]:


# Check for missing values
missing_values = ud.isnull().sum()

# Print the count of missing values
print(missing_values)


# In[9]:


# Check for missing values
udmissing_values_mask = ud.isna()
udmissing_values_description = ud[udmissing_values_mask].apply(lambda x: ', '.join(x.unique().astype(str)))

# Print the description of missing values
print(udmissing_values_description)


# # UD columns dropped with high value of missing values

# In[10]:


# Drop 'slope', 'ca', and 'thal' columns
ud = ud.drop(['slope', 'ca', 'thal'], axis=1)

# Print the updated DataFrame
print(ud)


# # UD Filling in remaining missing values

# In[11]:


# Fill missing values in numeric columns with mean
ud['trestbps'].fillna(ud['trestbps'].mean(), inplace=True)
ud['chol'].fillna(ud['chol'].mean(), inplace=True)
ud['thalach'].fillna(ud['thalach'].mean(), inplace=True)
ud['oldpeak'].fillna(ud['oldpeak'].mean(), inplace=True)

# Fill missing values in categorical columns with mode
ud['fbs'].fillna(ud['fbs'].mode()[0], inplace=True)
ud['restecg'].fillna(ud['restecg'].mode()[0], inplace=True)
ud['exang'].fillna(ud['exang'].mode()[0], inplace=True)

# Assign default value to missing values in ordinal columns
ud['sex'].fillna(-1, inplace=True)
ud['cp'].fillna(-1, inplace=True)

# Print the updated DataFrame
print(ud)


# In[12]:


# Check for string values in the DataFrame
has_strings = ud.select_dtypes(include=['object']).columns.tolist()

if len(has_strings) > 0:
    print("The following columns contain string values:")
    print(has_strings)
else:
    print("No columns contain string values.")


# In[13]:


# Check for missing values
missing_values = ud.isnull().sum()

# Print the count of missing values
print(missing_values)


# # Changing num variable to only two results (0 no heart disease, 1 heart disease)

# In[14]:


ud['num'] = ud['num'].apply(lambda x: 1 if x > 0 else 0)


# In[15]:


print(ud['num'].unique())


# # Check for outliers 

# # UD Age 

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=ud['age'])
plt.show()


# In[17]:


age_stats = ud['age'].describe()
print(age_stats)


# To identify outliers in the dataset, we can use the interquartile range (IQR) method. Based on the information you provided, here's how you can identify potential outliers:
# 
# Calculate the IQR (Interquartile Range):
# IQR = 75th percentile - 25th percentile
# 
# Determine the lower bound:
# Lower bound = 25th percentile - 1.5 * IQR
# 
# Determine the upper bound:
# Upper bound = 75th percentile + 1.5 * IQR
# 
# Any value below the lower bound or above the upper bound can be considered a potential outlier.
# 
# Based on the statistics you provided:
# 
# Lower bound: 47 - 1.5 * (60 - 47) = 27.5
# Upper bound: 60 + 1.5 * (60 - 47) = 79.5
# Therefore, any value below 27.5 or above 79.5 can be considered a potential outlier in this dataset.

# In[18]:


lower_bound = 27.5
upper_bound = 79.5

outliers = ud[(ud['age'] < lower_bound) | (ud['age'] > upper_bound)]

if outliers.empty:
    print("There are no outliers in the 'age' feature.")
else:
    print("There are outliers in the 'age' feature.")


# Sex is binary ie no outliers

# cp' appears to be a categorical feature with values ranging from 0 to 3, it is not appropriate to consider outliers in this case.

# # UD trestbps 

# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=ud['trestbps'])
plt.show()


# In[20]:


age_stats = ud['trestbps'].describe()
print(age_stats)


# Calculate the IQR (Interquartile Range):
# IQR = 75th percentile - 25th percentile
# 
# Determine the lower bound:
# Lower bound = 25th percentile - 1.5 * IQR
# 
# Determine the upper bound:
# Upper bound = 75th percentile + 1.5 * IQR 
# 
# Lower bound: 120 - 1.5 * (140 - 120) = 90
# Upper bound: 140 + 1.5 * (140 - 120) = 170

# In[21]:


lower_bound = 90
upper_bound = 170

outliers = ud[(ud['trestbps'] < lower_bound) | (ud['trestbps'] > upper_bound)]

if outliers.empty:
    print("There are no outliers in the 'trestbps' feature.")
else:
    print("There are outliers in the 'trestbps' feature.")


# There are outliers so now the amount is calculated

# In[22]:


lower_bound = 90
upper_bound = 170

outliers = ud[(ud['trestbps'] < lower_bound) | (ud['trestbps'] > upper_bound)]

num_outliers = len(outliers)
total_count = len(ud)

print("Number of outliers in the 'trestbps' feature:", num_outliers)
print("Total count of data points:", total_count)


# In[23]:


percentage_outliers = (num_outliers / total_count) * 100
print("Percentage of outliers in the 'trestbps' feature:", percentage_outliers)


#  This means that out of the total count of data points in the 'trestbps' feature, around 3.04% fall outside the specified range and are considered outliers.

# The outliers are removed

# In[24]:


lower_bound = 90
upper_bound = 170

ud = ud[(ud['trestbps'] >= lower_bound) & (ud['trestbps'] <= upper_bound)]


# In[25]:


lower_bound = 90
upper_bound = 170

outliers = ud[(ud['trestbps'] < lower_bound) | (ud['trestbps'] > upper_bound)]

num_outliers = len(outliers)
total_count = len(ud)

print("Number of outliers in the 'trestbps' feature:", num_outliers)
print("Total count of data points:", total_count)


# # UD Chol

# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=ud['chol'])
plt.show()


# In[27]:


age_stats = ud['chol'].describe()
print(age_stats)


# In[28]:


import numpy as np

feature_name = 'chol'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = ud[feature_name].quantile(0.25)
Q3 = ud[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = ud[(ud[feature_name] < lower_outlier_bound) | (ud[feature_name] > upper_outlier_bound)]

total_count = len(ud[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")


# This is a very high number of outliers so this column wil most likely have to be removed

# # UD Fbs

# fbs is binary ie no outliers

# # UD restecg

# restecg is mapped and has no outliers

# # UD Thalach

# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=ud['thalach'])
plt.show()


# In[30]:


import numpy as np

feature_name = 'thalach'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = ud[feature_name].quantile(0.25)
Q3 = ud[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = ud[(ud[feature_name] < lower_outlier_bound) | (ud[feature_name] > upper_outlier_bound)]

total_count = len(ud[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# remove outliers

# In[31]:


feature_name = 'thalach'
lower_bound = 66
upper_bound = 210

outliers = ud[(ud[feature_name] < lower_bound) | (ud[feature_name] > upper_bound)]

num_outliers = len(outliers)
total_count = len(ud)

print("Number of outliers in the", feature_name, "feature:", num_outliers)
print("Total count of data points:", total_count)


# In[32]:


feature_name = 'thalach'
lower_bound = 66
upper_bound = 210

# Identify the outliers
outliers = ud[(ud[feature_name] < lower_bound) | (ud[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
ud = ud[~((ud[feature_name] < lower_bound) | (ud[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(ud)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # UD Exang

# no outliers binary

# # UD Oldpeak

# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=ud['oldpeak'])
plt.show()


# In[34]:


import numpy as np

feature_name = 'oldpeak'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = ud[feature_name].quantile(0.25)
Q3 = ud[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = ud[(ud[feature_name] < lower_outlier_bound) | (ud[feature_name] > upper_outlier_bound)]

total_count = len(ud[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# In[35]:


feature_name = 'oldpeak'
lower_bound = -2.25
upper_bound = 3.75

outliers = ud[(ud[feature_name] < lower_bound) | (ud[feature_name] > upper_bound)]

num_outliers = len(outliers)
total_count = len(ud)

print("Number of outliers in the", feature_name, "feature:", num_outliers)
print("Total count of data points:", total_count)


# In[36]:


feature_name = 'oldpeak'
lower_bound = -2.25
upper_bound = 3.75

# Identify the outliers
outliers = ud[(ud[feature_name] < lower_bound) | (ud[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
ud = ud[~((ud[feature_name] < lower_bound) | (ud[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(ud)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # UD Num

# no ouliers as mapped

# # Saving dataframes after removing outliers

# In[37]:


# Save pd DataFrame as CSV
ud.to_csv('pd_after_outlier_removal.csv', index=False)



# # Number of rows for both UD

# In[38]:


# Print the number of data points remaining for each feature in ud DataFrame
for col in ud.columns:
    data_points_remaining = ud[col].value_counts().sum()
    print("Number of data points remaining for feature", col, "in ud DataFrame:", data_points_remaining)


# # Data Cleaning and PreprocessingUD

# In[39]:


# Load the saved dataframes
ud = pd.read_csv('pd_after_outlier_removal.csv')


# Display the first 5 rows of each dataframe to understand their structure
print("ud DataFrame:")
print(ud.head())


# Print the column names from each dataframe
print("\nColumns in ud:")
print(ud.columns)


# Display the shape of each dataframe
print("\nShape of ud:")
print(ud.shape)


# In[40]:


# For ud dataset
print("Number of NaN values in ud dataset:")
print(ud.isna().sum())


# # Gradient Boosting Classifier

# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Split the ud dataset into features and target variable
X_ud = ud.drop('num', axis=1)
y_ud = ud['num']



# Split ud dataset into training and test sets
X_train_ud, X_test_ud, y_train_ud, y_test_ud = train_test_split(X_ud, y_ud, test_size=0.2, random_state=42)



# Train the model on ud dataset
model_ud = GradientBoostingClassifier(random_state=42)
model_ud.fit(X_train_ud, y_train_ud)


# Make predictions on the test set for ud dataset
predictions_ud = model_ud.predict(X_test_ud)

# Print classification reports
print("UD dataset classification report:")
print(classification_report(y_test_ud, predictions_ud))


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming 'ud' is your dataframe and 'target' is your target variable
X = ud.drop('num', axis=1)  # Replace 'target' with the actual target column name in your dataset
y = ud['num']  # Replace 'target' with the actual target column name in your dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [LogisticRegression(), 
          DecisionTreeClassifier(), 
          RandomForestClassifier(), 
          GradientBoostingClassifier(), 
          SVC(),
          KNeighborsClassifier()]

model_names = ['Logistic Regression', 
               'Decision Tree', 
               'Random Forest', 
               'Gradient Boosting', 
               'SVC', 
               'KNeighbors']

precision_scores = []
recall_scores = []
f1_scores = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

# Print the scores for each model
for i in range(len(models)):
    print(f"Model: {model_names[i]}")
    print(f"Precision: {precision_scores[i]}")
    print(f"Recall: {recall_scores[i]}")
   


# #  tuning the n_estimators and max_depth parameters for your Random Forest model:

# In[43]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

# Initialize a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit it to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)


# #  initialize, fit and evaluate model:

# In[44]:


# Initialize the Random Forest classifier with the best parameters
best_rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)

# Fit the model on the training data
best_rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = best_rf.predict(X_test)

# Print the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))

from joblib import dump, load

# Save the model
dump(best_rf, 'random_forest.joblib') 

#save csv file
ud.to_csv('final_heart_disease_uci.csv', index=False)


# # Load new data further validate the model on this new unseen data to ensure its robustness and check if it generalizes well

# In[45]:


nd = pd.read_csv(r'C:\Users\User\Desktop\heart_statlog_cleveland_hungary_final.csv')


# In[46]:


# Display first five rows
nd.head()


# In[47]:


import pandas as pd

# Read the CSV file and load the data into a DataFrame
nd = pd.read_csv(r'C:\Users\User\Desktop\heart_statlog_cleveland_hungary_final.csv')

# Update the column names in the DataFrame to the new names
nd.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'num']

# Print the updated column names to verify
print(nd.columns)


# In[48]:


# Display first five rows
nd.head()


# In[49]:


# Check for missing values
missing_values = nd.isnull().sum()

# Print the count of missing values
print(missing_values)


# In[50]:


nd.groupby('sex').mean()


# In[51]:


import pandas as pd

# Define the mappings
sex_mapping = {'Male': 1, 'Female': 0}
cp_mapping = {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3}
fbs_mapping = {True: 1, False: 0}
restecg_mapping = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
exang_mapping = {True: 1, False: 0}
slope_mapping = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
num_mapping = {0: 'no heart disease', 1: 'mild heart disease', 2: 'moderate heart disease', 3: 'severe heart disease', 4: 'very severe heart disease'}

# Convert categorical columns to numerical representations
nd['sex'] = nd['sex'].map(sex_mapping).fillna(nd['sex'])
nd['cp'] = nd['cp'].map(cp_mapping).fillna(nd['cp'])
nd['fbs'] = nd['fbs'].map(fbs_mapping).fillna(nd['fbs'])
nd['restecg'] = nd['restecg'].map(restecg_mapping).fillna(nd['restecg'])
nd['exang'] = nd['exang'].map(exang_mapping).fillna(nd['exang'])
nd['slope'] = nd['slope'].map(slope_mapping).fillna(nd['slope'])

# Convert columns to numeric data type
nd = nd.apply(pd.to_numeric, errors='ignore')


# Print the updated DataFrame
print(nd)


# In[52]:


# Display first five rows
nd.head()


# # Check for outliers 

# # ND Age

# In[53]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['age'])
plt.show()


# In[54]:


import numpy as np

feature_name = 'age'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# # ND trestbps 

# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['trestbps'])
plt.show()


# In[56]:


import numpy as np

feature_name = 'trestbps'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# 
# lower_bound = 90.0
# upper_bound = 170.0
# 
# outliers = ud[(ud[feature_name] < lower_bound) | (ud[feature_name] > upper_bound)]
# 
# num_outliers = len(outliers)
# total_count = len(ud)
# 
# print("Number of outliers in the", feature_name, "feature:", num_outliers)
# print("Total count of data points:", total_count)
# 

# In[57]:


feature_name = 'trestbps'
lower_bound = 90.0
upper_bound = 170.0
# Identify the outliers
outliers = nd[(nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
nd = nd[~((nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(nd)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # ND CP

# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['cp'])
plt.show()


# In[59]:


import numpy as np

feature_name = 'cp'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# In[60]:


feature_name = 'cp'
lower_bound = 1.5
upper_bound = 5.5
# Identify the outliers
outliers = nd[(nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
nd = nd[~((nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(nd)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # ND chol

# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['chol'])
plt.show()


# In[62]:


import numpy as np

feature_name = 'chol'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# In[63]:


feature_name = 'chol'
lower_bound = 112.5
upper_bound = 380.5
# Identify the outliers
outliers = nd[(nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
nd = nd[~((nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(nd)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # ND fbs

# In[64]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['fbs'])
plt.show()


# In[65]:


import numpy as np

feature_name = 'fbs'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# In[66]:


feature_name = 'fbs'
lower_bound = 0
upper_bound = 0
# Identify the outliers
outliers = nd[(nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
nd = nd[~((nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(nd)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # ND restecg

# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['restecg'])
plt.show()


# In[68]:


import numpy as np

feature_name = 'restecg'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# # ND thalach

# In[69]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['thalach'])
plt.show()


# In[70]:


import numpy as np

feature_name = 'thalach'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# In[71]:


feature_name = 'thalach'
lower_bound = 80.25
upper_bound = 218.25
# Identify the outliers
outliers = nd[(nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
nd = nd[~((nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(nd)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # ND exang

# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['exang'])
plt.show()


# In[73]:


import numpy as np

feature_name = 'exang'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# # ND oldpeak

# In[74]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['oldpeak'])
plt.show()


# In[75]:


import numpy as np

feature_name = 'oldpeak'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# In[76]:


feature_name = 'oldpeak'
lower_bound = -2.4000000000000004
upper_bound = 4.0
# Identify the outliers
outliers = nd[(nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound)]

# Remove the outliers from the DataFrame
nd = nd[~((nd[feature_name] < lower_bound) | (nd[feature_name] > upper_bound))]

# Update the total count after removing outliers
total_count = len(nd)

print("Number of outliers removed from the", feature_name, "feature:", len(outliers))
print("Total count of data points after removing outliers:", total_count)


# # ND slope

# In[77]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=nd['slope'])
plt.show()


# In[78]:


import numpy as np

feature_name = 'slope'
lower_bound_factor = 1.5
upper_bound_factor = 1.5

Q1 = nd[feature_name].quantile(0.25)
Q3 = nd[feature_name].quantile(0.75)

IQR = Q3 - Q1

lower_outlier_bound = Q1 - lower_bound_factor * IQR
upper_outlier_bound = Q3 + upper_bound_factor * IQR

outliers = nd[(nd[feature_name] < lower_outlier_bound) | (nd[feature_name] > upper_outlier_bound)]

total_count = len(nd[feature_name])
outlier_count = len(outliers)
percentage_outliers = (outlier_count / total_count) * 100

print("Number of outliers in", feature_name, ":", outlier_count, "outliers out of total", total_count)
print("Percentage of outliers in", feature_name, ":", percentage_outliers, "%")
print (upper_outlier_bound)
print (lower_outlier_bound)


# # Saving ND dataframes after removing outliers

# In[79]:


# Save nd DataFrame as CSV
nd.to_csv('nd_after_outlier_removal.csv', index=False)



# # Number of rows for both UD

# In[80]:


# Print the number of data points remaining for each feature in nd DataFrame
for col in nd.columns:
    data_points_remaining = nd[col].value_counts().sum()
    print("Number of data points remaining for feature", col, "in nd DataFrame:", data_points_remaining)


# In[81]:


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import load

# Load the original model from disk
best_rf = load('random_forest.joblib')

# Load the new data (nd DataFrame) from the CSV file
new_data = pd.read_csv('nd_after_outlier_removal.csv')

# List of columns to be dropped from the new data
columns_to_drop = ['slope']

# Drop the specified columns from the DataFrame
new_data = new_data.drop(columns=columns_to_drop)

# Assuming 'presence' is the target label column
X_new = new_data.drop('num', axis=1)
y_new_actual = new_data['num']

# Make predictions on the new data using the loaded model
y_new_pred = best_rf.predict(X_new)

# Calculate the accuracy of the model on the new data
accuracy = accuracy_score(y_new_actual, y_new_pred)
print("Accuracy on new data:", accuracy)

# Print the classification report
print("Classification Report on new data:")
print(classification_report(y_new_actual, y_new_pred))

# Print the confusion matrix
print("Confusion Matrix on new data:")
print(confusion_matrix(y_new_actual, y_new_pred))


# In[82]:


import pandas as pd

# Load the original model from disk
best_rf = load('random_forest.joblib')

# Load the new data from the CSV file
new_data = pd.read_csv('nd_after_outlier_removal.csv')

# Display basic information about the new data
print("New Data Info:")
print(new_data.info())

# Display summary statistics for numerical columns
print("\nSummary Statistics for Numerical Columns:")
print(new_data.describe())

# Compare the distributions of numerical features between the original and new data
original_data = pd.read_csv(r'C:\Users\User\Desktop\statlog.csv')  # Replace 'original_data.csv' with the file name of the original dataset used for training
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for feature in numerical_features:
    print("\nDistribution of", feature)
    print("Original Data:")
    print(original_data[feature].describe())
    print("New Data:")
    print(new_data[feature].describe())

# Compare the distributions of categorical features between the original and new data
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang']

for feature in categorical_features:
    print("\nDistribution of", feature)
    print("Original Data:")
    print(original_data[feature].value_counts(normalize=True))
    print("New Data:")
    print(new_data[feature].value_counts(normalize=True))

# Visualize the distributions of numerical features (e.g., using histograms or box plots)
import matplotlib.pyplot as plt

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.hist(original_data[feature], bins=20, alpha=0.5, label='Original Data')
    plt.hist(new_data[feature], bins=20, alpha=0.5, label='New Data')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Visualize the distributions of categorical features (e.g., using bar plots)
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    original_data[feature].value_counts().plot(kind='bar', alpha=0.5, color='blue', label='Original Data')
    new_data[feature].value_counts().plot(kind='bar', alpha=0.5, color='orange', label='New Data')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend()
    plt.show()


# # Further metrics on the model

# # Cross validation

# In[83]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Split the ud dataset into features and target variable
X_ud = ud.drop('num', axis=1)
y_ud = ud['num']

# Split ud dataset into training and test sets
X_train_ud, X_test_ud, y_train_ud, y_test_ud = train_test_split(X_ud, y_ud, test_size=0.2, random_state=42)

# Train the model on ud dataset
model_ud = GradientBoostingClassifier(random_state=42)
model_ud.fit(X_train_ud, y_train_ud)

# Make predictions on the test set for ud dataset
predictions_ud = model_ud.predict(X_test_ud)

# Print classification reports
print("UD dataset classification report:")
print(classification_report(y_test_ud, predictions_ud))

# Perform cross-validation on the model
cv_scores = cross_val_score(model_ud, X_ud, y_ud, cv=5)
print("Cross-Validation Accuracy Scores:")
print(cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# # Feature Importance

# In[84]:


importances = best_rf.feature_importances_
feature_names = X.columns

# Sort the features based on importance score in descending order
sorted_indices = importances.argsort()[::-1]
sorted_feature_names = feature_names[sorted_indices]

# Print the feature importance scores and names
for i in range(len(sorted_feature_names)):
    print(f"{sorted_feature_names[i]}: {importances[sorted_indices[i]]}")


# # Model Interpretability:

# # For SHAP:

# In[85]:


pip install shap


# In[86]:


import shap

# Assuming you have already trained and saved the 'best_rf' model
# Load the model
best_rf = load('random_forest.joblib')

# Create a SHAP explainer object for the best_rf model
explainer = shap.TreeExplainer(best_rf)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Create an Explanation object using the SHAP values and the original features
shap_explanation = shap.Explanation(values=shap_values[1], base_values=explainer.expected_value[1], data=X_test)

# Plot the SHAP waterfall plot for the specific instance (e.g., the first instance in the test set)
shap.initjs()
shap.plots.waterfall(shap_explanation[0])  # Assuming the positive class is labeled as 1 and plotting the first instance

# If you want to plot the summary plot for the entire test set, you can do:
# shap.plots.waterfall(shap_explanation)


# # For LIME:

# In[87]:


from lime.lime_tabular import LimeTabularExplainer

# Create a LIME explainer object for the best_rf model
explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names)

# Get an explanation for a specific instance (e.g., X_test.iloc[0])
explanation = explainer.explain_instance(X_test.iloc[0].values, best_rf.predict_proba, num_features=len(feature_names))

# Plot the LIME explanation
explanation.as_pyplot_figure()


# # Error Analysis:

# In[88]:


import matplotlib.pyplot as plt

# Assuming you have already defined 'best_rf', 'X_test', 'y_test', 'y_pred'
# Get model predictions on the test set
y_pred = best_rf.predict(X_test)

# Find misclassified instances
misclassified_indices = y_pred != y_test
misclassified_samples = X_test[misclassified_indices]
misclassified_true_labels = y_test[misclassified_indices]
misclassified_predicted_labels = y_pred[misclassified_indices]

# Print the number of misclassified instances
print("Number of misclassified instances:", len(misclassified_samples))

# Perform analysis on misclassified samples to understand common patterns
# For example, you can plot misclassified samples based on specific features or compare true and predicted labels.

# Plot a histogram of the true labels for misclassified instances
plt.hist(misclassified_true_labels, bins=[0, 1, 2], align='left', rwidth=0.5)
plt.xticks([0, 1], labels=['Class 0', 'Class 1'])
plt.xlabel('True Labels')
plt.ylabel('Frequency')
plt.title('Histogram of True Labels for Misclassified Instances')
plt.show()

# Plot a histogram of the predicted labels for misclassified instances
plt.hist(misclassified_predicted_labels, bins=[0, 1, 2], align='left', rwidth=0.5)
plt.xticks([0, 1], labels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Labels for Misclassified Instances')
plt.show()

# Compare specific features between misclassified and correctly classified instances
feature_to_analyze = 'age'  # Replace 'age' with the desired feature
plt.scatter(X_test[feature_to_analyze], y_test, c=y_pred == y_test)
plt.xlabel(feature_to_analyze)
plt.ylabel('True Labels')
plt.title(f'{feature_to_analyze} vs. True Labels (Misclassified vs. Correctly Classified)')
plt.show()

# You can perform similar analysis for other features or create additional plots as needed.

# Note: Depending on the nature of your data and the number of misclassified instances, you might need to customize the analysis to gain better insights.


# # Deploy on Streamlit

# In[89]:


from joblib import dump

# Assuming 'best_rf' is your trained model
dump(best_rf, 'random_forest.joblib')


# In[107]:


import streamlit as st
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the model
model = load('random_forest.joblib')

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/sjeno/HeartSense-Proof-of-Concept/main/final_heart_disease_uci.csv')

# Define function to take user input
def get_user_input():
    age = st.sidebar.slider('Age', 30, 80, 50)
    sex = st.sidebar.selectbox('Sex', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG Results', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    
    # Create a data frame from the inputs
    user_data = pd.DataFrame({'age': [age],
                              'sex': [sex],
                              'cp': [cp],
                              'trestbps': [trestbps],
                              'chol': [chol],
                              'fbs': [fbs],
                              'restecg': [restecg],
                              'thalach': [thalach],
                              'exang': [exang],
                              'oldpeak': [oldpeak]})
    return user_data

# Main function to structure the web app
def main():
    st.title('Heart Disease Predictor')
    st.write('Please input the patient information on the left side of the page:')
    
    # Get user input
    user_input = get_user_input()

    # Make predictions
    prediction = model.predict(user_input)

    # Display prediction result and explanation
    if prediction[0] == 0:
        st.write("Prediction: No heart disease detected.")
        st.write("Explanation: The model predicts that the individual does not have heart disease.")
    elif prediction[0] == 1:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has mild heart disease (Level 1).")
    elif prediction[0] == 2:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has moderate heart disease (Level 2).")
    elif prediction[0] == 3:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has severe heart disease (Level 3).")
    elif prediction[0] == 4:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has very severe heart disease (Level 4).")

# Call the main function to run the app
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




