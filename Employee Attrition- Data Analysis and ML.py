#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Attrition_df = pd.read_csv(r'C:\Users\Admin\Desktop\Udemy_downloads\EmployeeAttrition.csv')
Attrition_df.describe()


# In[3]:


# Checking for null values
Attrition_df.isnull().sum()


# In[4]:


# Dropping columns of no- values
Attrition_df.drop(columns=['Over18','EmployeeCount','StandardHours'], inplace = True)


# In[5]:


#Dropping the outlier rows with Percentiles

upper_lim = Attrition_df['TotalWorkingYears'].quantile(.95)
lower_lim = Attrition_df['TotalWorkingYears'].quantile(.05)

Attrition_df = Attrition_df[(Attrition_df['TotalWorkingYears'] < upper_lim) & (Attrition_df['TotalWorkingYears'] > lower_lim)]


# In[6]:


#Attrition yes - 1 and Attrition no - 0 

Attrition_df.Attrition.replace({'Yes':1,'No':0} , inplace = True)


# In[7]:


# Dividing Data frame 
yes_attrition = Attrition_df[Attrition_df.Attrition == 1]
no_attrition = Attrition_df[Attrition_df.Attrition == 0]


# In[8]:


# Correlation between all fields
Correlation = Attrition_df.corr()
correlation = pd.DataFrame(Correlation)


# In[9]:


# Correlation Heatmap
sns.heatmap(correlation , cmap = 'Blues')


# In[10]:


# Correlation with Attrition field greater then 0.1
correlation.Attrition[correlation.Attrition <= - 0.1 ]


# In[11]:


# Distribution of fields

fig ,axes = plt.subplots(3,3)
fig.set_size_inches(10,10)

sns.distplot(yes_attrition.Age , color = 'blue',ax=axes[0,0])
sns.distplot(no_attrition.Age ,color= 'red', ax=axes[0,0])


sns.distplot(yes_attrition.PerformanceRating , color = 'blue', ax = axes[0,1])
sns.distplot(no_attrition.PerformanceRating ,color= 'red', ax = axes[0,1]) 


sns.distplot(yes_attrition.JobSatisfaction , color = 'blue', ax = axes[1,0])
sns.distplot(no_attrition.JobSatisfaction ,color= 'red', ax = axes[1,0]) 



sns.distplot(yes_attrition.WorkLifeBalance , color = 'blue', ax = axes[1,1])
sns.distplot(no_attrition.WorkLifeBalance  ,color= 'red', ax = axes[1,1]) 


sns.distplot(yes_attrition.EnvironmentSatisfaction  , color = 'blue', ax = axes[0,2])
sns.distplot(no_attrition.EnvironmentSatisfaction   ,color= 'red', ax = axes[0,2]) 


sns.distplot(yes_attrition.TotalWorkingYears , color = 'blue', ax = axes[1,2])
sns.distplot(no_attrition.TotalWorkingYears ,color= 'red', ax = axes[1,2]) 


sns.distplot(yes_attrition.YearsInCurrentRole  , color = 'blue', ax = axes[2,1])
sns.distplot(no_attrition.YearsInCurrentRole  ,color= 'red', ax = axes[2,1]) 



sns.distplot(yes_attrition.JobLevel , color = 'blue', ax = axes[2,0])
sns.distplot(no_attrition.JobLevel ,color= 'red', ax = axes[2,0]) 



sns.distplot(yes_attrition.MonthlyIncome , color = 'blue', ax = axes[2,2])
sns.distplot(no_attrition.MonthlyIncome ,color= 'red', ax = axes[2,2]) 


# In[12]:


# Getting dummy Variables for categorical fields
Dummy_Variables = pd.get_dummies(Attrition_df[['BusinessTravel','Department','EducationField','Gender','JobRole','OverTime','MaritalStatus']])


# In[13]:


# Removing fields from data frames
Attrition_df.drop(columns= ['BusinessTravel','Department','EducationField','Gender','JobRole','OverTime','MaritalStatus'], inplace = True)


# In[14]:


# Combining data frames 
Attrition_with = pd.concat((Attrition_df,Dummy_Variables), axis=1)


# In[15]:


Attrition = Attrition_with.Attrition
Attrition_with.drop(columns=['Attrition'] ,inplace = True)
Attrition_with = pd.concat((Attrition_with , Attrition), axis=1)


# In[16]:


# Separating X and Y 
X = Attrition_with.iloc[:,0:50].values
Y = Attrition_with.iloc[:,-1].values


# In[17]:


# Diving dataset into train and test 
from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test = train_test_split(X,Y,test_size = 0.25 , random_state = 0)


# In[18]:


# importing Random Forest
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators=100 ,criterion = 'entropy', random_state =0)
classifier.fit(x_train,y_train)


# In[19]:


# Prediction
y_pred = classifier.predict(x_test)


# In[20]:


# Confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
cm


# In[21]:


# Finding Accuracy Score on test set
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[22]:


# 10 fold cross-validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier , X = x_train , y = y_train ,cv =10)


# In[23]:


# Mean accuracy and Std
print(accuracies.mean())
print(accuracies.std())


# In[ ]:




