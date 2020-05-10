#!/usr/bin/env python
# coding: utf-8

# In[265]:


import pandas as pd


# In[266]:


#We will upload the CSV file in the local to a pandas data frame in my Jupyter notebook.
df = pd.read_csv("titanic1.csv")
df.head(20)


# In[267]:


#dropping unnecessary features.
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[268]:


#splitting data as input target.
inputs = df.drop('Survived',axis='columns')
target = df.Survived


# In[269]:


#inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})


# In[270]:


#changing gender feature as male and female.(The sex column is our text on we want to convert it into digits.)
#Machine Learning model can't handle string so convert them into digits.
dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)


# In[271]:


#We concatenate the input and dummies array.
inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)


# In[272]:


#We dropped the sex and male column because we will use just the female column.(if female is 0 the gender is male.)
inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.head(3)


# In[273]:


#As you can see some column has missing values.
inputs.columns[inputs.isna().any()]
inputs.Age[:10]


# In[248]:


#And here we define NaN values mean values from the age column.
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()


# In[250]:


#splitting data and train as test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2,random_state=0)


# In[252]:


# Calculate the Gaussian probability distribution.
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[254]:


#We training the model with fit function.
model.fit(X_train,y_train)


# In[256]:


#We measure the score find accuracy.
model.score(X_test,y_test)


# In[260]:


#first ten y values of test data.
y_test[0:10]


# In[262]:


#first ten predicted y values.
model.predict(X_test[0:10])


# In[264]:


#Orginal values, no rounding.
model.predict_proba(X_test[:10])

