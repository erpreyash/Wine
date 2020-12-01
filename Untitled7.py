#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


wine = pd.read_csv('winequality-red.csv')


# In[4]:


wine.head()


# In[6]:


wine.info


# In[7]:


fig=plt.figure(figsize=(10,7))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# In[8]:


fig=plt.figure(figsize=(10,7))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# In[9]:


fig=plt.figure(figsize=(10,7))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# In[10]:


fig=plt.figure(figsize=(10,7))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# In[11]:


fig=plt.figure(figsize=(10,7))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# In[12]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


# In[14]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[15]:


label_quality = LabelEncoder()


# In[16]:


wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[17]:


wine['quality'].value_counts()


# In[18]:


sns.countplot(wine['quality'])


# In[19]:


X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[21]:


sc = StandardScaler()


# In[22]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[23]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[24]:


print(classification_report(y_test, pred_rfc))


# In[25]:


print(confusion_matrix(y_test, pred_rfc))


# In[26]:


sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)


# In[27]:


print(classification_report(y_test, pred_sgd))


# In[28]:


print(confusion_matrix(y_test, pred_sgd))


# In[29]:


svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)


# In[30]:


print(classification_report(y_test, pred_svc))


# In[31]:


param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)


# In[32]:


grid_svc.fit(X_train, y_train)


# In[33]:


#Best parameters for our svc model
grid_svc.best_params_


# In[34]:


#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))


# In[ ]:




