#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression on Salary Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('metadata.csv')
df.head(10)


# In[3]:


df.shape
#df.head()


# In[5]:


x = df.iloc[:,:-1].values
y = df.iloc[:, 1].values
    


# In[6]:


print(x)


# In[7]:


print(y)


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# In[9]:


print(x_train)


# In[10]:


print(y_train)


# In[11]:


print(x_test)


# In[12]:


print(y_test)


# In[13]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[ ]:


y_pred=model.predict(x_test)
y_pred


# In[ ]:


x_pred = model.predict(x_train)
x_pred


# In[ ]:


y_test[:6]


# In[ ]:


model.predict(x_test[:6])


# In[ ]:


model.score(x_test,y_test)


# In[14]:


#visualizing the Train set results
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,x_pred, color='red')
plt.title('Salary vs Experience (Training Dataset)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


# In[22]:


#visualizing the Test set results  
plt.scatter(x_test, y_test, color="blue")   
plt.plot(x_train, x_pred, color="red")    
plt.title("Salary vs Experience (Test Dataset)")  
plt.xlabel("Years of Experience")  
plt.ylabel("Salary(In Rupees)")  

