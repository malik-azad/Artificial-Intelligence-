# Linear Regression Model Implementation by Dangerous CP Developer


import os
os.getcwd()


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot asplt


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


ds=pd.read_csv(r'C:\Users\User\linear regression\SaleryDataset\saleryDataset.csv')


# In[9]:


print(ds.head())


# In[11]:


X=ds["YearsExperience"].values
Y=ds["Salary"].values


# In[12]:


X


# In[13]:


Y


# In[14]:


plt.plot(X,Y)


# In[15]:


X=np.array(X)
Y=np.array(Y)


# In[16]:


def mean(X):
    return np.sum(X)/len(X)


# In[19]:


def variance(X):
    mean_value=mean(X)
    return np.sum((X-mean_value)**2)/len(X)


# In[20]:


def norm(X):
    mean_value=mean(X)
    variance_value=variance(X)
    return (X-mean_value)/np.sqrt(variance_value)


# In[21]:


X_norm=norm(X)


# In[22]:


X_norm


# In[23]:


plt.plot(X_norm,Y)


# In[24]:


get_ipython().system('pip install scikit-learn')


# In[25]:


from sklearn.linear_model import LinearRegression


# In[27]:


x_norm=X_norm.reshape(-1,1)


# In[28]:


reg=LinearRegression().fit(x_norm,Y)


# In[29]:


reg.score(x_norm,Y)


# In[31]:


reg.coef_


# In[32]:


reg.intercept_


# In[33]:


y_pred=reg.predict(x_norm)


# In[34]:


y_pred


# In[35]:


plt.plot(x_norm,Y)
plt.plot(x_norm,y_pred)


# In[ ]:
