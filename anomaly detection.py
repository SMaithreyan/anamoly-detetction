#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pycaret


# In[2]:


from pycaret.datasets import get_data

anomaly = get_data('anomaly')

#import anomaly detection module
from pycaret.anomaly import *

#intialize the setup
exp_ano = setup(anomaly)


# In[3]:


# display the example data from pycaret

anomaly


# In[ ]:


##model used - iforest

iforest=create_model('iforest')
## plotting a model
plot_model(iforest)


# In[ ]:




