#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default='notebook'


# In[2]:


data = pd.read_csv('casestudy.csv')


# In[4]:


data = data.drop('Unnamed: 0',axis = 1)


# In[5]:


data


# In[7]:


data_2015 = data[data['year'] == 2015]
data_2016 = data[data['year'] == 2016]
data_2017 = data[data['year'] == 2017]


# In[8]:


data_2015


# In[44]:


len(data_2015['customer_email'].unique())


# In[24]:


data_2016


# In[45]:


len(data_2016['customer_email'].unique())


# In[25]:


data_2017


# In[46]:


len(data_2017['customer_email'].unique())


# ### Total revenue for the current year

# 2015:

# In[9]:


data_2015['net_revenue'].sum()


# 2016:

# In[10]:


data_2016['net_revenue'].sum()


# 2017:

# In[11]:


data_2017['net_revenue'].sum()


# ### New Customer Revenue e.g. new customers not present in previous year only

# 2015-2016

# In[21]:


temp = data_2016.merge(data_2015,on = 'customer_email',how = 'left')


# In[22]:


temp


# In[23]:


temp[temp['net_revenue_y'].isna()==True]['net_revenue_x'].sum()


# 2016-2017

# In[26]:


temp2 = data_2017.merge(data_2016,on = 'customer_email',how = 'left')


# In[27]:


temp2[temp2['net_revenue_y'].isna()==True]['net_revenue_x'].sum()


# ### Existing Customer Growth

# 2015-2016

# In[36]:


existing_cus = temp[temp['net_revenue_y'].isna()==False]


# In[37]:


existing_cus['growth'] = existing_cus['net_revenue_x'] - existing_cus['net_revenue_y']


# In[39]:


existing_cus['growth'].sum()


# 2016-2017

# In[40]:


existing_cus2 = temp2[temp2['net_revenue_y'].isna()==False]


# In[41]:


existing_cus2['growth'] = existing_cus2['net_revenue_x'] - existing_cus2['net_revenue_y']


# In[42]:


existing_cus2['growth'].sum()


# ### Revenue lost from attrition

# Since we have no idea of how much profit we can earn from the lost customers in the coming year, I would just use the attrited customers' profit last year to estimate the attrition.

# 2015-2016

# In[52]:


temp3 = data_2015.merge(data_2016,on = 'customer_email',how = 'left')


# In[53]:


temp3


# In[55]:


lost_cus1 =temp3[temp3['net_revenue_y'].isna()==True]


# In[57]:


lost_cus1['net_revenue_x'].sum()


# 2016-2017

# In[58]:


temp4 = data_2016.merge(data_2017,on = 'customer_email',how = 'left')


# In[59]:


lost_cus2 = temp4[temp4['net_revenue_y'].isna()==True]


# In[60]:


lost_cus2['net_revenue_x'].sum()


# ### Existing Customer Revenue Current Year

# 2015-2016

# In[61]:


existing_cus['net_revenue_x'].sum()


# 2016-2017

# In[62]:


existing_cus2['net_revenue_x'].sum()


# ### Existing Customer Revenue Previous Year

# 2017-2016

# In[63]:


pre_cus = temp4[temp4['net_revenue_y'].isna()==False]


# In[64]:


pre_cus['net_revenue_y'].sum()


# 2016-2015

# In[65]:


pre_cus2 = temp3[temp3['net_revenue_y'].isna()==False]


# In[66]:


pre_cus2['net_revenue_y'].sum()


# ### Total Customers Current  Year

# 2015

# In[68]:


data_2015['customer_email'].nunique()


# 2016

# In[69]:


data_2016['customer_email'].nunique()


# 2017

# In[70]:


data_2017['customer_email'].nunique()


# ### Total Customers Previous Year

# 2016-2015

# In[72]:


data_2015['customer_email'].nunique()


# 2017-2016

# In[73]:


data_2016['customer_email'].nunique()


# ### New Customers

# 2015-2016

# In[74]:


temp[temp['net_revenue_y'].isna()==True]['customer_email'].nunique()


# 2016-2017

# In[75]:


temp2[temp2['net_revenue_y'].isna()==True]['customer_email'].nunique()


# ### Lost Customers

# 2015-2016

# In[76]:


temp3[temp3['net_revenue_y'].isna()==True]['customer_email'].nunique()


# 2016-2017

# In[77]:


temp4[temp4['net_revenue_y'].isna()==True]['customer_email'].nunique()


# ### Few plots

# 1.number of customers

# In[83]:


customers = [231294,204646,249987]
year = ['2015','2016','2017']
plt.bar(year, customers, color ='maroon',
        width = 0.4)
plt.title('number of customers each year')


# 2. total revenue each year

# In[85]:


revenue = [29036749.189999994,25730943.59,31417495.030000016]
year = ['2015','2016','2017']
plt.bar(year, revenue, color ='maroon',
        width = 0.4)
plt.title('Total revenue each year')


# 3.new customer

# In[89]:


revenue = [145062,229028]
year = ['2015-2016','2016-2017']
plt.bar(year, revenue, color ='maroon',
        width = 0.4)
plt.title('new customer')


# 4.customer number dynamics

# In[118]:


year = ['2015','2016','2017']
base_cus = [231294,231294,204646]
cus_accuired = [0,145062,229028]
cus_lost = [0,-171710,-183687]


# In[119]:


df = pd.DataFrame({'year':year,                   "customer_previous_year":base_cus,                   'new_customer_accquired':cus_accuired,
                   'customer_lost':cus_lost})


# In[121]:


df 


# In[120]:


import plotly.express as px
fig = px.bar(df, x="year", y=["customer_previous_year", "new_customer_accquired", "customer_lost"],
             title="customer number dynamics")
fig.show()


# Insight 1: Compared to 2015, 2016 evidenced the customer decrease because more customers attrited than new customers acquired, in 2017 the customer number starts to grow because the number of newly acquired customers exceeds the number of customer loss at the same period. Therefore, we need to on hand keep the good work on attracting new customers and on the other hand, try to fix the deteriorating customer loss issue.

# 5. total revenue dynamics

# In[109]:


year = ['2015','2016','2017']
base_revenue = [29036749.189999994,0,0]
existing_customer = [0,7485452.58,2641259.99]
new_customer_revenue = [0,18245491.01,28776235.039999995]


# In[110]:


df2 = pd.DataFrame({'year':year,"base_revenue":base_revenue,                   "existing_customer_current_year":existing_customer,                   'new_customer_revenue':new_customer_revenue})


# In[111]:


fig = px.bar(df2, x="year", y=["base_revenue", "existing_customer_current_year",'new_customer_revenue'],
             title="revenue dynamics")
fig.show()


# Insight 2: new customers are the key drivers of the revenue because they are always the larger portion of revenue decomposition. Therefore, we need to prioritize the new user aquisitions when we have limited resources. Also, the decrease of the existing customer spending's is a concern and we need to figure out why it decreases significantly compared to 2016.
