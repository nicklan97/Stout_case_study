#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv('loans_full_schema.csv')


# In[3]:


data.head(5)


# ## 1. Describe the dataset and any issues with it

# In[4]:


data.shape


# The data has 10,000 rows and 55 variables. This is a dataset of 10k loans with 55 attributes availabe.

# In[5]:


data.dtypes


# There are 13 categorial variables (including employment title) and 42 numeriacal variables.

# ### Numerical Variables Summary Table

# In[6]:


pd.options.display.float_format = '{:.2f}'.format


# In[7]:


numeric_description = data.describe().transpose()[['count','mean','min','max']]


# In[8]:


numrecords = 10000


# In[9]:


numeric_description['%populated'] = numeric_description['count']/numrecords*100
numeric_description


# Issues: months_since_90d_late, months_since_last_delinq, annual_income_joint and debt_to_income_joint have lower 50% of value filled, and num_accounts_120d_past_due, months_since_last_credit_inquiry, debt_to_income and emp_length have minor portion of missing values.

# ### Categorical Variables Summary Table

# In[10]:


data_categorical = data[['emp_title','state','homeownership','verified_income','verification_income_joint',                         'loan_purpose','application_type','grade','sub_grade','issue_month',                        'loan_status','initial_listing_status','disbursement_method']]


# In[11]:


data_categorical = data_categorical.astype('object', copy=True)


# In[12]:


summary_categrorical=data_categorical.describe().transpose()[['count','unique','top']].rename(columns = {'top':'most common','unique':'# unique values'})
summary_categrorical['% populated'] = summary_categrorical['count']*100/numrecords
summary_categrorical.drop('count',axis = 1)
summary_categrorical


# ## 2. Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations

# In[13]:


plt.rcParams.update({'figure.figsize':(12,6)})
plt.rcParams.update({'font.size':20})


# 1. Interest Rate

# In[14]:


data.interest_rate.plot(title = 'Interest Rates')


# Observation: From the graph we can see that the interest rate of each loan is within the range of 5%-30%,most of them are in the range of 10%-15%.

# 2. homeownership

# In[15]:


data['homeownership'].value_counts().plot(kind = 'bar',title = 'Home Ownership')


# Observation: From the graph we can see that most loan applicant's are on the morgage, only about 15% own a home.

# 3. Annual Income and Interest Rate

# In[16]:


data.plot(kind = 'scatter',x ='annual_income' , y='interest_rate',          title = 'Annual Income vs Interest Rate')


# Observation: We can see that higher income population tends to enjoy lower interest rate as the highest interest rates are applied uniformly to people with lower income.

# 4. State Distribution

# In[17]:


data['state'].value_counts().head(10).plot(kind = 'bar',title = 'Loan Distribution')


# Observation: California has the most loan issued, then Texas, New York and Florida. Those 4 sates are the most economically active states in the country.

# 5. Load Grad

# In[18]:


data['grade'].value_counts().plot(kind = 'bar',title = 'Loan Grade')


# Observation: We can see that most loans that released are in Grade A,B or C and the number of loans issued under C grade decreases significantly.

# ### Feature Set & Modeling 

# With 14.5% of loan popluation as joint,and the nature that a joint loan should considering the information of the other parties involved, I decided to remove the joint loan to solely consider people who will be solely responsible for the loan.

# In[19]:


data_model = data[data['annual_income_joint'].isnull() == True]


# In[20]:


data_model


# In[21]:


data.columns


# From the features that we have in the dataset, I choose to exclude some variables that is not related to the interest rate such as title, status of the loan, or paid status, and keep variables that are associated with the risk. Also,'months_since_90d_late' and 'months_since_last_delinq' have too many missing values and the missing could be both interpreted as not late/deliquency or simply missing, different imputations could greatly affect the model so I decide to remove them as well.

# In[22]:


feature_set = ['emp_length','state','homeownership','annual_income','verified_income','delinq_2y',              'debt_to_income','earliest_credit_line','inquiries_last_12m','total_credit_lines',              'open_credit_lines','total_credit_limit','total_credit_utilized','num_collections_last_12m',               'num_historical_failed_to_pay', 'current_accounts_delinq',               'total_collection_amount_ever', 'current_installment_accounts','accounts_opened_24m',                'months_since_last_credit_inquiry','num_satisfactory_accounts', 'num_accounts_120d_past_due',               'num_accounts_30d_past_due', 'num_active_debit_accounts','total_debit_limit', 'num_total_cc_accounts',                'num_open_cc_accounts','num_cc_carrying_balance', 'num_mort_accounts','account_never_delinq_percent',               'public_record_bankrupt','loan_purpose','loan_amount', 'term',                'installment', 'grade', 'sub_grade','interest_rate']
      


# In[23]:


len(feature_set)


# 38 variables used for model in total.

# In[24]:


data_model = data_model[feature_set]


# In[25]:


pd.set_option('display.max_columns', None)


# In[26]:


data_model 


# For the next step, I need to fill in the missing values in emp_length, debt_to_income, months_since_last_credit_inquiry, num_accounts_120d_past_due by average values of the same fields.

# In[27]:


sum(data_model['emp_length'].isna())


# In[28]:


data_model['emp_length'] = data['emp_length'].fillna(data['emp_length'].mean())


# In[29]:


sum(data_model['emp_length'].isna())


# In[30]:


sum(data_model['months_since_last_credit_inquiry'].isna())


# In[31]:


data_model['months_since_last_credit_inquiry'] = data['months_since_last_credit_inquiry'].fillna(data['months_since_last_credit_inquiry'].mean())


# In[32]:


sum(data_model['months_since_last_credit_inquiry'].isna())


# In[33]:


sum(data_model['num_accounts_120d_past_due'].isna())


# In[34]:


data_model['num_accounts_120d_past_due'] = data['num_accounts_120d_past_due'].fillna(data['num_accounts_120d_past_due'].mean())


# In[35]:


sum(data_model['num_accounts_120d_past_due'].isna())


# In[36]:


sum(data_model['debt_to_income'].isna())


# In[37]:


data_model['debt_to_income'] = data['debt_to_income'].fillna(data['debt_to_income'].mean())


# In[38]:


sum(data_model['debt_to_income'].isna())


# Next, I am going to parce the categorical variables into dummy variables for model building.

# In[39]:


data_model = pd.get_dummies(data_model, columns=['state','homeownership','verified_income','loan_purpose','grade','sub_grade'])


# In[40]:


data_model


# Train-Test Split

# In[41]:


X = data_model.drop('interest_rate', axis=1)


# In[42]:


y = data_model['interest_rate']


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# 1.Model 1: Linear regression model building.

# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


reg = LinearRegression().fit(X_train, y_train)


# In[46]:


reg.score(X_train,y_train)


# In[47]:


reg.score(X_test,y_test)


# In[48]:


predict_1 = reg.predict(X_test)


# In[49]:


mean_squared_error(y_test, predict_1)


# the linear regression model has 0.998 r_squared, meaning that it capures 99.8% of variability in interest rate, and 0.035 MSE, measuring the average error of each point prediction on the testing dataset.

# In[50]:


reg_data = pd.DataFrame({'actual':y_test,'predicted':predict_1})


# In[51]:


sns.regplot(x="actual", y="predicted", data=reg_data).set(title='actual vs predicted (LR)')


# 2. Model 2: Random Forest Regression.

# In[52]:


from sklearn.ensemble import RandomForestRegressor


# In[53]:


rf_reg = RandomForestRegressor(n_estimators = 100, random_state = 0)


# In[54]:


rf_reg.fit(X_train,y_train) 


# In[55]:


predict_2 = rf_reg.predict(X_test)


# In[56]:


rf_reg.score(X_train,y_train)


# In[57]:


rf_reg.score(X_test,y_test)


# In[58]:


mean_squared_error(y_test, predict_2)


# The RF model produced 0.9993 r_squared and 0.015 MSE, which are a bit better than LR model.

# In[59]:


rfreg_data = pd.DataFrame({'actual':y_test,'predicted':predict_2})


# In[60]:


sns.regplot(x="actual", y="predicted", data=rfreg_data).set(title='actual vs predicted(RF)')


# 3. Model 3: XGboost Regression

# In[61]:


from xgboost import XGBRegressor


# In[62]:


xg_reg = XGBRegressor()


# In[63]:


xg_reg.fit(X_train,y_train) 


# In[64]:


predict_3 = xg_reg.predict(X_test)


# In[65]:


xg_reg.score(X_train,y_train)


# In[66]:


xg_reg.score(X_test,y_test)


# In[67]:


mean_squared_error(y_test, predict_3)


# XGB regression model is the best in prediction as it yields best r_squared and MSE.

# In[68]:


xgreg_data = pd.DataFrame({'actual':y_test,'predicted':predict_3})


# In[69]:


sns.regplot(x="actual", y="predicted", data=xgreg_data).set(title='actual vs predicted(XGB)')


# Upon 3 models I tried, the XG Boost regression model produces the most pleasing results so I decided to use it as the final model.

# ###  Enhencement Proposal

# If I had more time, I would try enclude co-loan models to handle cases where more than 1 party were involed in a loan. I would like to treat that as another senario and try to include more features on the information of other applicants, which are not included in this dataset. For example, the annual income,and financial status of the second applicant in the joint loan. Also, although the model has great performance on this dataset, the 10k dataset is still relatively samll compared to millions of loan applications in the real life. Therefore, I would like to improve the performance of the model in the following ways if a larger dataset is available: 1. feature engineering, I want to consult professionals and build expert variables. 2. model tunning, I would like to try more machine learning algorithms such as neural network and perform grid search in parameters to find a model that produces the best prediction. 3. validation, I would use cross validation to make sure that the parameter tunning is stable in the real predictions.
