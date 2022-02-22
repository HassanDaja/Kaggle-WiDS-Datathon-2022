#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# In[94]:


#Reading The test and train files
test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
submission=pd.read_csv("sample_solution.csv")


# In[95]:


train.head()


# In[96]:


#getting Columns with nan Values 
nan_cols=train.isna().any()[lambda x: x]


# In[97]:


#removing outlayers
q_low = train["site_eui"].quantile(0.01)
q_hi  = train["site_eui"].quantile(0.99)
train = train[(train["site_eui"] < q_hi) & (train["site_eui"] > q_low)]


# In[98]:


#Deleting Useless data
train=train.drop('id',1)
train=train.drop('Year_Factor',1)


# In[99]:


#Swapping nan valuse with median
for col in nan_cols.keys():
    train[col]=train[col].fillna((train[col].median()))


# In[100]:


test.head()


# In[101]:


#getting the catigorical data(object type data)
cat_cols=train.select_dtypes(include=['object'])
print(cat_cols.columns)


# In[102]:


#turing catigorical data into numeric data
le = LabelEncoder()
for col in cat_cols.columns:
    nums = le.fit_transform(cat_cols[str(col)])
    train[str(col),"_Numeric"] = nums
#removing the catigorical data 
train=train.drop(cat_cols.columns,1)
    


# In[103]:


train.head(20)


# In[104]:


#splitting the data 
x=train.drop('site_eui',1)
y=train['site_eui']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# In[105]:


#getting best parameters for xgbaggressor 
def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'max_depth': [5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'n_estimators' : [100, 200],
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           cv = 2,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_
#getting catboost model
def get_model():
    cat_params = {
        'iterations': 35000,
        'learning_rate': 0.025,
        'od_wait': 1000,
        'depth': 8,
        'task_type' : 'GPU',
        'devices' : '0',
        'verbose' : 1000,
        "objective": "RMSE",
        "loss_function": "RMSE"
    }
    model =  CatBoostRegressor(**cat_params)
    return model


# In[106]:


model_cb=get_model()


# In[107]:


model_cb.fit(x_train,y_train)


# In[108]:


model_cb.score(x_test,y_test)


# In[109]:


#Test data

#getting Columns with nan Values 
nan_cols=test.isna().any()[lambda x: x]

#Deleting Useless cells
#test=test.drop('id',1)
#test=test.drop('Year_Factor',1)

#Swapping nan valuse with median
for col in nan_cols.keys():
    test[col]=test[col].fillna((test[col].median()))
cat_cols=test.select_dtypes(include=['object'])
le = LabelEncoder()
for col in cat_cols.columns:
    nums = le.fit_transform(cat_cols[str(col)])
    test[str(col),"_Numeric"] = nums
test=test.drop(cat_cols.columns,1)


# In[110]:


#predicting the test values
res=model_cb.predict(test)


# In[111]:


results = pd.DataFrame(submission['id'])
results['site_eui']=res


# In[112]:


results.to_csv("submission.csv", header=True, index=False)


# In[ ]:




