
# coding: utf-8

# In[2302]:


import pandas as pd
import numpy as np


# In[2303]:


df=pd.read_csv('train.csv')


# ## Preprocessing

# In[2304]:


df.describe()


# ### New Variables creations like ratio of checkout_price and base price , avg demand by centers and meal, Last three weeks demand etc are created

# In[2305]:


df['Month'] = df['week'].apply(lambda x: int(x / 4))
df['Year'] = df['week'].apply(lambda x: int(x / 52))
df['Quarter'] = df['week'].apply(lambda x: int(x / 13))


# In[2306]:


df['Year'].value_counts()


# In[2307]:


df_1=pd.read_csv('fulfilment_center_info.csv')
df_2=pd.read_csv('meal_info.csv')
df_3=pd.merge(df, df_1, on='center_id')
df_=pd.merge(df_3, df_2, on='meal_id')


# In[2308]:


df_test=pd.read_csv('test_QoiMO9B.csv')


# In[2309]:


df_test['Month'] = df_test['week'].apply(lambda x: int(x / 4))
df_test['Year'] = df_test['week'].apply(lambda x: int(x / 52))
df_test['Quarter'] = df_test['week'].apply(lambda x: int(x / 13))


# In[2312]:


df_.loc[df_['checkout_price'] < df_['base_price'], 'C'] = 1
df_.loc[df_['checkout_price'] > df_['base_price'], 'C'] = 2
df_.loc[df_['checkout_price'] == df_['base_price'], 'C'] = 0


# In[2313]:


df_test.loc[df_test['checkout_price'] < df_test['base_price'], 'C'] = 1
df_test.loc[df_test['checkout_price'] > df_test['base_price'], 'C'] = 2
df_test.loc[df_test['checkout_price'] == df_test['base_price'], 'C'] = 0


# In[2314]:


df_['ratio']=df_['checkout_price']/df_['base_price']
df_test['ratio']=df_test['checkout_price']/df_test['base_price']


# In[2315]:


df_['ratio'] = df_['ratio'].apply(lambda x: 1 if(x<0.5) else x)
df_test['ratio'] = df_test['ratio'].apply(lambda x: 1 if(x<0.5) else x)


# In[2316]:


df_3=pd.merge(df_test, df_1, on='center_id')
df_test=pd.merge(df_3, df_2, on='meal_id')


# In[2317]:


df_.loc[df_['checkout_price'] <= 100, 'Cat'] = 0
df_.loc[(df_['checkout_price'] > 100) & (df_['checkout_price'] <= 150), 'Cat'] = 1
df_.loc[(df_['checkout_price'] > 150) & (df_['checkout_price'] <= 200), 'Cat'] = 2
df_.loc[(df_['checkout_price'] > 200) & (df_['checkout_price'] <= 300), 'Cat'] = 3
df_.loc[(df_['checkout_price'] > 300) & (df_['checkout_price'] <= 500), 'Cat'] = 4
df_.loc[(df_['checkout_price'] > 500) , 'Cat'] = 5


# In[2318]:


df_.loc[df_['base_price'] <= 200, 'Cat_'] = 0
df_.loc[(df_['base_price'] > 200) & (df_['base_price'] <= 400), 'Cat_'] = 1
df_.loc[(df_['base_price'] > 400) & (df_['base_price'] <= 500), 'Cat_'] = 2
df_.loc[(df_['base_price'] > 500) , 'Cat_'] = 3


# In[2319]:


df_test.loc[df_test['base_price'] <= 200, 'Cat_'] = 0
df_test.loc[(df_test['base_price'] > 200) & (df_test['base_price'] <= 400), 'Cat_'] = 1
df_test.loc[(df_test['base_price'] > 400) & (df_test['base_price'] <= 500), 'Cat_'] = 2
df_test.loc[(df_test['base_price'] > 500) , 'Cat_'] = 3


# In[2320]:


df_test.loc[df_test['checkout_price'] <= 100, 'Cat'] = 0
df_test.loc[(df_test['checkout_price'] > 100) & (df_test['checkout_price'] < 150), 'Cat'] = 1
df_test.loc[(df_['checkout_price'] > 150) & (df_test['checkout_price'] < 200), 'Cat'] = 2
df_test.loc[(df_['checkout_price'] > 200) & (df_test['checkout_price'] < 300), 'Cat'] = 3
df_test.loc[(df_['checkout_price'] > 300) & (df_test['checkout_price'] < 500), 'Cat'] = 4
df_test.loc[(df_['checkout_price'] > 500) , 'Cat'] = 5


# In[2321]:


ID='id'
week='week'
target='num_orders'


# In[2322]:


df_test.columns


# In[2323]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[2324]:


df_test


# In[2325]:


columns_encode=['center_id','meal_id','city_code','region_code','center_type','op_area','category','cuisine']
for column in columns_encode:
    le.fit(df_[column])
    le.classes_
    df_[column]=le.transform(df_[column])
    df_test[column]=le.transform(df_test[column])


# In[2326]:


df_last=df_.loc[df_.groupby(['center_id','meal_id']).week.idxmax()]
df_last=df_last[['meal_id','center_id','num_orders']]
df_last=df_last.rename(columns={'num_orders':'last_order'})
df_last=df_last.reset_index()


# In[2327]:


df_last3=df_[(df_['week']==143) | (df_['week']==144) | (df_['week']==145) ]
df_last3_group=df_last3.groupby(['center_id','meal_id'])[target].mean()
df_group_3=df_last3_group.reset_index()
df_group_3=df_group_3.rename(columns={'num_orders':'avg_3_orders'})
df_group=df_.groupby(['center_id','meal_id'])[target].mean()
df_group=df_group.reset_index()
df_group=df_group.rename(columns={'num_orders':'avg_orders'})
df_new = pd.merge(df_, df_group,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])
df_new = pd.merge(df_new, df_group_3,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])
df__ = pd.merge(df_new, df_last,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])
df_new_2 = pd.merge(df_test, df_group,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])
df_new_2 = pd.merge(df_new_2, df_group_3,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])
df_test = pd.merge(df_new_2, df_last,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])


# In[2328]:


df_group=df_.groupby('center_id')[target].mean()
df_group=df_group.reset_index()
df_group_1=df_group.rename(columns={'num_orders':'avg_center'})
df__ = pd.merge(df__, df_group_1,  how='left', left_on=['center_id'], right_on = ['center_id'])
df_test = pd.merge(df_test, df_group_1,  how='left', left_on=['center_id'], right_on = ['center_id'])
df_group=df_.groupby('meal_id')[target].mean()
df_group=df_group.reset_index()
df_group_2=df_group.rename(columns={'num_orders':'avg_meal'})
df__ = pd.merge(df__, df_group_2,  how='left', left_on=['meal_id'], right_on = ['meal_id'])
df_test = pd.merge(df_test, df_group_2,  how='left', left_on=['meal_id'], right_on = ['meal_id'])


# In[2329]:


df_


# In[2330]:


df__.columns


# In[2331]:


df_=df__


# In[2332]:


df1 = df_[df_.isnull().any(axis=1)]


# In[2333]:


df1.describe()


# In[2334]:


df_['avg_3_orders']=df_['avg_3_orders'].fillna(df_['avg_orders'])


# In[2335]:


df_test['avg_3_orders']=df_test['avg_3_orders'].fillna(df_test['avg_orders'])


# In[2336]:


avg=df['num_orders'].median()
df_=df_.fillna(avg)
df_test=df_test.fillna(avg)
df_['avg_orders'] = df_['avg_orders'].apply(lambda x: round(x))
df_['avg_3_orders'] = df_['avg_3_orders'].apply(lambda x: round(x))


# In[2337]:


avg


# In[2338]:


df_[df_['avg_3_orders']==None]


# In[2339]:


df_test['avg_orders'] = df_test['avg_orders'].apply(lambda x: round(x))


# In[2340]:


df_test['avg_3_orders'] = df_test['avg_3_orders'].apply(lambda x: round(x))


# In[2341]:


for column in df_.columns:
    print(type(df_[column]))


# In[2342]:


def encodingfunction(df):
    columns_to_encode=['season','holiday','workingday','weather','Time_Category']
    for column in columns_to_encode:
        df[column] = df[column].apply(lambda x: str(x)+column)
        one_hot = pd.get_dummies(df[column])
        # Drop column B as it is now encoded
        df = df.drop(column,axis = 1)
        # Join the encoded df
        df = df.join(one_hot,how='left')
    return df


# ## Modelling

# In[2343]:


from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


# In[2344]:


def modelfit(target,alg, dtrain,test, predictors, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    test_predictions = alg.predict(test[predictors])
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    return test_predictions


# In[2345]:


df_check=df_[df_['center_id']==7]
df_new= df_check[df_check['meal_id']==42]


# In[2346]:


df_test.columns


# In[2347]:


import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[2348]:


from sklearn import linear_model


# In[2349]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 100, oob_score = True, n_jobs = -1,max_features = "auto", min_samples_leaf = 10)


# In[2350]:


from sklearn import ensemble


# In[2351]:


df_test.columns


# In[2352]:


best_predictor=[x for x in df_.columns if x not in [ID,target]]


# In[2353]:


test_predictions=modelfit(target,model, df_,df_test, best_predictor,True)
#val=rmsle(list(test_predictions),list(test['']))


# In[2354]:


test_predictions=np.floor(test_predictions)


# In[2355]:


test_predictions


# In[2356]:


test_new=df_test


# In[2357]:


test_new[target]=list(test_predictions)


# In[2358]:


test_new=test_new[[ID,target]]


# In[2359]:


test_new.to_csv('submission.csv',index=False)


# In[626]:


test.dtypes


# In[1980]:


def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


# In[2154]:


best_predictors=[]
from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(n_estimators=200,n_jobs = -1)

def best_variables(train,test,target1):
    sample_leaf_options = [5,10,50,100,500]
    for n in sample_leaf_options :
        model = RandomForestRegressor(n_estimators = n, oob_score = True, n_jobs = -1,max_features = "auto", min_samples_leaf = 10)
        predictors1 = [x for x in train.columns if x not in [target1,ID,'index']]
        test_predictions_1=modelfit(target1,model, train,test, predictors1,True)
        test_predictions_1 = [0 if i < 0 else i for i in test_predictions_1]
        test_predictions_1=np.floor(test_predictions_1)
        best_score=rmsle(list(test[target1]),test_predictions_1)
        print(best_score)


# In[2157]:


train=df_[:15000]
test=df_[20000:30000]


# In[2158]:


best_predictors=best_variables(train,test,target)


# ## Plotting

# In[45]:


sns.set(style="ticks")


# In[105]:


df_


# In[1597]:


plt.scatter(df_['ratio'],df_[target])


# In[200]:


df_new


# In[112]:


plt.hist(df_new['week'],df_new[target])


# In[2222]:


import seaborn as sns
sns.pairplot(df_)


