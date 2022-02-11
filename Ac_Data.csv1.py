#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[2]:


df=pd.read_csv("AC_data.csv")
df


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


# Extracting categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']


# In[8]:


# looking at unique values in categorical columns
for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")


# In[9]:


# checking numerical features distribution

plt.figure(figsize = (30, 25))
plotnumber = 1

for column in num_cols:
    if plotnumber <= 18:
        ax = plt.subplot(6, 3, plotnumber)
        sns.histplot(df[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[10]:


# finding all columns that have nan:

droping_list_all=[]
for j in range(0,19):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        print(df.iloc[:,j].unique())
droping_list_all


# In[11]:


# filling nan with mean in any columns

for j in range(1,19):        
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())


# In[12]:


df.isnull().sum()


# In[13]:


hist = df.iloc[:,4:8].hist()


# In[14]:


hist = df.iloc[:,8:12].hist()


# In[15]:


hist = df.iloc[:,12:16].hist()


# In[16]:


hist = df.iloc[:,16:20].hist()


# In[17]:


df['date'] = pd.to_datetime(df['DATE'])
df['time'] = df['date'].dt.time
df = df.set_index('date') 
weekly_summary = df.resample('W').sum()


# In[18]:


df.drop(['DATE'], axis='columns', inplace=True)


# In[19]:


df


# In[20]:


df.AC_1.resample('D').sum().plot(title='AC_1 resampled over day for sum') 
df.AC_1.resample('D').mean().plot(title='AC_1 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_1.resample('D').mean().plot(title='AC_1 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[21]:


df.AC_2.resample('D').sum().plot(title='AC_2 resampled over day for sum') 
df.AC_2.resample('D').mean().plot(title='AC_2 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_2.resample('D').mean().plot(title='AC_2 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[22]:


df.AC_3.resample('D').sum().plot(title='AC_3 resampled over day for sum') 
df.AC_3.resample('D').mean().plot(title='AC_3 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_3.resample('D').mean().plot(title='AC_3 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[23]:


df.AC_4.resample('D').sum().plot(title='AC_4 resampled over day for sum') 
df.AC_4.resample('D').mean().plot(title='AC_4 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_4.resample('D').mean().plot(title='AC_4 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[24]:


df.AC_5.resample('D').sum().plot(title='AC_5 resampled over day for sum') 
df.AC_5.resample('D').mean().plot(title='AC_5 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_5.resample('D').mean().plot(title='AC_5 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[25]:


df.AC_6.resample('D').sum().plot(title='AC_6 resampled over day for sum') 
df.AC_6.resample('D').mean().plot(title='AC_6 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_6.resample('D').mean().plot(title='AC_6 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[26]:


df.AC_7.resample('D').sum().plot(title='AC_7 resampled over day for sum') 
df.AC_7.resample('D').mean().plot(title='AC_7 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_7.resample('D').mean().plot(title='AC_7 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[27]:


df.AC_8.resample('D').sum().plot(title='AC_8 resampled over day for sum') 
df.AC_8.resample('D').mean().plot(title='AC_8 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_8.resample('D').mean().plot(title='AC_8 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[28]:


df.AC_9.resample('D').sum().plot(title='AC_9 resampled over day for sum') 
df.AC_9.resample('D').mean().plot(title='AC_9 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_9.resample('D').mean().plot(title='AC_9 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[29]:


df.AC_10.resample('D').sum().plot(title='AC_10 resampled over day for sum') 
df.AC_10.resample('D').mean().plot(title='AC_10 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_10.resample('D').mean().plot(title='AC_10 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[30]:


df.AC_11.resample('D').sum().plot(title='AC_11 resampled over day for sum') 
df.AC_11.resample('D').mean().plot(title='AC_11 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_11.resample('D').mean().plot(title='AC_11 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[31]:


df.AC_12.resample('D').sum().plot(title='AC_12 resampled over day for sum') 
df.AC_12.resample('D').mean().plot(title='AC_12 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_12.resample('D').mean().plot(title='AC_12 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[32]:


df.AC_13.resample('D').sum().plot(title='AC_13 resampled over day for sum') 
df.AC_13.resample('D').mean().plot(title='AC_13 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_13.resample('D').mean().plot(title='AC_13 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[33]:


df.AC_14.resample('D').sum().plot(title='AC_14 resampled over day for sum') 
df.AC_14.resample('D').mean().plot(title='AC_14 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_14.resample('D').mean().plot(title='AC_14 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[34]:


df.AC_15.resample('D').sum().plot(title='AC_15 resampled over day for sum') 
df.AC_15.resample('D').mean().plot(title='AC_15 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_15.resample('D').mean().plot(title='AC_15 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[35]:


df.AC_9.resample('D').sum().plot(title='AC_9 resampled over day for sum') 
df.AC_9.resample('D').mean().plot(title='AC_9 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_9.resample('D').mean().plot(title='AC_9 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[36]:


df.AC_17.resample('D').sum().plot(title='AC_17 resampled over day for sum') 
df.AC_17.resample('D').mean().plot(title='AC_17 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_17.resample('D').mean().plot(title='AC_17 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[37]:


df.AC_18.resample('D').sum().plot(title='AC_18 resampled over day for sum') 
df.AC_18.resample('D').mean().plot(title='AC_18 resampled over day', color='red') 
plt.tight_layout()
plt.show()   

df.AC_18.resample('D').mean().plot(title='AC_18 resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[38]:


r = df.AC_1.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_1 resampled over day')
plt.show()


# In[39]:


r = df.AC_2.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_2 resampled over day')
plt.show()


# In[40]:


r = df.AC_3.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_3 resampled over day')
plt.show()


# In[41]:


r = df.AC_4.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_4 resampled over day')
plt.show()


# In[42]:


r = df.AC_5.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_5 resampled over day')
plt.show()


# In[43]:


r = df.AC_6.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_6 resampled over day')
plt.show()


# In[44]:


r = df.AC_7.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_7 resampled over day')
plt.show()


# In[45]:


r = df.AC_8.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_8 resampled over day')
plt.show()


# In[46]:


r = df.AC_9.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_9 resampled over day')
plt.show()


# In[47]:


r = df.AC_10.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_10 resampled over day')
plt.show()


# In[48]:


r = df.AC_11.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_11 resampled over day')
plt.show()


# In[49]:


r = df.AC_12.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_12 resampled over day')
plt.show()


# In[50]:


r = df.AC_13.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_13 resampled over day')
plt.show()


# In[51]:


r = df.AC_14.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_14 resampled over day')
plt.show()


# In[52]:


r = df.AC_15.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_15 resampled over day')
plt.show()


# In[53]:


r = df.AC_16.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_16 resampled over day')
plt.show()


# In[54]:


r = df.AC_17.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_17 resampled over day')
plt.show()


# In[55]:


r = df.AC_18.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='AC_18 resampled over day')
plt.show()


# In[56]:


# Mean of 'AC_1' resampled over quarter
df['AC_1'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_1')
plt.title('AC_1 per quarter (averaged over quarter)')
plt.show()


# In[57]:


# Mean of 'AC_2' resampled over quarter
df['AC_2'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_2')
plt.title('AC_2 per quarter (averaged over quarter)')
plt.show()


# In[58]:


# Mean of 'AC_3' resampled over quarter
df['AC_3'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_3')
plt.title('AC_3 per quarter (averaged over quarter)')
plt.show()


# In[59]:


# Mean of 'AC_4' resampled over quarter
df['AC_4'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_4')
plt.title('AC_4 per quarter (averaged over quarter)')
plt.show()


# In[60]:


# Mean of 'AC_5' resampled over quarter
df['AC_5'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_5')
plt.title('AC_5 per quarter (averaged over quarter)')
plt.show()


# In[61]:


# Mean of 'AC_6' resampled over quarter
df['AC_6'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_6')
plt.title('AC_6 per quarter (averaged over quarter)')
plt.show()


# In[62]:


# Mean of 'AC_7' resampled over quarter
df['AC_7'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_7')
plt.title('AC_7 per quarter (averaged over quarter)')
plt.show()


# In[63]:


# Mean of 'AC_8' resampled over quarter
df['AC_8'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_8')
plt.title('AC_8 per quarter (averaged over quarter)')
plt.show()


# In[64]:


# Mean of 'AC_9' resampled over quarter
df['AC_9'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_9')
plt.title('AC_9 per quarter (averaged over quarter)')
plt.show()


# In[65]:


# Mean of 'AC_10' resampled over quarter
df['AC_10'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_10')
plt.title('AC_10 per quarter (averaged over quarter)')
plt.show()


# In[66]:


# Mean of 'AC_11' resampled over quarter
df['AC_11'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_11')
plt.title('AC_11 per quarter (averaged over quarter)')
plt.show()


# In[67]:


# Mean of 'AC_12' resampled over quarter
df['AC_12'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_12')
plt.title('AC_12 per quarter (averaged over quarter)')
plt.show()


# In[68]:


# Mean of 'AC_13' resampled over quarter
df['AC_13'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_13')
plt.title('AC_13 per quarter (averaged over quarter)')
plt.show()


# In[69]:


# Mean of 'AC_14' resampled over quarter
df['AC_14'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_14')
plt.title('AC_14 per quarter (averaged over quarter)')
plt.show()


# In[70]:


# Mean of 'AC_15' resampled over quarter
df['AC_15'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_15')
plt.title('AC_15 per quarter (averaged over quarter)')
plt.show()


# In[71]:


# Mean of 'AC_16' resampled over quarter
df['AC_16'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_16')
plt.title('AC_16 per quarter (averaged over quarter)')
plt.show()


# In[72]:


# Mean of 'AC_17' resampled over quarter
df['AC_17'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_17')
plt.title('AC_17 per quarter (averaged over quarter)')
plt.show()


# In[73]:


# Mean of 'AC_18' resampled over quarter
df['AC_18'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('AC_18')
plt.title('AC_18 per quarter (averaged over quarter)')
plt.show()


# In[74]:


cols = [0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17]
i = 1
groups=cols
values = df.resample('D').mean().values
# plot each column
plt.figure(figsize=(15, 10))
for group in groups:
    plt.subplot(len(cols), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.75, loc='right')
    i += 1
plt.show()


# In[75]:


# resampling over week and computing mean
df.AC_1.resample('W').mean().plot(color='y', legend=True)
df.AC_2.resample('W').mean().plot(color='r', legend=True)
df.AC_3.resample('W').mean().plot(color='b', legend=True)
df.AC_4.resample('W').mean().plot(color='g', legend=True)
plt.show()


# In[76]:


# resampling over week and computing mean
df.AC_5.resample('W').mean().plot(color='y', legend=True)
df.AC_6.resample('W').mean().plot(color='r', legend=True)
df.AC_7.resample('W').mean().plot(color='b', legend=True)
df.AC_8.resample('W').mean().plot(color='g', legend=True)
plt.show()


# In[77]:


# resampling over week and computing mean
df.AC_9.resample('W').mean().plot(color='y', legend=True)
df.AC_10.resample('W').mean().plot(color='r', legend=True)
df.AC_11.resample('W').mean().plot(color='b', legend=True)
df.AC_12.resample('W').mean().plot(color='g', legend=True)
plt.show()


# In[78]:


# resampling over week and computing mean
df.AC_13.resample('W').mean().plot(color='y', legend=True)
df.AC_14.resample('W').mean().plot(color='r', legend=True)
df.AC_15.resample('W').mean().plot(color='b', legend=True)
df.AC_16.resample('W').mean().plot(color='g', legend=True)
plt.show()


# In[79]:


# resampling over week and computing mean
df.AC_17.resample('W').mean().plot(color='y', legend=True)
df.AC_18.resample('W').mean().plot(color='r', legend=True)
plt.show()


# In[80]:


# Below I show hist plot of the mean of different feature resampled over month 
df.AC_1.resample('M').mean().plot(kind='hist', color='r', legend=True )
df.AC_2.resample('M').mean().plot(kind='hist',color='b', legend=True)
df.AC_3.resample('M').mean().plot(kind='hist', color='g', legend=True)
df.AC_4.resample('M').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[81]:


# Below I show hist plot of the mean of different feature resampled over month 
df.AC_5.resample('M').mean().plot(kind='hist', color='r', legend=True )
df.AC_6.resample('M').mean().plot(kind='hist',color='b', legend=True)
df.AC_7.resample('M').mean().plot(kind='hist', color='g', legend=True)
df.AC_8.resample('M').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[82]:


# Below I show hist plot of the mean of different feature resampled over month 
df.AC_9.resample('M').mean().plot(kind='hist', color='r', legend=True )
df.AC_10.resample('M').mean().plot(kind='hist',color='b', legend=True)
df.AC_11.resample('M').mean().plot(kind='hist', color='g', legend=True)
df.AC_12.resample('M').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[83]:


# Below I show hist plot of the mean of different feature resampled over month 
df.AC_13.resample('M').mean().plot(kind='hist', color='r', legend=True )
df.AC_14.resample('M').mean().plot(kind='hist',color='b', legend=True)
df.AC_15.resample('M').mean().plot(kind='hist', color='g', legend=True)
df.AC_16.resample('M').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[84]:


# Below I show hist plot of the mean of different feature resampled over month 
df.AC_17.resample('M').mean().plot(kind='hist', color='r', legend=True )
df.AC_18.resample('M').mean().plot(kind='hist',color='b', legend=True)
plt.show()


# In[85]:


# Below I show hist plot of the mean of different feature resampled over year
df.AC_1.resample('Y').mean().plot(kind='hist', color='r', legend=True )
df.AC_2.resample('Y').mean().plot(kind='hist',color='b', legend=True)
df.AC_3.resample('Y').mean().plot(kind='hist', color='g', legend=True)
df.AC_4.resample('Y').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[86]:


# Below I show hist plot of the mean of different feature resampled over year
df.AC_5.resample('Y').mean().plot(kind='hist', color='r', legend=True )
df.AC_6.resample('Y').mean().plot(kind='hist',color='b', legend=True)
df.AC_7.resample('Y').mean().plot(kind='hist', color='g', legend=True)
df.AC_8.resample('Y').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[87]:


# Below I show hist plot of the mean of different feature resampled over year
df.AC_9.resample('Y').mean().plot(kind='hist', color='r', legend=True )
df.AC_10.resample('Y').mean().plot(kind='hist',color='b', legend=True)
df.AC_11.resample('Y').mean().plot(kind='hist', color='g', legend=True)
df.AC_12.resample('Y').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[88]:


# Below I show hist plot of the mean of different feature resampled over year
df.AC_13.resample('Y').mean().plot(kind='hist', color='r', legend=True )
df.AC_14.resample('Y').mean().plot(kind='hist',color='b', legend=True)
df.AC_15.resample('Y').mean().plot(kind='hist', color='g', legend=True)
df.AC_16.resample('Y').mean().plot(kind='hist', color='y', legend=True)
plt.show()


# In[89]:


# Below I show hist plot of the mean of different feature resampled over year
df.AC_17.resample('Y').mean().plot(kind='hist', color='r', legend=True )
df.AC_18.resample('Y').mean().plot(kind='hist',color='b', legend=True)
plt.show()


# In[90]:


sns.scatterplot(x="AC_1",y="date",data=df)


# In[91]:


sns.scatterplot(x="AC_2",y="date",data=df)


# In[92]:


sns.scatterplot(x="AC_3",y="date",data=df)


# In[93]:


sns.scatterplot(x="AC_4",y="date",data=df)


# In[94]:


sns.scatterplot(x="AC_5",y="date",data=df)


# In[95]:


sns.scatterplot(x="AC_6",y="date",data=df)


# In[96]:


sns.scatterplot(x="AC_7",y="date",data=df)


# In[97]:


sns.scatterplot(x="AC_8",y="date",data=df)


# In[98]:


sns.scatterplot(x="AC_9",y="date",data=df)


# In[99]:


sns.scatterplot(x="AC_10",y="date",data=df)


# In[100]:


sns.scatterplot(x="AC_11",y="date",data=df)


# In[101]:


sns.scatterplot(x="AC_12",y="date",data=df)


# In[102]:


sns.scatterplot(x="AC_13",y="date",data=df)


# In[103]:


sns.scatterplot(x="AC_14",y="date",data=df)


# In[104]:


sns.scatterplot(x="AC_15",y="date",data=df)


# In[105]:


sns.scatterplot(x="AC_16",y="date",data=df)


# In[106]:


sns.scatterplot(x="AC_17",y="date",data=df)


# In[107]:


sns.scatterplot(x="AC_18",y="date",data=df)


# In[108]:


cols = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,]
i = 1
groups=cols
values = df.resample('T').mean().values
# plot each column
plt.figure(figsize=(15, 10))
for group in groups:
    plt.subplot(len(cols), 1, i)
    plt.plot(values[:, group])
    plt.title(df.columns[group], y=0.75, loc='right')
    i += 1
plt.show()


# In[109]:


df.plot()


# In[110]:


# Correlations among columns
plt.matshow(df.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('without resampling', size=15)
plt.colorbar()
plt.show()


# In[111]:


# Correlations of mean of features resampled over months


plt.matshow(df.resample('M').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over month', size=15)
plt.colorbar()
plt.margins(0.02)
plt.matshow(df.resample('A').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over year', size=15)
plt.colorbar()
plt.show()


# In[142]:


df['time'].nunique


# In[144]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["time"]=LE.fit_transform(df["time"])


# In[145]:


df.plot(kind='box',subplots=True, layout=(6,7), figsize= (15,10))


# In[148]:


# spliting the independent and target variables in x and y
x=df.drop("time",axis=1)
y=df["time"]


# In[149]:


# Important feature using ExtraTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection =ExtraTreesRegressor()
selection.fit(x,y)


# In[150]:


#ExtraTreesRegressor is used to choose the importan feature for the prediction
print(selection.feature_importances_)


# In[151]:


# plot graph of important feature for better visualization

plt.figure(figsize =(12,8))
feat_importances =pd.Series(selection.feature_importances_,index =x.columns)
feat_importances.nlargest(20).plot(kind="barh")
plt.show()


# In[152]:


from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(x) 
x.loc[:,:] = scaled_values


# In[153]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[154]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(x_train,y_train)


# In[155]:


y_pred =reg_rf.predict(x_test)


# In[156]:


reg_rf.score(x_train,y_train)


# In[157]:


reg_rf.score(x_test,y_test)


# In[158]:


plt.scatter(y_test,y_pred,alpha =0.5,color="DarkBlue")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[159]:


from sklearn import metrics


# In[160]:


print("MAE:" , metrics.mean_absolute_error(y_test,y_pred))
print("MSE:" , metrics.mean_squared_error(y_test,y_pred))
print("RMSE:" , np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[161]:


metrics.r2_score(y_test, y_pred)


# In[162]:


from sklearn.model_selection import RandomizedSearchCV


# In[163]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[164]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[165]:


rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 3, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[166]:


rf_random.fit(x_train,y_train)


# In[167]:


rf_random.best_params_


# In[168]:


prediction = rf_random.predict(x_test)


# In[169]:


plt.scatter(y_test,prediction,alpha =0.5,color="DarkBlue")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[170]:


print("MAE:" , metrics.mean_absolute_error(y_test,prediction))
print("MSE:" , metrics.mean_squared_error(y_test,prediction))
print("RMSE:" , np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[173]:


import joblib #Saving model
joblib.dump(rf_random,"Ac_Data.csv.obj")


# In[174]:


p=joblib.load("Ac_Data.csv.obj")


# In[176]:


import numpy as np
a=np.array(y_test)
predicted=np.array(rf_random.predict(x_test))
df_1=pd.DataFrame({"original":a,"predicted":predicted},index=range(len(a)))


# In[177]:


df_1


# In[ ]:




