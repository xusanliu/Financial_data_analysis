#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from datetime import datetime
from alphalens import utils
from alphalens import performance


# In[3]:


trading_dates=get_trading_dates(datetime(2019,1,1),datetime(2019,12,31))
startDate=trading_dates[0]
endDate=trading_dates[-1]
stock_list=index_components('000300.XSHG')
factor_one_name='market_cap'
factor_two_name='basic_earnings_per_share'


# In[4]:


q_one=query(fundamentals.eod_derivative_indicator.market_cap
           ).filter(fundamentals.eod_derivative_indicator.stockcode.in_(stock_list))
q_two=query(fundamentals.income_statement.basic_earnings_per_share
           ).filter(fundamentals.income_statement.stockcode.in_(stock_list))


# In[5]:


#先处理因子1
df_facs_data_one=pd.DataFrame()
for i in range(len(trading_dates)):
    daily_fac_data=get_fundamentals(q_one,trading_dates[i],expect_df=True)
    daily_fac_data=daily_fac_data.reset_index()
    daily_fac_data['date']=trading_dates[i]
    daily_fac_data=daily_fac_data.set_index(['date','order_book_id'])
    df_facs_data_one=pd.concat([df_facs_data_one,daily_fac_data])


# In[6]:


df_facs_data_one


# In[7]:


def winsorize_series(series):
    q=series.quantile([0.02,0.98])
    if isinstance(q,pd.Series) and len(q)==2:
        series[series<q.iloc[0]]=q.iloc[0]
        series[series>q.iloc[1]]=q.iloc[1]
    return series

def standardize_series(series):
    std=series.std()
    mean=series.mean()
    return (series-mean)/std  


# In[8]:


series_facs_data_one=df_facs_data_one['market_cap']
series_facs_data_one = series_facs_data_one.groupby(level = 'date').apply(winsorize_series)
series_facs_data_one = series_facs_data_one.groupby(level = 'date').apply(standardize_series)
series_facs_data_one.hist(figsize=(12,6),bins=20)
df_facs_data_one['market_cap']=series_facs_data_one


# In[9]:


price=get_price(stock_list,start_date=startDate,end_date=endDate,fields='close')
price.head()


# In[10]:


facs_data_one_analysis=utils.get_clean_factor_and_forward_returns(series_facs_data_one,price)


# In[11]:


facs_data_one_analysis.head()


# In[12]:


facs_data_one_analysis['factor']=np.float128(facs_data_one_analysis['factor'])


# In[13]:


IC_data_one=performance.factor_information_coefficient(facs_data_one_analysis)


# In[14]:


IC_data_one


# In[15]:


IC_data_one_mean=IC_data_one.iloc[:,0].mean()
IC_data_one_std=IC_data_one.iloc[:,0].std()
IR_data_one=IC_data_one_mean/IC_data_one_std


# In[16]:


IC_data_one_mean


# In[17]:


IC_data_one_std


# In[18]:


freq_one=IC_data_one[IC_data_one.iloc[:,0]>0.02].iloc[:,0].count()/IC_data_one.iloc[:,0].count()
freq_one


# In[19]:


IR_data_one


# In[20]:


factor_one_return=performance.factor_returns(facs_data_one_analysis)
factor_one_return


# In[21]:


mean_factor_one_return=factor_one_return.iloc[:,0].mean()
mean_factor_one_return


# In[22]:


#处理因子2
df_facs_data_two=pd.DataFrame()
for i in range(len(trading_dates)):
    daily_fac_data=get_fundamentals(q_two,trading_dates[i],expect_df=True)
    daily_fac_data=daily_fac_data.reset_index()
    daily_fac_data['date']=trading_dates[i]
    daily_fac_data=daily_fac_data.set_index(['date','order_book_id'])
    df_facs_data_two=pd.concat([df_facs_data_two,daily_fac_data])


# In[23]:


df_facs_data_two


# In[24]:


series_facs_data_two=df_facs_data_two['basic_earnings_per_share']
series_facs_data_two = series_facs_data_two.groupby(level = 'date').apply(winsorize_series)
series_facs_data_two = series_facs_data_two.groupby(level = 'date').apply(standardize_series)
series_facs_data_two.hist(figsize=(12,6),bins=20)
df_facs_data_two['basic_earnings_per_share']=series_facs_data_two


# In[25]:


facs_data_two_analysis=utils.get_clean_factor_and_forward_returns(series_facs_data_two,price)


# In[26]:


facs_data_two_analysis


# In[27]:


facs_data_two_analysis['factor']=np.float128(facs_data_two_analysis['factor'])


# In[28]:


IC_data_two=performance.factor_information_coefficient(facs_data_two_analysis)


# In[29]:


IC_data_two


# In[30]:


IC_data_two_mean=IC_data_two.iloc[:,0].mean()
IC_data_two_std=IC_data_two.iloc[:,0].std()


# In[31]:


IC_data_two_mean


# In[32]:


IC_data_two_std


# In[33]:


freq_two=IC_data_two[IC_data_two.iloc[:,0]>0.02].iloc[:,0].count()/IC_data_two.iloc[:,0].count()
freq_two


# In[34]:


IR_data_two=IC_data_two_mean/IC_data_two_std
IR_data_two


# In[35]:


factor_two_return=performance.factor_returns(facs_data_two_analysis)
factor_two_return


# In[36]:


mean_factor_two_return=factor_two_return.iloc[:,0].mean()
mean_factor_two_return


# In[38]:


#选取前百分之二十的股票
df_facs_data_two=df_facs_data_two.reset_index()


# In[47]:


df_facs_data_two_mean_eps=df_facs_data_two.groupby('order_book_id').mean().sort_values(by=['basic_earnings_per_share'],
                                                                                      ascending=False)


# In[56]:


df_facs_data_two_mean_eps


# In[63]:


df_facs_data_two_mean_eps_top=df_facs_data_two_mean_eps[:60]
df_facs_data_two_mean_eps_top


# In[68]:


#获取前百分之二十的股票的收益率
stockList=df_facs_data_two_mean_eps_top.index


# In[71]:


all_close=get_price(stockList,'2019-01-01','2019-12-31',fields='close')


# In[72]:


all_close


# In[73]:


all_close_return=all_close.pct_change()


# In[82]:


all_close_return


# In[85]:


all_close_return_mean=all_close_return.mean(axis=1)
all_close_return_mean


# In[101]:


all_close_return_mean=pd.DataFrame(all_close_return_mean)
all_close_return_mean.columns=['return']
all_close_return_mean


# In[ ]:


#获取上证指数收益率


# In[91]:


index_component=index_components('000001.XSHG')


# In[92]:


SZ_close=get_price(index_component,'2019-01-01','2019-12-31',fields='close')
SZ_close


# In[93]:


SZ_return=SZ_close.pct_change()
SZ_return


# In[94]:


SZ_return_mean=SZ_return.mean(axis=1)
SZ_return_mean


# In[100]:


SZ_return_mean=pd.DataFrame(SZ_return_mean)
SZ_return_mean.columns=['return']
SZ_return_mean


# In[108]:


from matplotlib import pyplot as plt
y1=all_close_return_mean['return']
y2=SZ_return_mean['return']
x=SZ_return_mean.index
plt.figure(figsize=(20,10))
plt.plot(x,y1,label="选取的股票",color="#000000")
plt.plot(x,y2,label="上证指数",color="#008000",linestyle="--")
plt.legend()
plt.show()


# In[ ]:




