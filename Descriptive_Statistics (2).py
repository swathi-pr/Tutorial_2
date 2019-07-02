
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean
from pandas import DataFrame as df


# In[52]:



data = pd.read_csv("Z://7th sem//dataanalytics//traffic-collision.csv")


# In[3]:


data.describe()


# In[4]:


N = 20
P = ["Crime Code","Victim Age"]
Q = [1,2,3]


# In[5]:


values = [[998,511], [1119,620], [1300,790]]


# In[6]:


mus = np.concatenate([np.repeat(value, N) for value in values])



# In[11]:


data = df(data = {'iv1': np.concatenate([np.array([p]*N) for p in P]*len(Q))
,'iv2': np.concatenate([np.array([q]*(N*len(P))) for q in Q])
,'rt': np.random.normal(mus, scale=112.0, size=N*len(P)*len(Q))})


# In[12]:


grouped_data = data.groupby(['iv1', 'iv2'])
grouped_data['rt'].describe().unstack()


# In[13]:


grouped_data['rt'].mean().reset_index()


# In[14]:


grouped_data['rt'].aggregate(np.mean).reset_index()


# In[15]:


grouped_data['rt'].apply(gmean, axis=None).reset_index()


# In[16]:


grouped_data['rt'].apply(hmean, axis=None).reset_index()


# In[17]:


trimmed_mean = grouped_data['rt'].apply(trim_mean, .1)
trimmed_mean.reset_index()


# In[19]:


grouped_data['rt'].quantile([.25, .5, .75]).unstack()


# In[31]:


df = pd.DataFrame(np.random.rand(10, 5), columns=['Victim_Age', 'B', 'C', 'D', 'E'])
df.plot.box(grid='True')


# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[53]:


data.head()


# In[54]:


data.Victim_Age


# In[55]:


traffic=data.Victim_Age


# In[56]:


traffic.mean()


# In[58]:


import scipy
from scipy import stats
scipy.stats.hmean(data.loc[:,"Area_ID"])


# In[59]:


scipy.stats.mstats.gmean(traffic, axis=0)


# In[61]:


scipy.stats.iqr(data.Area_ID)


# In[62]:


plt.boxplot(data.Area_ID)


# In[64]:


print(scipy.stats.hmean(data.loc[:,"Area_ID"]))


# In[65]:


print(scipy.stats.mstats.gmean(data.Area_ID, axis=0))


# In[66]:


plt.boxplot(data.Time_Occured)


# In[67]:


import seaborn as sns
ax=sns.countplot(x="Area_ID",data=data.sample(10))


# In[68]:


sns.boxplot(x="Area_ID",y="Victim_Age",data=data)


# In[69]:


m=data.Victim_Age.sample(20)
n=data.Area_ID.sample(20)
plt.pie(n,labels=m,autopct='%.1f%%',startangle=90)
plt.show()


# In[70]:


plt.bar(data.Victim_Age.sample(20),data.Area_ID.sample(20))
plt.show()	


# In[76]:


import matplotlib.pyplot as plt
x=data.Victim_Age
y=data.Area_ID
plt.scatter(x,y,label='swa_out',color='k',s=10,marker='o')
plt.xlabel('x')
p1t.ylabel('y')
plt.title('Traffic collision scatterplot')
plt.legend()
plt.show()

