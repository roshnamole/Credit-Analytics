#!/usr/bin/env python
# coding: utf-8

# # Credit Analytics

# In[3]:


#importing librabries and loading data
#Input and output features
#DATA Exploration
import pandas

#-- load in the data
data = pandas.read_csv("credit_card_history (1).csv")

#the features
X = data[data.columns[:-1]]
Y = data[data.columns[-1]]

print("Number of datapoints:", X.shape[0])
print("Number of input features:", X.shape[1])


# In[4]:


data.head(5)


# In[6]:


data.dtypes


# In[8]:


# FICO categorizations
demographics = data.columns[0:5]
statuses = data.columns[5:11]
bills = data.columns[11:17]
payments = data.columns[17:23]

data[statuses].head(10)


# In[9]:


data[demographics].head(5)


# In[13]:


#Summarizing the data with Graphs
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.style.use('ggplot')


plt.rcParams['figure.figsize'] = (20.0, 20.0)
plt.rcParams.update({'font.size': 10})


fig, ax = plt.subplots(2,3)
fig.set_size_inches(15,10)

print(ax)


for i, feature in enumerate(statuses):

  
   counts  = data[feature].value_counts()

  
   row, col = int(i / 3), i % 3

   
   ax[row, col].bar(counts.index, counts, align='center')
   ax[row, col].set_title(feature)


plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
plt.show()


# In[14]:



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.style.use('ggplot')


plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.rcParams.update({'font.size': 10})

fig.set_size_inches(10,5)

data[payments].boxplot()


plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
plt.show()


# In[15]:


plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.rcParams.update({'font.size': 10})

fig.set_size_inches(10,5)

import numpy as np
np.log(data[payments]).boxplot()


plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
plt.show()


# In[16]:


#for each person what's the average bill on their account each month?
plt.rcParams['figure.figsize'] = (10.0, 10.0)

#histogram line 
plt.hist(data[bills].mean(axis=1), 20)

#labelling.
plt.xlabel("Average bill amount (£)")
plt.ylabel("Number of people")
plt.show()


# In[22]:


plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.hist(data[payments].mean(axis=1), 20,color='Blue')

# labelling.
plt.xlabel("Average payment amount (£)")
plt.ylabel("Number of people")
plt.show()


# In[18]:


plt.hist(data["LIMIT"], bins=50, color='purple')
plt.xlabel("Credit limit category (£)")
plt.ylabel("Number of people")
plt.show()


# In[24]:


#: Explore the "DEFAULTED" Column
fig = plt.figure()
fig.set_size_inches(10,5)

d = data.groupby(['DEFAULTED']).size()

print("Defaulting accounts are {}% out of {} observations".format(100* d[1]/(d[1]+d[0]), d[1]+d[0]))

p = d.plot(kind='barh', color='Black')


# In[25]:


fig, ax = plt.subplots(1,3)
fig.set_size_inches(10,5)
fig.suptitle('Defaulting by absolute numbers, for various demographics')

# a plot for split by gender
d = data.groupby(['DEFAULTED', 'SEX']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[0])

# plot for "MARRIAGE"
d = data.groupby(['DEFAULTED', 'MARRIAGE']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[1])

#plot for split by "AGE"
data['AGE'] = pandas.cut(data['AGE'], range(0, 100, 10), right=False)
d = data.groupby(['DEFAULTED', 'AGE']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[2])


# In[26]:


#Feature Engineering
X['STATUS_MEAN'] = data[statuses].mean(axis=1)
X['STATUS_STD'] = data[statuses].std(axis=1)

X['BILL_MEAN'] = data[bills].mean(axis=1)
X['BILL_STD'] = data[bills].std(axis=1)

X['PAY_MEAN'] = data[payments].mean(axis=1)
X['PAY_STD'] = data[payments].std(axis=1)

X.describe()


# In[27]:


X.head(5)


# In[28]:


#Feature importance to avoid the curse of dimensionality.
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k= 5)

selector_ft = selector.fit_transform(X, Y)


# In[29]:


print(selector.scores_)


# In[30]:


import numpy as np
import pandas as pd

scores= selector.scores_
names_scores = list(zip(X.columns, scores))

ns_df = pd.DataFrame(data = names_scores, columns= ['Feature name','F-Score'])
ns_df_sorted = ns_df.sort_values(['F-Score','Feature name'], ascending = [False, True])[:20] #-- this creates an ordered dataframe of scores of the 20 best variables
print(ns_df_sorted)


# Predictive models

# In[31]:


from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import dummy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


folds = KFold(n_splits=10, shuffle=True, random_state=0)


# In[32]:


lr = linear_model.LogisticRegression()
lr_scores = cross_val_score(lr, X, Y, cv=folds)
print("Mean LR Accuracy:", np.mean(lr_scores))


# In[33]:


print("Mean LR Accuracy:", np.mean(lr_scores))


# In[34]:


dt = tree.DecisionTreeClassifier(max_depth=20, min_samples_leaf=7)
dt_scores = cross_val_score(dt, X, Y, cv=folds)
print("Mean DT Accuracy:", np.mean(dt_scores))


# In[35]:


ab = ensemble.AdaBoostClassifier(n_estimators=300, learning_rate=1)
ab_scores = cross_val_score(ab, X, Y, cv=folds)
print("Mean AB Accuracy:", np.mean(ab_scores))


# In[36]:


dc = dummy.DummyClassifier()
dc_scores = cross_val_score(dc, X, Y, cv=folds)
print("Baseline Accuracy:", np.mean(dc_scores))


# Statistical Testing

# In[37]:


from scipy.stats import ttest_rel #p value remains lower than 0.05

stat, p = ttest_rel(dt_scores, ab_scores)
print(np.round(p, decimals= 4))


# In[38]:


from scipy.stats import wilcoxon #p value remains lower than 0.05

stat, p = wilcoxon(lr_scores, ab_scores)
print(np.round(p, 4))


# 
