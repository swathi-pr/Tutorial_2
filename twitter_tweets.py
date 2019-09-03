#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys
import tweepy


# In[7]:


consumer_key="xRptwP8hiIPAGxBJStv8QDLOF"
consumer_secret="6yVFE5cr4LNIoWrRsqeOYKgFsYwLZtIbg9sf0d6R683CNepeFk"
access_token="881380653404569600-AmcDuMjgdB1qJgXwd474IqoIiqNECbJ"
access_token_secret="BG1sTPmkO4JqeP6EPksiROLqKG1Z9R4TcnvxNnF5CV8gs"


# In[11]:


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
#auth_api = API(auth)
api = tweepy.API(auth)


# In[12]:


public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


# In[13]:


api = tweepy.API(auth)


name = "nytimes"

tweetCount = 20


results = api.user_timeline(id=name, count=tweetCount)


for tweet in results:
   
   print tweet.text
    


# In[15]:


api = tweepy.API(auth)
name = "nytimes"
tweetCount = 20
results = api.user_timeline(id=name, count=tweetCount)
for tweet in results:
    print (tweet.text)


# In[17]:


api = tweepy.API(auth)
query = "Toptal"
language = "en"
results = api.search(q=query, lang=language)
for tweet in results:
      print (tweet.user.screen_name,"Tweeted:",tweet.text)


# In[ ]:




