
# coding: utf-8

# In[11]:


# Dependencies
import tweepy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import time
import requests
import json
import statistics

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from twit_api import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


# In[12]:


# Target Account
target_users = ['@tesla', '@HMNYHQ1', '@tim_cook']

all_user_list = []
all_compound_list = []
all_pos_list = []
all_neu_list = []
all_neg_list = []
all_sentiment_list = []
tsla_com_list=[]
tsla_pos_list=[]
tsla_neg_list=[]
tsla_neu_list=[]
appl_com_list=[]
appl_pos_list=[]
appl_neg_list=[]
appl_neu_list=[]
hmny_com_list=[]
hmny_pos_list=[]
hmny_neg_list=[]
hmny_neu_list=[]

# Loop through all targer terms
for target_user in target_users:
    

    # Variables for holding sentiments
    user_list = []
    compound_list = []
    pos_list = []
    neu_list = []
    neg_list = []


    for x in range(5):
        public_tweets = api.user_timeline(target_user, page=x)

        for tweet in public_tweets:

            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            user_list.append(tweet["user"]["screen_name"])
            

            compound_list.append(compound)
            pos_list.append(pos)
            neu_list.append(neu)
            neg_list.append(neg)
          
            if target_user==target_users[0]:
                tsla_com_list.append(compound)
                tsla_pos_list.append(pos)
                tsla_neg_list.append(neg)
                tsla_neu_list.append(neu)
            elif target_user==target_users[1]:
                hmny_com_list.append(compound)
                hmny_pos_list.append(pos)
                hmny_neg_list.append(neg)
                hmny_neu_list.append(neu)
            elif target_user==target_users[2]:
                appl_com_list.append(compound)
                appl_pos_list.append(pos)
                appl_neg_list.append(neg)
                appl_neu_list.append(neu)
            else:
                print('error')
    all_user_list.append(user_list)
    all_compound_list.append(compound_list)
    all_pos_list.append(pos_list)
    all_neu_list.append(neu_list)
    all_neg_list.append(neg_list)

            
        
    sentiments = np.mean(compound_list)
    all_sentiment_list.append(sentiments)


# In[13]:


all_user_list = np.array(all_user_list).flatten().tolist()
all_compound_list = np.array(all_compound_list).flatten().tolist()
all_pos_list = np.array(all_pos_list).flatten().tolist()
all_neu_list = np.array(all_neu_list).flatten().tolist()
all_neg_list = np.array(all_neg_list).flatten().tolist()


sentiment = {'User': all_user_list,'Compound_Score': np.mean(all_compound_list), 'Pos_Score': np.mean(all_pos_list), 'Neu_Score': np.mean(all_neu_list), 'Neg_Score': np.mean(all_neg_list)}
sentiment_df = pd.DataFrame(sentiment)
#sentiment_df.to_csv("Sentiment Analysis.csv", index=False, header=True)
sentiment_df


# In[14]:


tesla = {
        "User": "Tesla",
        "Compound": np.mean(tsla_com_list),
        "Positive": np.mean(tsla_pos_list),
        "Neutral": np.mean(tsla_neg_list),
        "Negative": np.mean(tsla_neu_list)
    }
apple = {
        "User": "Apple",
        "Compound": np.mean(appl_com_list),
        "Positive": np.mean(appl_pos_list),
        "Neutral": np.mean(appl_neg_list),
        "Negative": np.mean(appl_neu_list)
    }
hmny = {
        "User": "HMNY",
        "Compound": np.mean(hmny_com_list),
        "Positive": np.mean(hmny_pos_list),
        "Neutral": np.mean(hmny_neg_list),
        "Negative": np.mean(hmny_neu_list)
    }
print(tesla,
      apple,
      hmny)


# In[15]:


if tesla['Neutral']> tesla['Positive'] and tesla['Negative']:
    tsla_twit_outcome= "Neutral"
elif tesla['Positive']>tesla['Negative']:
    tsla_twit_outcome= "Positive"
else:
    tsla_twit_outcome= "Negative"
#tsla_twit_outcome

if apple['Neutral']> apple['Positive'] and apple['Negative']:
    appl_twit_outcome= "Neutral"
elif apple['Positive']>apple['Negative']:
    appl_twit_outcome= "Positive"
else:
    appl_twit_outcome= "Negative"
#appl_twit_outcome
if hmny['Neutral']> hmny['Positive'] and hmny['Negative']:
    hmny_twit_outcome= "Neutral"
elif hmny['Positive']>hmny['Negative']:
    hmny_twit_outcome= "Positive"
else:
    hmny_twit_outcome= "Negative"


# In[16]:


tsla_twit_feedback=[]
#if  +,-, or = then add subtract the difference to the tsla twit feedback to adjust the results
#add a check so it checks yesterdays prediction to today but doesnt duplicate.


# In[17]:


# URL for GET requests to retrieve vehicle data
ticker = ["AAPL","TSLA","HMNY"]

url = "https://api.iextrading.com/1.0/stock/"
duration = "/chart/3m"


# In[18]:



#7 day moving average

week_counter = 0 
avg_counter = 0
series1 = 10
series2 = 7
count = 0  
moving_avg =[]
company = []

for i in ticker:
    
    response = requests.get(url+i+duration).json()
    print(i)
    count = 0
    
    for x in range(series1):

            close = []
            
            for y in range(series2):
                try:
                    close.append(response[count]['close'])
                    #print(close)
                    count = count +1

                except:
                    print("No value")    
                    
            print(close) 
            avg = statistics.mean(close)
            
            company.append(i)
            moving_avg.append(avg)

            #zip(company, moving_avg)

print("Ticker and Moving Average")
for c, m in zip(company, moving_avg):  
    print(f'{c} {m}')        
    #print(moving_avg)


# In[19]:


#create data frame from API call
summary = pd.DataFrame({"Ticker": company,
                      "Moving Average": moving_avg})
summary


# In[20]:


#create dataframe and reset index to grpah
AAPL = summary.loc[(summary["Ticker"] == "AAPL")]
AAPL = AAPL.reset_index(drop=True)

TSLA = summary.loc[(summary["Ticker"] == "TSLA")]
TSLA = TSLA.reset_index(drop=True)

HMNY = summary.loc[(summary["Ticker"] == "HMNY")]
HMNY = HMNY.reset_index(drop=True)


# In[21]:


#Graph AAPL 
#graph properties
plt.xlabel("Moving Average")
plt.ylabel("Price ($)")
plt.title(f"Ticker:AAPL 7 Day Moving Average")
plt.grid()

#graph
plt.rcParams["figure.figsize"] = [14,8]
plt.xlim([0,series1])
plt.plot(AAPL["Moving Average"], marker="o", color="red", linewidth=2)
plt.show()


# In[22]:


#Graph TSLA
#graph properties
plt.xlabel("Moving Average")
plt.ylabel("Price ($)")
plt.title(f"Ticker:TSLA 7 Day Moving Average")
plt.grid()

#graph
plt.rcParams["figure.figsize"] = [14,8]
plt.xlim([0,series1])
plt.plot(TSLA["Moving Average"], marker="o", color="red", linewidth=2)
plt.show()


# In[23]:


#Graph HMNY
#graph properties
plt.xlabel("Moving Average")
plt.ylabel("Price ($)")
plt.title(f"Ticker:HMNY 7 Day Moving Average")
plt.grid()

#graph
plt.rcParams["figure.figsize"] = [14,8]
plt.xlim([0,series1])
plt.plot(HMNY["Moving Average"], marker="o", color="red", linewidth=2)
plt.show()

