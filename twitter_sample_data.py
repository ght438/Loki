# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:51:41 2015

@author: GHT438
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:53:00 2015

@author: bxz747
"""

import tweepy
from tweepy import OAuthHandler

#For streaming
"""
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)
        
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False
"""

ckey="Llo2OnotASKkzOHRwj90ccLOW"
csecret="LGjSuYEBMmdpOyGZZwYCoxxNJVe3QsigxMshsbbZ2GI97dkomg"
atoken="1648017836-zv7QLW9N2dXCo8l517gfDUdI2J8PrhczGaIYfzT"
asecret="jLagsSrXKYvVml0zmcfrdmn4miSZ237N7z9xhQp4EaEqb"

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api = tweepy.API(auth)
track=['"Capital One"', 'CapitalOne', '"Cap1 Card"','"Venture Card"', 'VentureOne', '"Quicksilver Card"', 'QuicksilverOne']

txt=[]
for word in track:
    for tweet in tweepy.Cursor(api.search,q=word,show_user=True).items(100):
        txt.append(tweet.text)

with open('txtfile.txt','w+') as f:
    for l in txt:
        f.write("%s\n" % str(l.encode('utf-8')))


#This is for streaming
#myStream = tweepy.Stream(auth = api.auth, listener=MyStreamListener())
#myStream.filter(track=track)

#print api.rate_limit_status()