import json
# Package for using twitter api
import tweepy as tw

# Open json file for reading twitter's keys
with open('twitter_keys.json', 'r') as f:
    keys = json.load(f)

# Put the keys into each variable - for avoiding to expose important information
api_key= keys["API key"]
api_secret= keys["API secret key"]
access_token= keys["Access token"]
access_token_secret= keys["Access token secret"]

# Set the oauth to access tiwtter api
auth = tw.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define keywords and date to find data from twitter
keywords = "#justin trudeau"
date_since = "2020-05-01"

# Gather tweets data
tweets = tw.Cursor(api.search,
              q=keywords,
              lang="en",
              since=date_since).items(10)

# Store a list of tweets
tweet_data1 = [tweet.text for tweet in tweets] 
print(tweet_data1)

# Remove duplicated tweets by filtering retweets
# https://developer.twitter.com/en/docs/tweets/rules-and-filtering/overview/standard-operators
filtered_keywords = keywords + " -filter:retweets"

filtered_tweets = tw.Cursor(api.search,
                       q=filtered_keywords,
                       lang="en",
                       since=date_since).items(10)

tweet_data2 = [tweet.text for tweet in filtered_tweets]
print(tweet_data2)