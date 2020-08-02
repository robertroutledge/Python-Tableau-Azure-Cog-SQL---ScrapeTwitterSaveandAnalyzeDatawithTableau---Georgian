#install the following libraries to run the code
#install azure-ai-textanalytics (version 3)
#install sqlalchemy
#install pymysql
#install wordcloud
#install matplotlib
#install pandas

# import libraries
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
#needed to create wordcloud
from wordcloud import WordCloud, STOPWORDS
import re
import math


# frameworks for the Word Cloud:
# First of all, you need to download the c++ build tools for the VS 2019 (the size is 4.76 g)
# https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16


#authenticates with Azure for Sentiment Analysis
def authenticate_client():
    key = "a41d5581d6a146438b741ae0f5ba719a"
    endpoint = "https://eastus.api.cognitive.microsoft.com/"
    # paramaters needed for Azure Analytics
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return text_analytics_client

#runs sentiment analysis on tweets
def sentiment_analysis(tweets):
    tweeetlist = [tweets]
    response = client.analyze_sentiment(tweeetlist)
    keyphrase = client.extract_key_phrases(tweeetlist)
    list = [[response],[keyphrase]]
    return list

#function that gets just the text and image columns data from the database
#filters for tweets and images related to Donald Trump
def get_db_data():
    db_connection_str = 'mysql+pymysql://admin:bdat1007@rssnews-db.cmhk8qnm6hfj.us-east-2.rds.amazonaws.com/data_mining'
    db_connection = create_engine(db_connection_str)
    query= 'SELECT images, text FROM data_mining.twitter_w_images WHERE text like "%Donald Trump%" or text like "%Trump%" or text like "@donaldjtrump%"'
    print(query)
    df = pd.read_sql_query(text(query),con=db_connection)
    #tweets = df['text'].tolist()

    return df

#runs Azure cognitive analysis on tweets
def scoresource(alltweets):
    #These lists are needed to get the results of the sentiment analysis
    positive_score = []
    neutral_score = []
    negative_score  = []
    keyphrases = []
    #get the sentiment score overall
    for i in range(0,len(alltweets)):
        results = sentiment_analysis(alltweets[i])
        scores = results[0][0][0]
        phrases = results[1][0][0]
        positive_score.append(scores.confidence_scores.positive)
        neutral_score.append(scores.confidence_scores.neutral)
        negative_score.append(scores.confidence_scores.negative)
        keyphrases.append(phrases.key_phrases)
    tuples = list(zip(positive_score,neutral_score,negative_score,keyphrases))
    results_df = pd.DataFrame(tuples,columns=['Positive','Neutral','Negative','Key_Phrases'])
    return results_df

#creates the tweets dataframe
tweet_df = get_db_data()
tweet_text = tweet_df['text'].tolist()
client = authenticate_client()
tweet_scores_df = scoresource(tweet_text)

# going through the macleans_df:
def cleankeyphrases(df):
    comment_words = ''
    # typecaste every value to the string
    for val in df.Key_Phrases:
        val = str(val)
        # then split
        tokens = val.split()
        # Converts each token to lowercase
        for i in range(len(tokens)):
            tokens[i] = re.sub(r'[\',-./]|\sBD', r'', tokens[i]).lower()
        for words in tokens:
            comment_words = comment_words + words + ' '
    return comment_words

#create the string for wordcloud and generate word clouds
tweets_wordlcoud_string =cleankeyphrases(tweet_scores_df)
stopwords = set(STOPWORDS)
stopwords.update(["€™", "©"])
wordcloud_tweets = WordCloud(width = 800, height = 800,
                     background_color ='white',
                     stopwords = stopwords,
                     min_font_size = 10).generate(tweets_wordlcoud_string)


# create a wordcloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_tweets)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# Overall statistics for the pie
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# macleans:
labels = 'Positive', 'Neutral', 'Negative'
sizes = [tweet_scores_df['Positive'].mean(), tweet_scores_df['Neutral'].mean(), tweet_scores_df['Negative'].mean()]
explode = (0.1, 0.1, 0.1)
colors = ['lightgreen', 'orange', 'orangered']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
