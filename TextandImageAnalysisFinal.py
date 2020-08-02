#install the following libraries to run the code
#install azure-ai-textanalytics (version 3)
#install sqlalchemy
#install pymysql
#install pandas
#pip install azure-cognitiveservices-vision-customvision
#pip install azure-cognitiveservices-vision-computervision
#pip3 install time
#pip3 isntall mysql.connector

# import libraries
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
#needed to create wordcloud
from wordcloud import WordCloud, STOPWORDS
import re
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
import time
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import mysql.connector

#authenticates with Azure for Sentiment Analysis
def authenticate_text_client():
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
    df = pd.read_sql_query(text(query),con=db_connection)
    return df

#runs Azure cognitive analysis on tweets
def scoresource(alltweets,images):
    #These lists are needed to get the results of the sentiment analysis
    results_df = pd.DataFrame(images,columns=['images'])
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
    newkeyphrases = []
    for item in keyphrases:
        words = ' '.join(map(str,item))
        newkeyphrases.append(words)
    #tuples = list(zip(alltweets,positive_score,neutral_score,negative_score,keyphrases,images))
    results_df['tweettext'] = alltweets
    results_df['positive'] = positive_score
    results_df['neutral'] = neutral_score
    results_df['negative'] = negative_score
    results_df['keyphrases'] = newkeyphrases
    #results_df = pd.DataFrame(tuples,columns=['tweettext','positive','neutral','negative','keyphrases','images'])
    return results_df

# cleans the tweets of punctuation
def cleankeyphrasesforwordcloud(df):
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

#creates the tweets dataframe
tweet_df = get_db_data()
tweet_text = tweet_df['text'].tolist()
image_urls = tweet_df['images'].tolist()
client = authenticate_text_client()
tweet_df = scoresource(tweet_text,image_urls)

ENDPOINT = "https://dataminingasgfour-prediction.cognitiveservices.azure.com/"
training_key = "08acd5cf3c8948829673c53d930d2f50"
prediction_key = "dfe4b7746ad74c1f98b130d7bfc5048a"
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)


#constants for the prediction model and tags
#libraries needed to import
import urllib
from bs4 import BeautifulSoup
import requests
project = trainer.create_project("TrumpAnalytics")
prediction_resource_id = "/subscriptions/31dad0b5-b624-4e49-815e-51e72ad9f6e4/resourceGroups/GeorgianCollege/providers/Microsoft.CognitiveServices/accounts/DataMiningAsgFour-Prediction"
publish_iteration_name = "classifyModel"
image_list = []

#First we train our model to predict if Trump is in an image

#collecting and counting the number of training images
training_image_url = r"https://github.com/robertroutledge/DataMiningFinalAssignment/raw/master/DocumentationandArchive/TrumpImageLibrary/"
page = requests.get(training_image_url)
soup = BeautifulSoup(page.text,'html.parser')
soupstring = str(soup)
num_training_images = soupstring.count("Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item")
trump_tag = trainer.create_tag(project.id, "Trump")

#iterates through the images, adds a tag
for x in range(1,num_training_images+1):
    name = "image{}.jpg".format(x)
    imagelink = training_image_url + name
    try:
        resource = urllib.request.urlopen(imagelink)
    except:
        print("one of the files doesn't exists")
    image_list.append(ImageFileCreateEntry(name=name, contents=resource.read(), tag_ids=[trump_tag.id]))

#uploads tagged images to Azure Custom Vision
upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
if not upload_result.is_batch_successful:
    print("Image batch upload failed.")
    for image in upload_result.images:
        print("Image status: ", image.status)
    exit(-1)

#train the model with the images uploaded to azure ai
iteration = trainer.train_project(project.id)
while iteration.status != "Completed":
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The model is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)

#Now we prepare the prediction based on training model above
#get list of URLs scraped from twitter with images to predict
prediction_images = tweet_df['images']
results_list = []


#authenticate with prediction engine
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

#make predictions, save to list for tweet_df
for image in prediction_images:
    try:
        resource = urllib.request.urlopen(image)
        results = predictor.classify_image(project.id, publish_iteration_name, resource.read())
        # Display the results.
        for prediction in results.predictions:
            #x is for screen display
            x = "\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100)
            print(x)
            # y is to add score to tweet_df
            y = "{0:.2f}%".format(prediction.probability * 100)
            results_list.append(y)
    except:
        print("one of the file doesn't exists or is difficult to predict")
        results_list.append("image doesn't exist or couldn't predict")

print("the algorithm is finished")

#append score result to tweet_df
tweet_df['likelihoodtrumpimage'] = results_list

#send the data to a new SQL table
conn = mysql.connector.connect(
    host="rssnews-db.cmhk8qnm6hfj.us-east-2.rds.amazonaws.com",
    user="admin",
    password="bdat1007",
    database="data_mining"
)
mycursor = conn.cursor()
sql = ("DROP TABLE IF EXISTS analyzeddata")
mycursor.execute(sql)
conn.commit()
engine = create_engine('mysql+mysqlconnector://admin:bdat1007@rssnews-db.cmhk8qnm6hfj.us-east-2.rds.amazonaws.com:3306/data_mining', echo=False)
conn = engine.connect()
tweet_df.to_sql('analyzeddata',con=conn)



