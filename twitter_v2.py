import json
# Package for using twitter api
import tweepy as tw
import mysql.connector


def get_oath():

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

    return api


def store_data(val):
    conn = mysql.connector.connect(
        host="rssnews-db.cmhk8qnm6hfj.us-east-2.rds.amazonaws.com",
        user="admin",
        password="bdat1007",
        database="data_mining"
    )
    mycursor = conn.cursor()
    sql = (
            "INSERT INTO data_mining.twitter_w_images (images, id, screen_name, location, created_date, text)" \
            " VALUES (%s, %s, %s, %s, %s, %s)"
            )
    added_data = 0
    try:
        mycursor.execute(sql, val)
        conn.commit()
        added_data = 1
    except:
        #print("Duplication error")
        conn.rollback()

    conn.close()

    return added_data


def main():
    tw_api = get_oath()

    # Define keywords and date to find data from twitter
    keywords = "#trump"
    date_since = "2020-05-01"

    # Gather tweets data
    tweets = tw.Cursor(tw_api.search,
                q=keywords,
                lang="en",
                since=date_since).items()

    # Store a list of tweets
    # tweet.entities.get('media', [])[0]['media_url']
    added_data = 0
    count_data = 0
    for tweet in tweets:
        check_image = tweet.entities.get('media', [])

        if len(check_image) > 0 :

            added_data += store_data((tweet.entities.get('media', [])[0]['media_url'],
                    tweet.user.id,
                    tweet.user.screen_name,
                    tweet.user.location,
                    tweet.user.created_at,
                    tweet.text))
            count_data += 1

    print(added_data + " of " + count_data + " are stored.")


if __name__ == "__main__":
    main()