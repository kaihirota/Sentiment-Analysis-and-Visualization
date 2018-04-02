import sys
import os
import re
import sqlite3
from scipy import stats
import numpy as np
import pandas as pd
import cufflinks as cf  # for pandas - plotly compatibility

# vizualization modules
import matplotlib.pyplot as plt  # % matplotlib notebook
import plotly.plotly as py
from plotly.graph_objs import Scatter, Figure, Layout, Pie
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# plotly.tools.set_credentials_file(username=config.username, api_key=config.api_key)
init_notebook_mode(connected=True)

# twitter API + sentiment analysis
import tweepy
import config
from nltk.tokenize import word_tokenize  # sentiment analysis
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize.moses import MosesDetokenizer

# connect to twitter
auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tweepy.API(auth)

# stop words
stop_words = set(stopwords.words('english'))
# connect to SQL database
conn = sqlite3.connect('tweets.db')
# colors
piecolors = ['#8BFF88', '#3691FF', '#FF5353']
# positive, negative, compound colors in order
colors = ['#3892FF', '#FF387A', '#5000D1']


def insertHash(keyword):
    if keyword[:4] == 'HASH':
        return "#" + keyword[4:]
    else:
        return keyword


def stripHash(keyword):
    if keyword[0] == "#":
        return 'HASH' + keyword[1:]
    else:
        return keyword


def collectTweets(keyword, num):
    # exclude retweets
    keywordF = keyword + ' -filter:retweets'

    # collect tweets
    collectedTweets = []
    for tweet in tweepy.Cursor(api.search, q=keywordF, tweet_mode='extended', lang='en', rpp=5, show_user=True).items(num):
        collectedTweets.append(tweet)

    # convert to DataFrame
    df = toDataFrame(collectedTweets)
    toSQL(df, keyword)
    return


def toDataFrame(tweets):
    # convert to dataframe
    DataSet = pd.DataFrame()

    # add parameters
    DataSet['tweetID'] = [tweet.id for tweet in tweets]
    DataSet['datetime'] = [tweet.created_at for tweet in tweets]
    DataSet['date'] = DataSet.datetime.dt.date
    DataSet['hour'] = DataSet.datetime.dt.hour
    DataSet['minute'] = DataSet.datetime.dt.minute
    DataSet['dayofweek'] = DataSet.datetime.dt.weekday_name
    DataSet['tweetRetweetCt'] = [tweet.retweet_count for tweet in tweets]
    DataSet['tweetFavoriteCt'] = [tweet.favorite_count for tweet in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['userID'] = [tweet.user.id for tweet in tweets]
    DataSet['userScreen'] = [tweet.user.screen_name for tweet in tweets]
    DataSet['userName'] = [tweet.user.name for tweet in tweets]
    DataSet['userCreateDt'] = [tweet.user.created_at for tweet in tweets]
    DataSet['userDesc'] = [tweet.user.description for tweet in tweets]
    DataSet['userFollowerCt'] = [
        tweet.user.followers_count for tweet in tweets]
    DataSet['userFriendsCt'] = [tweet.user.friends_count for tweet in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]
    DataSet['userTimezone'] = [tweet.user.time_zone for tweet in tweets]
    DataSet['tweetText'] = [tweet.full_text for tweet in tweets]

    # tokenize tweetsText, and filter for stop words
    detokenizer = MosesDetokenizer()
    noStopWords = []
    for i in tweets:
        word_tokens = word_tokenize(i.full_text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        noStopWords.append(detokenizer.detokenize(
            filtered_sentence, return_str=True))
    DataSet['tweetNoSW'] = noStopWords

    # sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    DataSet['sentimentPos'] = [analyzer.polarity_scores(
        tweet)['pos'] for tweet in DataSet['tweetNoSW']]
    DataSet['sentimentNeut'] = [analyzer.polarity_scores(
        tweet)['neu'] for tweet in DataSet['tweetNoSW']]
    DataSet['sentimentNeg'] = [analyzer.polarity_scores(
        tweet)['neg'] for tweet in DataSet['tweetNoSW']]
    DataSet['sentimentComp'] = [analyzer.polarity_scores(
        tweet)['compound'] for tweet in DataSet['tweetNoSW']]
    return DataSet


def toSQL(df, keyword):
    # export to ./tweets.db
    df.to_sql(stripHash(keyword), conn, if_exists='append', index=False)
    return


def toCSV(keyword):
    # export to Data/keyword.csv and append if file exists
    df = readSQL(keyword)
    outpath = 'Data/' + keyword + '.csv'

    if os.path.exists(outpath):
        with open(outpath, 'a') as outfile:
            df.to_csv(outfile, header=False, index=False)
    else:
        df.to_csv(outpath, header=True, index=False)
    return


def readSQL(keyword):
    # import the entire table from tweets.db
    # one table = one type of keyword used in the past
    return pd.read_sql_query('select * from \"{}\";'.format(stripHash(keyword)), conn)


def analyzeTweets(keyword, option):
    # once in analyze mode, determine which analysis / visualization to do
    df = readSQL(keyword)
    df['datetime'] = pd.to_datetime(df['datetime'])
    n = len(df)

    if option == 'stats':
        stats(df, stripHash(keyword))
        return
    elif option == 'interval':
        interval(df, keyword, n)
        return
    elif option == 'line':
        line(df, keyword, n)
        return
    elif option == 'dist':
        dist(df, keyword, n)
        return
    elif option == 'scatter':
        scatter(df, keyword, n)
        return
    elif option == 'pie':
        pie(keyword)
        return
    elif option == 'map':
        sentMap(df, keyword)
        return


def stats(df, keyword):
    # export df.describe() as csv file inside ./Data
    dates = df.loc[df.iloc[[0, -1]].index, ['datetime']]
    dates = dates.iloc[::-1]
    df = df.drop(['tweetID', 'hour', 'minute', 'userID'], axis=1)
    stat = df.describe()
    keyword = 'Data/' + keyword + '_stats' + '.csv'

    if os.path.exists(keyword):
        with open(keyword, 'a') as outfile:
            dates.to_csv(outfile, header=False, index=False)
            stat.to_csv(outfile, header=True, index=True)
    else:
        dates.to_csv(keyword, header=False, index=False)
        stat.to_csv(keyword, header=True, index=True)
    return


def interval(df, keyword, n):
    # visualize and save as png
    # resample and aggregate data into interval of length n//30
    # so that graph is not messy and can be analyzed with ease
    interval = n // 30
    df = df.set_index('datetime')
    df = df.rolling(interval).sum()
    df = df.fillna(method='ffill', limit=1)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['sentimentPos'], label='Positive')
    plt.plot(df.index, df['sentimentNeg'], label='Negative')
    plt.plot(df.index, df['sentimentComp'], label='Compound')

    plt.title('Sentiment Analysis (Keyword={}, Sample Size={}, Interval={})'.format(
        keyword, n, interval))
    plt.xlabel('Tweet DateTime')
    plt.ylabel('Sentiment Score')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc=1, ncol=4, borderaxespad=0.)
    plt.fill_between(
        df.index, 0, df['sentimentComp'], facecolor='green', alpha=0.3)
    plt.fill_between(
        df.index, 0, df['sentimentPos'], facecolor='blue', alpha=0.4)
    plt.fill_between(
        df.index, 0, df['sentimentNeg'], facecolor='red', alpha=0.4)
    plt.savefig('Data/' + stripHash(keyword) + '.png')
    plt.show()
    return


def line(df, keyword, n):
    interval = n // 30
    df = df.set_index('datetime')
    df = df.rolling(interval).sum()
    df = df.fillna(method='ffill', limit=1)

    # plots line graph in an interactive html file
    # user can toggle each input variable on and off, or zoom in/out
    trace = Scatter(x=df.index, y=df.loc[df['sentimentPos'] != 0, 'sentimentPos'],
                    mode='lines+markers', name='Positive', fill='tozeroy',
                    marker=dict(color=colors[0]))
    trace1 = Scatter(x=df.index, y=df.loc[df['sentimentNeg'] != 0, 'sentimentNeg'],
                     mode='lines+markers', name='Negative', fill='tozeroy',
                     marker=dict(color=colors[1]))
    trace2 = Scatter(x=df.index, y=df.loc[df['sentimentComp'] != 0, 'sentimentComp'],
                     mode='lines+markers', name='Compound', fill='tozeroy',
                     visible='legendonly', opacity=0.1,
                     marker=dict(color=colors[2]))

    data = [trace, trace1, trace2]
    layout = dict(title='Sentiment Analysis (Keyword={}, Sample Size={}, Interval={})'.format(
        keyword, n, interval), yaxis=dict(zeroline=False), xaxis=dict(zeroline=False))
    fig = dict(data=data, layout=layout)
    plot(fig, filename='Data/' + stripHash(keyword) + '_line' + '.html')
    return


def dist(df, keyword, n):
    # exclude tweets with score of 0
    # otherwise, graph is difficult to analyze
    hist_data = [df.loc[df['sentimentPos'] != 0, 'sentimentPos'],
                 df.loc[df['sentimentNeg'] != 0, 'sentimentNeg']]
    group_labels = ['Positive', 'Negative']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.05,
                             curve_type='normal', colors=[colors[0], colors[1]])

    # Add title
    fig['layout'].update(
        title='Sentiment Analysis: {} (Sample size:{})'.format(keyword, n))

    plot(fig, filename='Data/' + stripHash(keyword) + '_dist' + '.html')
    return


def scatter(df, keyword, n):
    # plot positive score against negative score to look or clusters / trends
    dfpos = df.loc[df['sentimentPos'] > 0, ['sentimentPos']]
    dfneg = df.loc[df['sentimentNeg'] > 0, ['sentimentNeg']]

    trace = Scatter(x=dfpos['sentimentPos'], y=dfneg['sentimentNeg'],
                    mode='markers', name='Positive / Negative Sentiment',
                    marker=dict(size='14', color=colors[2],
                                line=dict(width=1, color='rgb(255, 255, 255)')))
    data = [trace]
    layout = dict(title='Sentiment Analysis: {} (Sample size:{})'.format(keyword, n),
                  xaxis=dict(title='Positive',
                             color=colors[0], rangemode='tozero'),
                  yaxis=dict(title='Negative', color=colors[1], rangemode='tozero'))

    fig = dict(data=data, layout=layout)
    plot(fig, filename='Data/' + stripHash(keyword) + '_scatter' + '.html')
    return


def pie(keyword):
    df = pd.read_sql_query(
        'select count(*) as count, sum(sentimentPos) as pos, sum(sentimentNeut) as neut, sum(sentimentNeg) as neg from {};'.format(stripHash(keyword)), conn)
    val = []
    for i in df:
        val.append(df[i][0])

    n = val[0]
    colors = piecolors
    fig = {
        "data": [{
            "labels": ['Positive', 'Neutral', 'Negative'],
            "values": val[1:],
            "hoverinfo":"label+percent",
            "hoverlabel":dict(font=dict(size=[24, 24, 24])),
            "hole": .4,
            "type": "pie",
            "marker": dict(colors=colors)
        }],
        "layout": {
            "title": "Sentiment Score (Sample size:{})".format(n),
            "annotations": [{
                "font": {"size": 28},
                "showarrow": False,
                "text": keyword,
            }]}}
    plot(fig, filename='Data/' + stripHash(keyword) + '_pie' + '.html')
    return


def sentMap(df, keyword):
    df, n = parseStates(df)

    scl = [[0.0, 'rgb(255, 51, 51)'], [0.5, 'rgb(255,255,255)'], [
        1.0, 'rgb(51, 102, 255)']]

    data = [dict(type='choropleth',
                 colorscale=scl,
                 autocolorscale=False,
                 locations=df['abb'],
                 z=df['sum'] / df['samplesize'],
                 locationmode='USA-states',
                 text="State: " + df['state'] + '<br>' +
                 "Sample Size: " + df['samplesize'].astype(str),
                 marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                 colorbar=dict(title="Average Sentiment"))]

    layout = dict(
        title='Sentiment Analysis of {} by State (Sample size:{})'.format(
            keyword, n),
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'),
    )

    fig = dict(data=data, layout=layout)
    plot(fig, filename='Data/' + stripHash(keyword) + '_map' + '.html')
    return


def parseStates(df):
    example_dictionary = {}
    n = 0
    for i in range(len(df)):
        state_abb = re.search(" [A-Z][A-Z]$", str(df.iloc[i]['userLocation']))
        if state_abb is not None and state_abb.group(0)[1:] in config.States:
            example_dictionary.setdefault(state_abb.group(
                0)[1:], []).append(df.iloc[i]['sentimentComp'])
            n += 1

    stateList = []
    for i in example_dictionary:
        stateList.append((config.Statesdict[i], i, sum(
            example_dictionary[i]), len(example_dictionary[i])))
    return pd.DataFrame(stateList, columns=['state', 'abb', 'sum', 'samplesize']), n


def main():
    # parse command line arguments
    mode = sys.argv[1]
    keyword = sys.argv[2]
    if len(sys.argv) == 4:
        option = sys.argv[3]

    # determine which mode to enter
    if mode == 'collect':
        collectTweets(insertHash(keyword), int(option))
    elif mode == 'analyze':
        analyzeTweets(insertHash(keyword), option)
    elif mode == 'export':
        toCSV(stripHash(keyword))

    print('Task successfully completed')
    sys.exit(0)


if __name__ == "__main__":
    main()
