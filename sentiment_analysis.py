#importing components: 1. urlopen, Request to open URL link 2. bs4 to get the data 3. NLTK to vectorize the headlines

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

#to make the "engine" more flexible I created a variable finviz_url with part of te URL. The other, missing part is a ticker. Because in future I'd like to make analysis for a different assets, I made tickers contained in the list object.

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers =['AMZN', 'FB', 'GOOG']

#now we are iterating through the tickers. First we are creating full_url by summing finviz_url with the ticker

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

#create a var req - request object (headers: user-agent value used by browser to identify itself, my-app - thats just a given name, response which actually opens the url

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

#to use BS we need to provide two elements, HTML string and what parser type to use EDIT the most popular parser is xlml but in this case that is enough to use html one

    html = BeautifulSoup(response, 'html')

#looking for te element that holds headlines on te page (news)

    news_table = html.find(id='news-table')

#we are building news_tables dictionary by defining a key (ticker and a value (news_table)). This is how the data is being loaded from the website. Now we have contained raw html tr element which is being put in the dictionary with the correct key-ticker.

    news_tables[ticker] = news_table

#create a parsed_data list, its gonna be a list of lists (i.e. [AMZ, 01-01-2021, 04:50, 'text'])

parsed_data =[]

#iterating through the html structure to filter the data we are intrested in, and apending them ti parsed_data object

for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):

        tittle = row.a.text
        #because date has two elements on the page date and time, we need to plit the data by a space ' ', the forst element is a date the second is a time itself
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, tittle])

#defining DataFrame

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'tittle'])

#implementing the vectorizer in this case SentimentIntensityAnalyzer by applying it as the function of the tittle, then added results to compund cplumn

vader = SentimentIntensityAnalyzer()
f = lambda x: vader.polarity_scores(x)['compound']
df['compound'] = df['tittle'].apply(f)
df['date'] = pd.to_datetime(df.date).dt.date

#plotting the results of the project as a graph (bar chart looks the best for the data we are having)

plt.figure(figsize=(10,8))
#quick manipulation for plotting purposes
#1. grouping the data by ticker and the date
mean_df = df.groupby(['ticker', 'date']).mean()
#2. pivoting the level of the data
mean_df = mean_df.unstack()
#3. returning the corss section , in simple words getting rid of 'compound text' that is set as first column
mean_df = mean_df.xs('compound', axis='columns').transpose()
#4. actually plotting the chart
mean_df.plot(kind='bar')
plt.show()
