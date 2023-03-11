# Load tornado module
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import requests
import json

from itineraryBuilder import queryFlights
from itineraryBuilder import itineraryBuilder
from itineraryBuilder import getValidDestinations

#load text analytics module
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1.replace(" ","%20")

def strip_html_tags(s):
    str=s.replace("<b>", "")
    str=str.replace("<\/b>", "")
    str=str.replace("</b>", "")   
    str=str.replace("\r","")
    str=str.replace("\n","")
    str=str.replace("&quot;","")
    str=str.replace("\u0092","") 
    return str


def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"
    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"
    else :
        overall_sentiment = "Neutral"
    return overall_sentiment

# URL request handler
class QueryHandler(RequestHandler):

  def set_default_headers(self):
    print("setting headers")
    self.set_header("Access-Control-Allow-Origin", "*")
    self.set_header("Access-Control-Allow-Headers", "x-requested-with")
    self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

  def get(self):
    sname = listToString(self.get_arguments('origin'))
    strip_tags = listToString(self.get_arguments('strip_tags'))
    headers = {'Content-Type': 'application/json'}
    search_api_url = 'https://www.oneclickcompliance.com/api/json-search.php?search=1&user=SASdemo&results=20&start=1&type=and&query=' + sname
    response = requests.get(search_api_url, headers=headers, timeout=5)
    item_dict = response.json()
    cnt=len(item_dict['results'])
    i = 0
    while (i < cnt):
      item_dict['results'][i]['sentiment'] = sentiment_vader(item_dict['results'][i]['text'])
      if strip_tags == "1":
        item_dict['results'][i]['title'] = strip_html_tags(item_dict['results'][i]['title'])
        item_dict['results'][i]['text'] = strip_html_tags(item_dict['results'][i]['text'])
      i=i+1
    self.write({'links':[],"name": "items","start": 0,"count": cnt, "items": item_dict['results'],"limit": 20,"version": 2})

  def options(self, *args):
    # no body
    # `*args` is for route with `path arguments` supports
    self.set_status(204)
    self.finish()
  
  def write_error(self,status_code,**kwargs):
      if status_code == 500:
         self.write({'links':[],"name": "items","start": 0,"count": 0, "items": [],"limit": 20,"version": 2})

# define end points
def make_app():
  urls = [(r"/query",QueryHandler)]
  return Application(urls)

# Start server  
if __name__ == '__main__':
    app = make_app()
    app.listen(3000)
    print("Flight Connections Risk Advisor, running on port 3000")
    IOLoop.instance().start()