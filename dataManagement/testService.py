# Load tornado module
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import requests
import json

from itineraryBuilder import *


def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1.replace(" ","%20")


# URL request handler
class QueryHandler(RequestHandler):

  def set_default_headers(self):
    print("setting headers")
    self.set_header("Access-Control-Allow-Origin", "*")
    self.set_header("Access-Control-Allow-Headers", "x-requested-with")
    self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

  def get(self):
    origin = listToString(self.get_arguments('origin'))
    print("Getting Destinatins for Origin: " + origin)        
    flight_date = "8/5/2022"
    df = getValidDestinations('faa', origin, flight_date)        
    dfjson = json.dumps(df.values.tolist())
    print(dfjson)
    self.write({'links':[],"name": "items","start": 0,"count": len(df.index), "items": dfjson,"limit": 20,"version": 2})

  def options(self, *args):
    # no body
    # `*args` is for route with `path arguments` supports
    self.set_status(204)
    self.finish()
  
  def write_error(self,status_code,**kwargs):
      if status_code == 500:
         self.write({'links':[],"name": "items","start": 0,"count": 0, "items": [],"limit": 20,"version": 2})


class FlightQueryHandler(RequestHandler):

  # def set_default_headers(self):
  #   print("setting headers")
  #   self.set_header("Access-Control-Allow-Origin", "*")
  #   self.set_header("Access-Control-Allow-Headers", "x-requested-with")
  #   self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

  def get(self):
    origin = listToString(self.get_arguments('origin'))
    dest = listToString(self.get_arguments('dest'))
    date = listToString(self.get_arguments('date'))
    minlayover = int(listToString(self.get_arguments('connect_time')))
    print("Getting Flights for Origin: -" + origin + "-" + dest + "-" + date + "-" + str(minlayover) + "-")
     
    df=itineraryBuilder('faa', origin, dest, date, minlayover)
    print(df.head(10))
    dfjson = json.dumps(df.values) #.tolist())    
    print(dfjson)
    #dfjson='{}'    
    self.write({'links':[],"name": "items","start": 0,"count": 0, "items": dfjson,"limit": 20,"version": 2})

  def options(self, *args):
    # no body
    # `*args` is for route with `path arguments` supports
    self.set_status(204)
    self.finish()
  
  def write_error(self,status_code,**kwargs):
      if status_code == 500:
         self.write({'error':500, 'links':[],"name": "items","start": 0,"count": 0, "items": [],"limit": 20,"version": 2})





# # define end points
# def make_app():
#   urls = [(r"/query",QueryHandler),(r"/flights",FlightQueryHandler) ]
#   return Application(urls)

# # Start server  
# if __name__ == '__main__':
#     app = make_app()
#     app.listen(3000)
#     print("Flight Connections Risk Advisor, running on port 3000")
#     IOLoop.instance().start()
    
    


origin = "SAN"
destination = "RDU"
flight_date = "8/5/2022"
df = getValidDestinations('faa', origin, flight_date)
print(df.head(5))
    
df2 = itineraryBuilder('faa', origin, destination, flight_date, 60)
df2.drop(list(df2.filter(regex = 'TIMESTAMP')), axis = 1, inplace = True)
mydict = df2.to_dict('records')
jd = json.dumps(mydict)
print(jd)
print(type(jd))

