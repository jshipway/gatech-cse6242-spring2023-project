# Load tornado module
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import requests
import json

import pandas
from itineraryBuilder import queryFlights
from itineraryBuilder import itineraryBuilder
from itineraryBuilder import getValidDestinations





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
    
    # self.set_header("Access-Control-Allow-Origin", "http://localhost:8000")
    # self.set_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept, Authorization")
    # self.set_header('Access-Control-Allow-Methods', 'GET,HEAD,OPTIONS,POST,PUT,DELETE')
    # self.set_status(204)   
    
    # self.set_header("access-control-allow-origin", "*")
    # self.set_header("Access-Control-Allow-Headers", "x-requested-with")
    # self.set_header('Access-Control-Allow-Methods', 'GET, PUT, DELETE, OPTIONS')
    # # HEADERS!
    # self.set_header("Access-Control-Allow-Headers", "access-control-allow-origin,authorization,content-type") 
    # self.set_status(204)
    
 
        
    def get(self):
        origin = listToString(self.get_arguments('origin'))
        print("Getting Destinatins for Origin: " + origin)
            
        flight_date = "8/5/2022"
        df = getValidDestinations('faa', origin, flight_date)    
        #print(df.head(5))
        
        dfjson = json.dumps(df.values.tolist())
        print(dfjson)
        
        self.write({'links':[],"name": "items","start": 0,"count": len(df.index), "items": dfjson,"limit": 20,"version": 2})


    def options(self, *args):
         # no body
         # `*args` is for route with `path arguments` supports
         print("in options")
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
    print("Flight Connections Risk Advisor Service, running on port 3000")
    IOLoop.instance().start()



def test():
    origin = "SAN"
    destination = "DTW"
    flight_date = "8/5/2022"
    df = getValidDestinations('faa', origin, flight_date)
    print(df.head(5))
    
        
    mylist = df.head(5).values.tolist()
    j = json.dumps(mylist)
    print(j)
    #df2 = queryFlights('faa', origin, destination, flight_date)
    #print(df2.head(5))
    #df3= itineraryBuilder(df, 60)

#test()
