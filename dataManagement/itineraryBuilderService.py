#import packages
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import json
import tornado
import os
import sys
import pandas as pd

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
class DestQueryHandler(RequestHandler):

  def set_default_headers(self):
    print("setting headers")
    self.set_header("Access-Control-Allow-Origin", "*")
    self.set_header("Access-Control-Allow-Headers", "x-requested-with")
    self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

  def get(self):
    origin = listToString(self.get_arguments('origin'))    
    #flight_date = "12/1/2019"
    flight_date = listToString(self.get_arguments('date'))
    #itinerary builder doesn't like leading zeroes in day
    if flight_date.index('0') == 3:
        flight_date = flight_date[:3] + flight_date[4:]
    print("Getting Destinatins for Origin: " + origin + " on date: " + flight_date)        
    df = getValidDestinations('faa_2019_12', origin, flight_date)        
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

  def set_default_headers(self):
    print("setting headers")
    self.set_header("Access-Control-Allow-Origin", "*")
    self.set_header("Access-Control-Allow-Headers", "x-requested-with")
    self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')


  def get(self):
    origin = listToString(self.get_arguments('origin'))
    dest = listToString(self.get_arguments('dest'))
    date_flight = listToString(self.get_arguments('date'))
    minlayover = int(listToString(self.get_arguments('connect_time')))
    dne = listToString(self.get_arguments('dne'))
    dnl = listToString(self.get_arguments('dnl'))
    ane = listToString(self.get_arguments('ane'))
    anl = listToString(self.get_arguments('anl'))
    anldate = listToString(self.get_arguments('anldate'))

    ## post-process date to remove any leading zeroes in month or day column
    tmp = date_flight.split('/')
    mo, day, year = tmp[0].lstrip("0"), tmp[1].lstrip("0"), tmp[2]
    date_flight = mo + '/' + day + '/' + year

    tmp = anldate.split('/')
    mo, day, year = tmp[0].lstrip("0"), tmp[1].lstrip("0"), tmp[2]
    anldate = mo + '/' + day + '/' + year

    print(origin)
    print(dest)
    print(date_flight)
    print(minlayover)
    print(dne)
    print(dnl)
    print(ane)
    print(anl)
    print(anldate)

    print("Getting Flights for: -" + origin + "-" + dest + "-" + date_flight + "-" + str(minlayover) + "-")
     
    df=itineraryBuilder('faa_2019_12', origin, dest, date_flight, anldate,\
                        minlayover,\
                        dep_no_earlier = dne,\
                        dep_no_later = dnl,\
                        arr_no_earlier = ane,\
                        arr_no_later = anl,\
                        orderby='duration')
      
    print(str(len(df.index)) + " records.")

    final_cols = ['ConnectCity', 'Initial Flight', 'Connection Time', 'Final Flight',\
                  'Trip Duration','Chance of Missed Connection', 'Time Lost if Missed',\
                  'Itinerary Risk', 'FIRST_LEG_ORIG', 'FIRST_LEG_DEST', 'SECOND_LEG_ORIG', 'SECOND_LEG_DEST']

    if len(df.index) > 0:

      df['ConnectCity'] = df['FIRST_LEG_AIRLINE'] + " thru: " +\
          df['SECOND_LEG_ORIG_CITY'] + "..Depart: " + df['FIRST_LEG_DEP_TIMESTAMP'].dt.strftime("%I:%M:%S %p") +\
              "..Arrive: " + df['SECOND_LEG_ARR_TIMESTAMP'].dt.strftime("%I:%M:%S %p")
      
      df['Initial Flight'] = round(df['FIRST_FLIGHT_DURATION']/60,3)
      df['Connection Time'] = round(df['CONNECT_TIME']/60,3)
      df['Final Flight'] = round(df['SECOND_FLIGHT_DURATION']/60,3)
      df['Trip Duration'] = round(df['TRIP_TIME']/60,3)
      df['Chance of Missed Connection'] = round(df['RISK_MISSED_CONNECTION'],4)
      df['Time Lost if Missed'] = round(df['NEXT_FLIGHT_TIMELOSS']/60,1)
      #df['Itinerary Risk'] = round(df['TOTAL_RISK']/60,1)
      df['Itinerary Risk'] = round(df['TOTAL_RISK'] / df['TRIP_TIME'], 3)
      df['Arrival Time'] = df['SECOND_LEG_ARR_TIMESTAMP'].dt.strftime('%S').astype('int')

      
      df_output = df[['ConnectCity', 'Initial Flight', 'Connection Time', 'Final Flight',\
                  'Trip Duration','Chance of Missed Connection', 'Time Lost if Missed',\
                  'Itinerary Risk', 'FIRST_LEG_ORIG', 'FIRST_LEG_DEST', 'SECOND_LEG_ORIG', 'SECOND_LEG_DEST', 'Arrival Time']]
      
    else:
       df_output = pd.DataFrame(columns=final_cols)
       
    
    print(df_output.head(len(df_output.index)))
    
    mydict = df_output.to_dict('records')
    dfjson = json.dumps(mydict)           
    self.write({'links':[],"name": "items","start": 0,"count": len(df.index), "items": dfjson,"limit": 20,"version": 2})

  def options(self, *args):
    # no body
    # `*args` is for route with `path arguments` supports
    self.set_status(204)
    self.finish()
  
  def write_error(self,status_code,**kwargs):
      if status_code == 500:
         self.write({'error':500, 'links':[],"name": "items","start": 0,"count": 0, "items": [],"limit": 20,"version": 2})



# define end points
def make_app():  
  web_root_path = os.path.dirname(__file__)
  web_root_path = web_root_path.replace("dataManagement", "webui")
  print("Serving index.html from ", web_root_path)
  urls = [(r"/destinations",DestQueryHandler),(r"/flights",FlightQueryHandler), (r"/(.*)", tornado.web.StaticFileHandler, {"path": web_root_path, "default_filename": "index.html"}) ]
  return Application(urls)

# Start server  
if __name__ == '__main__':
    app = make_app()
    argcount = len(sys.argv)    
    port = 80
    if argcount >= 2:
        arg = sys.argv[1]
        if arg.isdigit():            
            port=arg
    
    app.listen(port)    
    print("Flight Connections Risk Advisor Service, running on port ", port)
    IOLoop.instance().start()