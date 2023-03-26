# Load tornado module
# Load tornado module
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import json

from itineraryBuilder import itineraryBuilder
from itineraryBuilder import getValidDestinations
import pandas as pd


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

    print("Getting Flights for: -" + origin + "-" + dest + "-" + date_flight + "-" + str(minlayover) + "-")
     
    df=itineraryBuilder('faa', origin, dest, date_flight, minlayover,\
                        dep_no_earlier = dne,\
                        dep_no_later = dnl,\
                        arr_no_earlier = ane,\
                        arr_no_later = anl,\
                        orderby='duration')
      
    print(str(len(df.index)) + " records.")

    final_cols = ['ConnectCity', 'Initial Flight', 'Connection Layover', 'Final Flight',\
                  'Trip Duration','Chance of Missed Connection', 'Time Lost if Missed',\
                  'Itinerary Risk', 'FIRST_LEG_ORIG', 'FIRST_LEG_DEST', 'SECOND_LEG_ORIG', 'SECOND_LEG_DEST']

    if len(df.index) > 0:

      df['ConnectCity'] = df['FIRST_LEG_AIRLINE'] + " thru: " +\
          df['SECOND_LEG_ORIG_CITY'] + "..Depart: " + df['FIRST_LEG_DEP_TIMESTAMP'].dt.strftime("%I:%M:%S %p") +\
              "..Arrive: " + df['SECOND_LEG_ARR_TIMESTAMP'].dt.strftime("%I:%M:%S %p")
      
      df['Initial Flight'] = round(df['FIRST_FLIGHT_DURATION']/60,1)
      df['Connection Layover'] = round(df['CONNECT_TIME']/60,1)
      df['Final Flight'] = round(df['SECOND_FLIGHT_DURATION']/60,1)
      df['Trip Duration'] = round(df['TRIP_TIME']/60,1)
      df['Chance of Missed Connection'] = round(df['RISK_MISSED_CONNECTION'],3)
      df['Time Lost if Missed'] = round(df['NEXT_FLIGHT_TIMELOSS']/60,1)
      df['Itinerary Risk'] = round(df['TOTAL_RISK']/60,1)
      
      df_output = df[['ConnectCity', 'Initial Flight', 'Connection Layover', 'Final Flight',\
                  'Trip Duration','Chance of Missed Connection', 'Time Lost if Missed',\
                  'Itinerary Risk', 'FIRST_LEG_ORIG', 'FIRST_LEG_DEST', 'SECOND_LEG_ORIG', 'SECOND_LEG_DEST']]
      
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
  urls = [(r"/destinations",DestQueryHandler),(r"/flights",FlightQueryHandler) ]
  return Application(urls)

# Start server  
if __name__ == '__main__':
    app = make_app()
    app.listen(3000)
    print("Flight Connections Risk Advisor, running on port 3000")
    IOLoop.instance().start()
    


# origin = "SAN"
# destination = "RDU"
# flight_date = "8/5/2022"
# df = getValidDestinations('faa', origin, flight_date)
# print(df.head(5))
    
# df2 = itineraryBuilder('faa', origin, destination, flight_date, 60)
# df2.drop(list(df2.filter(regex = 'TIMESTAMP')), axis = 1, inplace = True)
# mydict = df2.to_dict('records')
# jd = json.dumps(mydict)
# print(jd)
# print(type(jd))