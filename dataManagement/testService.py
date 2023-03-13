import json
from itineraryBuilder import getValidDestinations
from itineraryBuilder import itineraryBuilder


origin = "SAN"
destination = "RDU"
flight_date = "8/5/2022"
# df = getValidDestinations('faa', origin, flight_date)
# print(df.head(5))
    
df = itineraryBuilder('faa', origin, destination, flight_date, 60)
df['ConnectCity'] = df['SECOND_LEG_ORIG_CITY'] + " - " + df['FIRST_LEG_AIRLINE']
df['Initial Flight'] = round(df['FIRST_FLIGHT_DURATION']/60,1)
df['Connect Time'] = round(df['CONNECT_TIME']/60,1)
df['Final Flight'] = round(df['SECOND_FLIGHT_DURATION']/60,1)

df_output = df[['ConnectCity', 'Initial Flight', 'Connect Time', 'Final Flight']]
print(df_output.head(5))

mydict = df_output.to_dict('records')
jd = json.dumps(mydict)
print(jd)
print(type(jd))

