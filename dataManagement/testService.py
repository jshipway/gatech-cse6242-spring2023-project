import json
from itineraryBuilder import getValidDestinations
from itineraryBuilder import itineraryBuilder


origin = "SAN"
destination = "RDU"
flight_date = "8/5/2022"
# df = getValidDestinations('faa', origin, flight_date)
# print(df.head(5))
    
df = itineraryBuilder('faa', origin, destination, flight_date, 60, orderby='duration')
#df['ConnectCity'] = df['SECOND_LEG_ORIG_CITY'] + " - " + df['FIRST_LEG_AIRLINE']
df['ConnectCity'] = df['FIRST_LEG_AIRLINE'] + " " + df['FIRST_LEG_DEP_TIME'] + " " + df['SECOND_LEG_ORIG_CITY']
df['Initial Flight'] = round(df['FIRST_FLIGHT_DURATION']/60,1)
df['Connection Layover'] = round(df['CONNECT_TIME']/60,1)
df['Final Flight'] = round(df['SECOND_FLIGHT_DURATION']/60,1)
df['total'] = round(df['TRIP_TIME']/60,1)

print(str(len(df.index)) + " records.")
df_output = df[['ConnectCity', 'Initial Flight', 'Connection Layover', 'Final Flight', 'total']]
print(df_output.head(len(df.index)))

#mydict = df_output.to_dict('records')
#jd = json.dumps(mydict)
#print(jd)
#print(type(jd))

