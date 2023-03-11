import sqlite3
import pandas as pd
import pytz
import numpy as np # used for dummy ML only


def queryFlights(db_name: str,\
                    ORIG: str,\
                    DEST: str,\
                    DATE: str,\
                    dep_no_earlier = '1',\
                    dep_no_later = '2359',\
                    arr_no_earlier = '1',\
                    arr_no_later = '2359'):
    
    origin = "'" + ORIG + "'"
    destination = "'" + DEST + "'"
    flight_date = "'" + DATE + "'"
    next_date = "'" + getNextDate(DATE) + "'"

    conn = sqlite3.connect(db_name)

    sql = f'''SELECT
                    start.Airline AS FIRST_LEG_AIRLINE,
                    start.Origin AS FIRST_LEG_ORIG,
                    start.OriginCity AS FIRST_LEG_ORIG_CITY,
                    start.Dest AS FIRST_LEG_DEST,
                    start.DestCity AS FIRST_LEG_DEST_CITY,
                    start.Date AS FIRST_LEG_DATE,
                    start.FlightNum AS FIRST_LEG_FLIGHT_NUM,
                    start.DepTime AS FIRST_LEG_DEP_TIME,
                    start.ArrTime AS FIRST_LEG_ARR_TIME,
                    finish.Airline AS SECOND_LEG_AIRLINE,                    
                    finish.Origin AS SECOND_LEG_ORIG,
                    finish.OriginCity AS SECOND_LEG_ORIG_CITY,
                    finish.Dest AS SECOND_LEG_DEST,
                    finish.DestCity AS SECOND_LEG_DEST_CITY,
                    finish.Date AS SECOND_LEG_DATE,
                    finish.FlightNum AS SECOND_LEG_FLIGHT_NUM,
                    finish.DepTime AS SECOND_LEG_DEP_TIME,
                    finish.ArrTime AS SECOND_LEG_ARR_TIME
                FROM 
                    (SELECT
                        IATA_Code_Marketing_Airline AS Airline,
                        Origin,
                        OriginCityName AS OriginCity,
                        Dest,
                        DestCityName AS DestCity,
                        FlightDate AS Date,
                        Flight_Number_Operating_Airline AS FlightNum,
                        CRSDepTime AS DepTime,
                        CRSArrTime AS ArrTime
                    FROM faa
                    WHERE
                        Origin = {origin} AND
                        Dest <> {destination} AND
                        FlightDate = {flight_date} AND
                        CAST(DepTime AS int) BETWEEN {dep_no_earlier} AND {dep_no_later}) AS start,
                    (SELECT
                        IATA_Code_Marketing_Airline AS Airline,
                        Origin,
                        OriginCityName AS OriginCity,
                        Dest,
                        DestCityName AS DestCity,
                        FlightDate AS Date,
                        Flight_Number_Operating_Airline AS FlightNum,
                        CRSDepTime AS DepTime,
                        CRSArrTime AS ArrTime
                    FROM faa
                    WHERE
                        Origin <> {origin} AND
                        Dest = {destination} AND
                        FlightDate IN ({flight_date}, {next_date}) AND
                        Cast(ArrTime AS int) BETWEEN {arr_no_earlier} AND {arr_no_later}) AS finish
                WHERE
                    FIRST_LEG_DEST = SECOND_LEG_ORIG AND
                    FIRST_LEG_AIRLINE = SECOND_LEG_AIRLINE


        '''

    df = pd.read_sql_query(sql, conn)

    conn.close()

    return df


def getValidDestinations(db_name: str, ORIG: str, DATE: str):
    
    origin = "'" + ORIG + "'"
    flight_date = "'" + DATE + "'"
    next_date = "'" + getNextDate(DATE) + "'"

    conn = sqlite3.connect(db_name)

    sql = f'''SELECT
                    DISTINCT(finish.Dest) AS AIRPORT,
                    finish.DestCity AS CITY
                FROM 
                    (SELECT
                        Origin,
                        OriginCityName AS OriginCity,
                        Dest,
                        DestCityName AS DestCity,
                        FlightDate AS Date
                    FROM faa
                    WHERE
                        Origin = {origin} AND
                        FlightDate = {flight_date}) AS start,
                    (SELECT
                        Origin,
                        OriginCityName AS OriginCity,
                        Dest,
                        DestCityName AS DestCity,
                        FlightDate AS Date
                    FROM faa
                    WHERE
                        Origin <> {origin} AND
                        FlightDate IN ({flight_date}, {next_date})) AS finish
                WHERE
                    start.Dest = finish.Origin AND
                    finish.Dest <> start.Origin
                ORDER BY CITY
        '''

    df = pd.read_sql_query(sql, conn)

    conn.close()

    return df


def load_timezone_dictionary(location):
    df = pd.read_csv(location)
    dictionary = {k:v for k,v in zip(df['Origin'], df['Timezone'])}
    return dictionary


def dummyMLResult(num_rows: int):
    # this function is used for demonstration purposes only
    # it will be replaced our algorithm's "predict" function once available.

    return np.random.uniform(0,1,num_rows)


def dummyMissedConnectionTimeAdd(num_rows: int):
    # this function is used for demonstration purposes only
    # it will be replaced our algorithm's "predict" function once available.

    return np.random.uniform(0,1440,num_rows)


def itineraryBuilder(itinerary_df, tc, max_tc = 360, timezone_location = './airport_timezones_pytz.csv'):
    df = itinerary_df
    
    tz = load_timezone_dictionary(timezone_location)

    df['FIRST_LEG_ORIG_TZ'] = df['FIRST_LEG_ORIG'].map(tz)
    df['FIRST_LEG_DEST_TZ'] = df['FIRST_LEG_DEST'].map(tz)
    df['SECOND_LEG_ORIG_TZ'] = df['SECOND_LEG_ORIG'].map(tz)
    df['SECOND_LEG_DEST_TZ'] = df['SECOND_LEG_DEST'].map(tz)

    cols_to_generate = ['FIRST_LEG_DEP_TIMESTAMP', 'FIRST_LEG_ARR_TIMESTAMP', 'SECOND_LEG_DEP_TIMESTAMP', 'SECOND_LEG_ARR_TIMESTAMP']
    cols_to_read = [['FIRST_LEG_DATE', 'FIRST_LEG_DEP_TIME', 'FIRST_LEG_ORIG_TZ'],\
                    ['FIRST_LEG_DATE', 'FIRST_LEG_ARR_TIME', 'FIRST_LEG_DEST_TZ'],\
                    ['SECOND_LEG_DATE', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ORIG_TZ'],\
                    ['SECOND_LEG_DATE', 'SECOND_LEG_ARR_TIME', 'SECOND_LEG_DEST_TZ']]
    
    for i, c in enumerate(cols_to_generate):
        s = df[cols_to_read[i][0]] + 'T' + df[cols_to_read[i][1]].str.zfill(4)
        df['placeholder'] = pd.to_datetime(s).dt.tz_localize(pytz.utc)
        df[c] = df.apply(lambda x: x["placeholder"].tz_convert(x[cols_to_read[i][2]]), 1)
        df.drop(['placeholder'], axis=1, inplace=True)

    first_leg_zip = zip(df['FIRST_LEG_ARR_TIMESTAMP'], df['FIRST_LEG_DEP_TIMESTAMP'])
    second_leg_zip = zip(df['SECOND_LEG_ARR_TIMESTAMP'], df['SECOND_LEG_DEP_TIMESTAMP'])

    df['overnight_bool_1'] = pd.Series([arr < dept for (arr, dept) in first_leg_zip]).astype(int)
    df['overnight_bool_2'] = pd.Series([arr < dept for (arr, dept) in second_leg_zip]).astype(int)

    df['FIRST_LEG_ARR_TIMESTAMP'] = df.apply(lambda x: x['FIRST_LEG_ARR_TIMESTAMP'] + pd.DateOffset(days=x['overnight_bool_1']), 1)
    df['SECOND_LEG_ARR_TIMESTAMP'] = df.apply(lambda x: x['SECOND_LEG_ARR_TIMESTAMP'] + pd.DateOffset(days=x['overnight_bool_2']), 1)

    df['CONNECT_TIME'] = (df['SECOND_LEG_DEP_TIMESTAMP'] - df['FIRST_LEG_ARR_TIMESTAMP']).dt.total_seconds()/60
    df['TRIP_TIME'] = (df['SECOND_LEG_ARR_TIMESTAMP'] - df['FIRST_LEG_DEP_TIMESTAMP']).dt.total_seconds()/60

    df = df[df.CONNECT_TIME.between(tc, max_tc)]

    num_rows = df.shape[0]

    df['RISK_MISSED_CONNECTION'] = dummyMLResult(num_rows)
    df['NEXT_FLIGHT_TIMELOSS'] = dummyMissedConnectionTimeAdd(num_rows)

    df['TOTAL_RISK'] = df['RISK_MISSED_CONNECTION'].values * df['NEXT_FLIGHT_TIMELOSS'].values

    return df



def getNextDate(date: str):
    ## given a date in string format 
    ## there should be no leading zeroes on the date or month (e.g., 03/04/2023)
    month = int(date.split('/')[0])
    day = int(date.split('/')[1])
    year = int(date.split('/')[2])

    thirty_days = [4, 6, 9, 11]
    
    def iterateMonth(present_month: int):
        return f'{present_month+1}/1'
    
    def iterateDay(present_month: int, present_day: int):
        return f'{present_month}/{present_day+1}'

    if (month == 2 and day == 28 and year % 4 != 0) or \
        (month == 2 and day == 29 and year % 4 == 0) or \
        (month in thirty_days and day == 30) or \
        (month not in thirty_days and day == 31):

        return iterateMonth(month) + '/' + str(year)
    else:
        return iterateDay(month, day) + '/' + str(year)
