import sqlite3
import pandas as pd
import pytz
import numpy as np # used for dummy ML only


def itineraryBuilder(db_name: str,\
                    ORIG: str,\
                    DEST: str,\
                    DATE: str,\
                    tc: int,\
                    dep_no_earlier = '1',\
                    dep_no_later = '2359',\
                    arr_no_earlier = '1',\
                    arr_no_later = '2359',\
                    max_tc = 360,\
                    orderby = 'risk',\
                    timezone_location = './airport_timezones_pytz.csv'):
    
    ### This function returns all data required for the itinerary risk visualization.
    ### A multitude of columns are returned in a pandas dataframe, as listed below.
    ### The resulting dataframe may be ordered by 'risk' or by 'duration'.
    """ 'FIRST_LEG_AIRLINE', 'FIRST_LEG_ORIG', 'FIRST_LEG_ORIG_CITY',
       'FIRST_LEG_DEST', 'FIRST_LEG_DEST_CITY', 'FIRST_LEG_DATE',
       'FIRST_LEG_FLIGHT_NUM', 'FIRST_LEG_DEP_TIME', 'FIRST_LEG_ARR_TIME',
       'SECOND_LEG_AIRLINE', 'SECOND_LEG_ORIG', 'SECOND_LEG_ORIG_CITY',
       'SECOND_LEG_DEST', 'SECOND_LEG_DEST_CITY', 'SECOND_LEG_DATE',
       'SECOND_LEG_FLIGHT_NUM', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME',
       'NEXT_BEST_SECOND_LEG_DATE', 'NEXT_BEST_SECOND_LEG_DEP_TIME',
       'NEXT_BEST_SECOND_LEG_ARR_TIME', 'FIRST_LEG_ORIG_TZ',
       'FIRST_LEG_DEST_TZ', 'SECOND_LEG_ORIG_TZ', 'SECOND_LEG_DEST_TZ',
       'FIRST_LEG_DEP_TIMESTAMP', 'FIRST_LEG_ARR_TIMESTAMP',
       'SECOND_LEG_DEP_TIMESTAMP', 'SECOND_LEG_ARR_TIMESTAMP',
       'NEXT_BEST_SECOND_LEG_DEP_TIMESTAMP',
       'NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP', 'overnight_bool_1',
       'overnight_bool_2', 'overnight_bool_3', 'FIRST_FLIGHT_DURATION',
       'SECOND_FLIGHT_DURATION', 'CONNECT_TIME', 'TRIP_TIME',
       'RISK_MISSED_CONNECTION', 'NEXT_FLIGHT_TIMELOSS', 'TOTAL_RISK' """
      
    if orderby not in ['risk', 'duration']:
        raise ValueError('Please order by either "risk" or "duration".')
    
    df = queryFlights(db_name, ORIG, DEST, DATE, dep_no_earlier, dep_no_later, arr_no_earlier, arr_no_later)
    
    tz = load_timezone_dictionary(timezone_location)

    pd.set_option('mode.chained_assignment', None)

    df['FIRST_LEG_ORIG_TZ'] = df['FIRST_LEG_ORIG'].map(tz)
    df['FIRST_LEG_DEST_TZ'] = df['FIRST_LEG_DEST'].map(tz)
    df['SECOND_LEG_ORIG_TZ'] = df['SECOND_LEG_ORIG'].map(tz)
    df['SECOND_LEG_DEST_TZ'] = df['SECOND_LEG_DEST'].map(tz)

    cols_to_generate = ['FIRST_LEG_DEP_TIMESTAMP', 'FIRST_LEG_ARR_TIMESTAMP',\
                        'SECOND_LEG_DEP_TIMESTAMP', 'SECOND_LEG_ARR_TIMESTAMP',\
                        'NEXT_BEST_SECOND_LEG_DEP_TIMESTAMP', 'NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP']
    
    cols_to_read = [['FIRST_LEG_DATE', 'FIRST_LEG_DEP_TIME', 'FIRST_LEG_ORIG_TZ'],\
                    ['FIRST_LEG_DATE', 'FIRST_LEG_ARR_TIME', 'FIRST_LEG_DEST_TZ'],\
                    ['SECOND_LEG_DATE', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ORIG_TZ'],\
                    ['SECOND_LEG_DATE', 'SECOND_LEG_ARR_TIME', 'SECOND_LEG_DEST_TZ'],\
                    ['NEXT_BEST_SECOND_LEG_DATE', 'NEXT_BEST_SECOND_LEG_DEP_TIME', 'SECOND_LEG_ORIG_TZ'],\
                    ['NEXT_BEST_SECOND_LEG_DATE', 'NEXT_BEST_SECOND_LEG_ARR_TIME', 'SECOND_LEG_DEST_TZ']]
    
    for i, c in enumerate(cols_to_generate):
        st = df[cols_to_read[i][0]] + 'T' + df[cols_to_read[i][1]].str.zfill(4)
        timezip = zip(st, df[cols_to_read[i][2]])
        df[c] = [pd.to_datetime(s).tz_localize(t) for s,t in timezip]
        #df['placeholder'] = pd.to_datetime(s).dt.tz_localize(pytz.utc)
        #df[c] = df.apply(lambda x: x["placeholder"].tz_convert(x[cols_to_read[i][2]]), 1)
        #df.drop(['placeholder'], axis=1, inplace=True)

    first_leg_zip = zip(df['FIRST_LEG_ARR_TIMESTAMP'], df['FIRST_LEG_DEP_TIMESTAMP'])
    second_leg_zip = zip(df['SECOND_LEG_ARR_TIMESTAMP'], df['SECOND_LEG_DEP_TIMESTAMP'])
    next_best_zip = zip(df['NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'], df['NEXT_BEST_SECOND_LEG_DEP_TIMESTAMP'])

    df['overnight_bool_1'] = pd.Series([arr < dept for (arr, dept) in first_leg_zip]).astype(int)
    df['overnight_bool_2'] = pd.Series([arr < dept for (arr, dept) in second_leg_zip]).astype(int)
    df['overnight_bool_3'] = pd.Series([arr < dept for (arr, dept) in next_best_zip]).astype(int)

    df['FIRST_LEG_ARR_TIMESTAMP'] = df.apply(lambda x: x['FIRST_LEG_ARR_TIMESTAMP'] + pd.DateOffset(days=x['overnight_bool_1']), 1)
    df['SECOND_LEG_ARR_TIMESTAMP'] = df.apply(lambda x: x['SECOND_LEG_ARR_TIMESTAMP'] + pd.DateOffset(days=x['overnight_bool_2']), 1)
    df['NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'] = df.apply(lambda x: x['NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'] + pd.DateOffset(days=x['overnight_bool_3']), 1)

    df['FIRST_FLIGHT_DURATION'] = df.apply(lambda x: x['FIRST_LEG_ARR_TIMESTAMP'] - x['FIRST_LEG_DEP_TIMESTAMP'], axis=1).dt.total_seconds()/60
    df['SECOND_FLIGHT_DURATION'] = df.apply(lambda x: x['SECOND_LEG_ARR_TIMESTAMP'] - x['SECOND_LEG_DEP_TIMESTAMP'], axis=1).dt.total_seconds()/60
    
    df['CONNECT_TIME'] = (df.loc[:, 'SECOND_LEG_DEP_TIMESTAMP'] - df.loc[:, 'FIRST_LEG_ARR_TIMESTAMP']).dt.total_seconds()/60
    
    df['TRIP_TIME'] = (df.loc[:, 'SECOND_LEG_ARR_TIMESTAMP'] - df.loc[:, 'FIRST_LEG_DEP_TIMESTAMP']).dt.total_seconds()/60

    df = df.loc[df.loc[:, 'CONNECT_TIME'].between(tc, max_tc)]

    num_rows = df.shape[0]

    df['RISK_MISSED_CONNECTION'] = dummyMLResult(num_rows)
    df['NEXT_FLIGHT_TIMELOSS'] = (df.loc[:, 'NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'] - df.loc[:, 'SECOND_LEG_ARR_TIMESTAMP']).dt.total_seconds()/60

    df['TOTAL_RISK'] = df.loc[:, 'RISK_MISSED_CONNECTION'] * df.loc[:, 'NEXT_FLIGHT_TIMELOSS']

    if orderby == 'risk':
        df.sort_values(by=['TOTAL_RISK'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    elif orderby == 'duration':
        df.sort_values(by=['TRIP_TIME'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df



def queryFlights(db_name: str,\
                    ORIG: str,\
                    DEST: str,\
                    DATE: str,\
                    dep_no_earlier = '1',\
                    dep_no_later = '2359',\
                    arr_no_earlier = '1',\
                    arr_no_later = '2359'):
    
    ### This helper function is used by itinerary builder to interface with a database file via SQLITE.
    ### The database returned returns times as strings without timezones, and therefore needs to be subsequently transformed.
    ### Finding the next best itinerary is conducted in this function as well, through a dedicated helper function.
    ### The database returned contains the following columns:
    """'FIRST_LEG_AIRLINE', 'FIRST_LEG_ORIG', 'FIRST_LEG_ORIG_CITY',
       'FIRST_LEG_DEST', 'FIRST_LEG_DEST_CITY', 'FIRST_LEG_DATE',
       'FIRST_LEG_FLIGHT_NUM', 'FIRST_LEG_DEP_TIME', 'FIRST_LEG_ARR_TIME',
       'SECOND_LEG_AIRLINE', 'SECOND_LEG_ORIG', 'SECOND_LEG_ORIG_CITY',
       'SECOND_LEG_DEST', 'SECOND_LEG_DEST_CITY', 'SECOND_LEG_DATE',
       'SECOND_LEG_FLIGHT_NUM', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME',
       'NEXT_BEST_SECOND_LEG_DATE', 'NEXT_BEST_SECOND_LEG_DEP_TIME',
       'NEXT_BEST_SECOND_LEG_ARR_TIME"""

    
    origin = "'" + ORIG + "'"
    destination = "'" + DEST + "'"
    flight_date = "'" + DATE + "'"
    next_date = "'" + getNextDate(DATE) + "'"
    third_date = "'" + getNextDate(getNextDate(DATE)) + "'"

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
                        FlightDate IN ({flight_date}, {next_date}, {third_date}) AND
                        Cast(ArrTime AS int) BETWEEN {arr_no_earlier} AND {arr_no_later}) AS finish
                WHERE
                    FIRST_LEG_DEST = SECOND_LEG_ORIG AND
                    FIRST_LEG_AIRLINE = SECOND_LEG_AIRLINE
                ORDER BY SECOND_LEG_ORIG, SECOND_LEG_DATE, CAST(SECOND_LEG_DEP_TIME AS int)
        '''

    df = pd.read_sql_query(sql, conn)

    conn.close()

    df_partition = df[['SECOND_LEG_ORIG', 'SECOND_LEG_DATE', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME']]

    #return getNextBest(df_partition, returnType = 'new_only')
    next_best_flights = getNextBest(df_partition, returnType = 'new_only')

    df_next_best = pd.concat([df, next_best_flights], axis=1)
    df_next_best = df_next_best[df_next_best['NEXT_BEST_SECOND_LEG_DATE'].notnull()]

    return df_next_best.reset_index(drop=True)


def getValidDestinations(db_name: str, ORIG: str, DATE: str):
    ### returns all valid destinations from a designated initial airport.
    ### intended use if for visualization, so that users are not presented with invalid destinations.
    ### valid destinations are those that can be reached in exactly two legs.
    ### returns a pandas dataframe with the following columns: 'AIRPORT', 'CITY'.
    
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


def getNextBest(itinerary_partition, returnType = 'new_only'):
    # given a data frame of second leg itineraries, horizontally concatenates the next best second leg for each.
    # second legs which have no "next best" itinerary within the timeframe analyzed return "None" in each row.
    # required column names in original itinerary are:
    #### 'SECOND_LEG_ORIG', 'SECOND_LEG_DATE', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME'

    # suggest sending ONLY the columns listed into this function, rather than entire dataframe.

    # returnType = 'all_columns' keeps the original columns in place, mostly for troubleshooting and review.
    # returnType = 'new_only' (default) returns a dataframe of only the generated "next best" columns.

    pd.set_option('mode.chained_assignment', None)

    if returnType not in ['all_columns', 'new_only']:
        raise ValueError('Please select either "new_only" or "all_columns" for returnType.')

    df_partition = itinerary_partition

    next_best = []

    origs = df_partition['SECOND_LEG_ORIG'].tolist()
    dates = df_partition['SECOND_LEG_DATE'].tolist()
    deps = df_partition['SECOND_LEG_DEP_TIME'].tolist()

    present_index = 0
    max_index = len(origs) - 1
    while present_index <= max_index:
        if (present_index == max_index) or (origs[present_index] != origs[present_index + 1]):
            next_best.append(max_index + 1) #return the dummy row, since this row will be dropped.
        else:
            look_ahead = 1
            while (present_index + look_ahead) <= max_index:
                
                if (dates[present_index] != dates[present_index + look_ahead] or \
                    deps[present_index] != deps[present_index + look_ahead]):
                    
                    next_best.append(present_index + look_ahead)
                    break

                elif (present_index + look_ahead) == max_index:
                    next_best.append(max_index + 1) #return the dummy row; there are no more rows to examine
                
                look_ahead = look_ahead + 1
            
        present_index = present_index + 1

    colsForNull = ['SECOND_LEG_ORIG', 'SECOND_LEG_DATE', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME']
    dummy_row_location = len(df_partition.index)
    df_partition.loc[dummy_row_location, colsForNull] = [None, None, None, None]
    next_best.append(dummy_row_location)

    df_partition['NEXT_BEST_SECOND_LEG_DATE'] = [df_partition['SECOND_LEG_DATE'].tolist()[i] for i in next_best]
    df_partition['NEXT_BEST_SECOND_LEG_DEP_TIME'] = [df_partition['SECOND_LEG_DEP_TIME'].tolist()[i] for i in next_best]
    df_partition['NEXT_BEST_SECOND_LEG_ARR_TIME'] = [df_partition['SECOND_LEG_ARR_TIME'].tolist()[i] for i in next_best]

    df_partition = df_partition.iloc[:df_partition.shape[0]-1, :] # remove the dummy row.

    if returnType == 'new_only':
        df_partition = df_partition[['NEXT_BEST_SECOND_LEG_DATE','NEXT_BEST_SECOND_LEG_DEP_TIME','NEXT_BEST_SECOND_LEG_ARR_TIME']]

    return df_partition



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


def load_timezone_dictionary(location):
    ### loads a dictionary which maps each location to a pytz timezone.
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

