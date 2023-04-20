import sqlite3
import pandas as pd
import pytz


def itineraryBuilder(db_name: str,\
                    ORIG: str,\
                    DEST: str,\
                    DATE: str,\
                    arr_no_earlier_date: str,\
                    arr_no_later_date: str,\
                    tc: int,\
                    dep_no_earlier = '1',\
                    dep_no_later = '2359',\
                    arr_no_earlier = '1',\
                    arr_no_later = '2359',\
                    max_tc = 360,\
                    orderby = 'duration',\
                    timezone_location = './airport_timezones_pytz.csv'):
    
    
    ### This function returns all data required for the itinerary risk visualization.
    ### A multitude of columns are returned in a pandas dataframe, as listed below.
    ### The resulting dataframe may be ordered by 'risk' or by 'duration'.
    """ 'FIRST_LEG_AIRLINE', 'FIRST_LEG_ORIG', 'FIRST_LEG_ORIG_CITY',
       'FIRST_LEG_DEST', 'FIRST_LEG_DEST_CITY', 'FIRST_LEG_DATE',
       'FIRST_LEG_DEP_TIME', 'FIRST_LEG_ARR_TIME',
       'SECOND_LEG_AIRLINE', 'SECOND_LEG_ORIG', 'SECOND_LEG_ORIG_CITY',
       'SECOND_LEG_DEST', 'SECOND_LEG_DEST_CITY', 'SECOND_LEG_DATE',
       'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME',
       'NEXT_BEST_SECOND_LEG_DATE', 'NEXT_BEST_SECOND_LEG_DEP_TIME',
       'NEXT_BEST_SECOND_LEG_ARR_TIME', 'FIRST_LEG_ORIG_TZ',
       'FIRST_LEG_DEST_TZ', 'SECOND_LEG_ORIG_TZ', 'SECOND_LEG_DEST_TZ',
       'FIRST_LEG_DEP_TIMESTAMP', 'FIRST_LEG_ARR_TIMESTAMP',
       'SECOND_LEG_DEP_TIMESTAMP', 'SECOND_LEG_ARR_TIMESTAMP',
       'FIRST_LEG_PRED15', 'FIRST_LEG_PRED30', 'FIRST_LEG_PRED45',
       'FIRST_LEG_PRED60', 'FIRST_LEG_PRED75',
        'FIRST_LEG_PRED90', 'FIRST_LEG_PRED105', 'FIRST_LEG_PRED120',
       'NEXT_BEST_SECOND_LEG_DEP_TIMESTAMP',
       'NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP', 'overnight_bool_1',
       'overnight_bool_2', 'overnight_bool_3', 'FIRST_FLIGHT_DURATION',
       'SECOND_FLIGHT_DURATION', 'CONNECT_TIME', 'TRIP_TIME',
       'RISK_MISSED_CONNECTION', 'NEXT_FLIGHT_TIMELOSS', 'TOTAL_RISK' """

    if orderby not in ['risk', 'duration', 'earliest_arrival', 'min_connection_time']:
        raise ValueError('Please order by either "risk", "duration", "earliest_arrival", or "min_connect_time".')
    
    df = queryFlights(db_name, ORIG, DEST, DATE, arr_no_earlier_date, arr_no_later_date, dep_no_earlier, dep_no_later, arr_no_earlier, arr_no_later)

    if df.shape[0] > 0:
        # only run the transformations below if flights are returned

        # convert the timezones
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

            df['temp'] = pd.to_datetime(st)

            #if c in ['FIRST_LEG_DEP_TIMESTAMP', 'SECOND_LEG_ARR_TIMESTAMP', 'NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP']:
            if len(df[cols_to_read[i][2]].unique()) <= 1:
                # in this case the tz is always the same, and the below is most efficient
                # grabs the first row's entry for timezone to localize on
                df[c] = df['temp'].dt.tz_localize(df[cols_to_read[i][2]][0])
            else:
                # in this case we use groupby to break up into segment with the same tz
                # then we can apply the same idea above
                df[c] = pd.DataFrame(df.groupby(cols_to_read[i][2]).apply(lambda x: x['temp'].dt.tz_localize(x.name))).reset_index().set_index('level_1')['temp'].sort_index()
            
        df.drop(columns=['temp'], inplace=True)

        first_leg_zip = zip(df['FIRST_LEG_ARR_TIMESTAMP'], df['FIRST_LEG_DEP_TIMESTAMP'])
        second_leg_zip = zip(df['SECOND_LEG_ARR_TIMESTAMP'], df['SECOND_LEG_DEP_TIMESTAMP'])
        next_best_zip = zip(df['NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'], df['NEXT_BEST_SECOND_LEG_DEP_TIMESTAMP'])

        df['overnight_bool_1'] = pd.Series([arr < dept for (arr, dept) in first_leg_zip]).astype(int)
        df['overnight_bool_2'] = pd.Series([arr < dept for (arr, dept) in second_leg_zip]).astype(int)
        df['overnight_bool_3'] = pd.Series([arr < dept for (arr, dept) in next_best_zip]).astype(int)

        df['FIRST_LEG_ARR_TIMESTAMP'] = df['FIRST_LEG_ARR_TIMESTAMP'] + df['overnight_bool_1'].astype('timedelta64[D]')
        df['SECOND_LEG_ARR_TIMESTAMP'] = df['SECOND_LEG_ARR_TIMESTAMP'] + df['overnight_bool_2'].astype('timedelta64[D]')
        df['NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'] = df['NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'] + df['overnight_bool_3'].astype('timedelta64[D]')

        df['FIRST_FLIGHT_DURATION'] = df.apply(lambda x: x['FIRST_LEG_ARR_TIMESTAMP'] - x['FIRST_LEG_DEP_TIMESTAMP'], axis=1).dt.total_seconds()/60
        df['SECOND_FLIGHT_DURATION'] = df.apply(lambda x: x['SECOND_LEG_ARR_TIMESTAMP'] - x['SECOND_LEG_DEP_TIMESTAMP'], axis=1).dt.total_seconds()/60
        
        df['CONNECT_TIME'] = df.apply(lambda x: x['SECOND_LEG_DEP_TIMESTAMP'] - x['FIRST_LEG_ARR_TIMESTAMP'], axis=1).dt.total_seconds()/60
        
        #df['CONNECT_TIME'] = (df.loc[:, 'SECOND_LEG_DEP_TIMESTAMP'] - df.loc[:, 'FIRST_LEG_ARR_TIMESTAMP']).dt.total_seconds()/60
        
        df['TRIP_TIME'] = df.apply(lambda x: x['SECOND_LEG_ARR_TIMESTAMP'] - x['FIRST_LEG_DEP_TIMESTAMP'], axis=1).dt.total_seconds()/60
        
        #df['TRIP_TIME'] = (df.loc[:, 'SECOND_LEG_ARR_TIMESTAMP'] - df.loc[:, 'FIRST_LEG_DEP_TIMESTAMP']).dt.total_seconds()/60

        df = df.loc[df.loc[:, 'CONNECT_TIME'].between(tc, max_tc)]

        ## determine appropriate risk factor for each flight
        ## risk is established as a step function

        df.reset_index(drop=True, inplace=True)
        risk_cols_to_ref = [getRiskColumnName(c) for c in df['CONNECT_TIME']]
        risks_to_apply = [df.loc[index, col] for index,col in enumerate(risk_cols_to_ref)]

        df['RISK_MISSED_CONNECTION'] = risks_to_apply

        df['NEXT_FLIGHT_TIMELOSS'] = df.apply(lambda x: x['NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'] - x['SECOND_LEG_ARR_TIMESTAMP'], axis=1).dt.total_seconds()/60

        #df['NEXT_FLIGHT_TIMELOSS'] = (df.loc[:, 'NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP'] - df.loc[:, 'SECOND_LEG_ARR_TIMESTAMP']).dt.total_seconds()/60

        df['TOTAL_RISK'] = df.loc[:, 'RISK_MISSED_CONNECTION'] * df.loc[:, 'NEXT_FLIGHT_TIMELOSS']

        if orderby == 'risk':
            df.sort_values(by=['TOTAL_RISK'], inplace=True)
        
        elif orderby == 'duration':
            df.sort_values(by=['TRIP_TIME'], inplace=True)

        elif orderby == 'earliest_arrival':
            df.sort_values(by=['SECOND_LEG_ARR_TIMESTAMP'], inplace=True)

        elif orderby == 'min_connection_time':
            df.sort_values(by=['CONNECT_TIME'], inplace=True)
        
        df.reset_index(drop=True, inplace=True)

    else:
        # if there are no flights available in date/time range requested, return empty df with column names
        cols = ['FIRST_LEG_AIRLINE', 'FIRST_LEG_ORIG', 'FIRST_LEG_ORIG_CITY',\
                'FIRST_LEG_DEST', 'FIRST_LEG_DEST_CITY', 'FIRST_LEG_DATE',\
                'FIRST_LEG_DEP_TIME', 'FIRST_LEG_ARR_TIME',\
                'SECOND_LEG_AIRLINE', 'SECOND_LEG_ORIG', 'SECOND_LEG_ORIG_CITY',\
                'SECOND_LEG_DEST', 'SECOND_LEG_DEST_CITY', 'SECOND_LEG_DATE',\
                'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME',\
                'NEXT_BEST_SECOND_LEG_DATE', 'NEXT_BEST_SECOND_LEG_DEP_TIME',\
                'NEXT_BEST_SECOND_LEG_ARR_TIME', 'FIRST_LEG_ORIG_TZ',\
                'FIRST_LEG_DEST_TZ', 'SECOND_LEG_ORIG_TZ', 'SECOND_LEG_DEST_TZ',\
                'FIRST_LEG_DEP_TIMESTAMP', 'FIRST_LEG_ARR_TIMESTAMP',\
                'SECOND_LEG_DEP_TIMESTAMP', 'SECOND_LEG_ARR_TIMESTAMP',\
                'FIRST_LEG_PRED15', 'FIRST_LEG_PRED30', 'FIRST_LEG_PRED45',\
                'FIRST_LEG_PRED60', 'FIRST_LEG_PRED75',\
                'FIRST_LEG_PRED90', 'FIRST_LEG_PRED105', 'FIRST_LEG_PRED120'\
                'NEXT_BEST_SECOND_LEG_DEP_TIMESTAMP',\
                'NEXT_BEST_SECOND_LEG_ARR_TIMESTAMP', 'overnight_bool_1',\
                'overnight_bool_2', 'overnight_bool_3', 'FIRST_FLIGHT_DURATION',\
                'SECOND_FLIGHT_DURATION', 'CONNECT_TIME', 'TRIP_TIME',\
                'RISK_MISSED_CONNECTION', 'NEXT_FLIGHT_TIMELOSS', 'TOTAL_RISK']
        
        df = pd.DataFrame(columns=cols)

    return df


def queryFlights(db_name: str,\
                    ORIG: str,\
                    DEST: str,\
                    DATE: str,\
                    arr_no_earlier_date: str,\
                    arr_no_later_date: str,\
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
       'FIRST_LEG_DEP_TIME', 'FIRST_LEG_ARR_TIME',
       'SECOND_LEG_AIRLINE', 'SECOND_LEG_ORIG', 'SECOND_LEG_ORIG_CITY',
       'SECOND_LEG_DEST', 'SECOND_LEG_DEST_CITY', 'SECOND_LEG_DATE',
       'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME',
       'FIRST_LEG_PRED15', 'FIRST_LEG_PRED30', 'FIRST_LEG_PRED45',
       'FIRST_LEG_PRED60', 'FIRST_LEG_PRED75',
        'FIRST_LEG_PRED90', 'FIRST_LEG_PRED105', 'FIRST_LEG_PRED120'
       'NEXT_BEST_SECOND_LEG_DATE', 'NEXT_BEST_SECOND_LEG_DEP_TIME',
       'NEXT_BEST_SECOND_LEG_ARR_TIME"""

    
    origin = "'" + ORIG + "'"
    destination = "'" + DEST + "'"
    flight_date_dep = "'" + DATE + "'"
    next_date = "'" + getNextDate(DATE) + "'"
    third_date = "'" + getNextDate(getNextDate(DATE)) + "'"
   
    if DATE == arr_no_earlier_date and DATE != arr_no_later_date:
        # no need to restrict the date range
        # we will need to assess the arrival time range after initial query
        # we will return all flights in the subsequent 3 day period
        arrival_dates = flight_date_dep + ", " + next_date + ", " + third_date
    
    elif DATE == arr_no_earlier_date and DATE == arr_no_later_date:
        # restrict the date range per user request
        # user is requesting to depart and arrive on day of original flight
        # need to still include following day of flights for getNextBest()
        # we will need to assess the arrival time range after initial query
        arrival_dates = flight_date_dep + ", " + next_date

    elif DATE != arr_no_earlier_date and DATE != arr_no_later_date:
        # user is requesting to only arrive on the following day
        # exclude all flights arriving on day of departure
        # still leave the third day of flight arrivals in for getNextBest() 
        arrival_dates = next_date + ", " + third_date
    
    else:
        # should be possible with present slider set up in index.html
        # trigger an error if any other condition occurs.
        raise Exception("Impossible condition in queryFlights() initial query conditions.")


    conn = sqlite3.connect(db_name)

    sql = f'''SELECT
                    start.Airline AS FIRST_LEG_AIRLINE,
                    start.Origin AS FIRST_LEG_ORIG,
                    start.OriginCity AS FIRST_LEG_ORIG_CITY,
                    start.Dest AS FIRST_LEG_DEST,
                    start.DestCity AS FIRST_LEG_DEST_CITY,
                    start.Date AS FIRST_LEG_DATE,
                    start.DepartureTime AS FIRST_LEG_DEP_TIME,
                    start.ArrivalTime AS FIRST_LEG_ARR_TIME,
                    start.Prediction_15min AS FIRST_LEG_PRED15,
                    start.Prediction_30min AS FIRST_LEG_PRED30,
                    start.Prediction_45min AS FIRST_LEG_PRED45,
                    start.Prediction_60min AS FIRST_LEG_PRED60,
                    start.Prediction_75min AS FIRST_LEG_PRED75,
                    start.Prediction_90min AS FIRST_LEG_PRED90,
                    start.Prediction_105min AS FIRST_LEG_PRED105,
                    start.Prediction_120min AS FIRST_LEG_PRED120,
                    finish.Airline AS SECOND_LEG_AIRLINE,                    
                    finish.Origin AS SECOND_LEG_ORIG,
                    finish.OriginCity AS SECOND_LEG_ORIG_CITY,
                    finish.Dest AS SECOND_LEG_DEST,
                    finish.DestCity AS SECOND_LEG_DEST_CITY,
                    finish.Date AS SECOND_LEG_DATE,
                    finish.DepartureTime AS SECOND_LEG_DEP_TIME,
                    finish.ArrivalTime AS SECOND_LEG_ARR_TIME
                FROM 
                    (SELECT
                        Marketing_Airline_Network AS Airline,
                        Origin,
                        OriginCityName AS OriginCity,
                        Dest,
                        DestCityName AS DestCity,
                        FlightDate AS Date,
                        CAST(CRSDepTime AS INT) AS DepartureTime,
                        CAST(CRSArrTime AS INT) AS ArrivalTime,
                        Prediction_15min,
                        Prediction_30min,
                        Prediction_45min,
                        Prediction_60min,
                        Prediction_75min,
                        Prediction_90min,
                        Prediction_105min,
                        Prediction_120min
                    FROM {db_name}
                    WHERE
                        Origin = {origin} AND
                        Dest <> {destination} AND
                        FlightDate = {flight_date_dep} AND
                        DepartureTime >= CAST({dep_no_earlier} AS INT) AND
                        DepartureTime <= CAST({dep_no_later} AS INT)) AS start,
                    (SELECT
                        Marketing_Airline_Network AS Airline,
                        Origin,
                        OriginCityName AS OriginCity,
                        Dest,
                        DestCityName AS DestCity,
                        FlightDate AS Date,
                        CAST(CRSDepTime AS INT) AS DepartureTime,
                        CAST(CRSArrTime AS INT) AS ArrivalTime
                    FROM {db_name}
                    WHERE
                        Origin <> {origin} AND
                        Dest = {destination} AND
                        FlightDate IN ({arrival_dates})) AS finish
                WHERE
                    FIRST_LEG_DEST = SECOND_LEG_ORIG AND
                    FIRST_LEG_AIRLINE = SECOND_LEG_AIRLINE
                ORDER BY SECOND_LEG_ORIG, SECOND_LEG_DATE, CAST(SECOND_LEG_DEP_TIME AS int)
        '''
    df = pd.read_sql_query(sql, conn)

    conn.close()

    # now we go and get the next best flights
    df_partition = df[['SECOND_LEG_ORIG', 'SECOND_LEG_DATE', 'SECOND_LEG_DEP_TIME', 'SECOND_LEG_ARR_TIME']]

    #return getNextBest(df_partition, returnType = 'new_only')
    next_best_flights = getNextBest(df_partition, returnType = 'new_only')

    df_next_best = pd.concat([df, next_best_flights], axis=1)
    df_next_best = df_next_best[df_next_best['NEXT_BEST_SECOND_LEG_DATE'].notnull()]

    # now we need to apply the time filter
    if DATE == arr_no_earlier_date and DATE != arr_no_later_date:
        # user is ok arriving on either day
        # assess arr_no_earlier time for flights on first day
        # assess arr_no_later time for flights on the second day
        df_next_best = df_next_best[~(((df_next_best['SECOND_LEG_ARR_TIME']<int(arr_no_earlier)) & (df_next_best['SECOND_LEG_DATE'] == DATE)) | \
                                    ((df_next_best['SECOND_LEG_ARR_TIME']>int(arr_no_later)) & (df_next_best['SECOND_LEG_DATE'] == next_date.strip("'"))))]
        

        #df_next_best = df_next_best[((df_next_best['SECOND_LEG_ARR_TIME']>int(arr_no_earlier)) | (df_next_best['SECOND_LEG_DATE'] != DATE)) | \
        #                            ((df_next_best['SECOND_LEG_ARR_TIME']<int(arr_no_later)) | (df_next_best['SECOND_LEG_DATE'] == DATE))]
    
    elif DATE == arr_no_earlier_date and DATE == arr_no_later_date:
        # user is requesting to depart and arrive on day of original flight
        # assess both arr_no_earlier time and arr_no_later time on the first day
        # exclude all flights from other days
        df_next_best = df_next_best[(df_next_best['SECOND_LEG_ARR_TIME'].between(int(arr_no_earlier), int(arr_no_later))) & (df_next_best['SECOND_LEG_DATE'] == DATE)]

    elif DATE != arr_no_earlier_date and DATE != arr_no_later_date:
        # user is requesting to only arrive on the following day
        # assess both arr_no_earlier time and arr_no_later time on the second day
        # exclude all flights arriving on day of departure
        df_next_best = df_next_best[(df_next_best['SECOND_LEG_ARR_TIME'].between(int(arr_no_earlier), int(arr_no_later))) & (df_next_best['SECOND_LEG_DATE'] == next_date.strip("'"))]
    
    else:
        # should be possible with present slider set up in index.html
        # trigger an error if any other condition occurs.
        raise Exception("Impossible condition in queryFlights() final filtering.")
    
    df_next_best = df_next_best.astype({'NEXT_BEST_SECOND_LEG_DEP_TIME': 'int',\
                                        'NEXT_BEST_SECOND_LEG_ARR_TIME': 'int'})
    df_next_best = df_next_best.astype({'FIRST_LEG_DEP_TIME': 'str',\
                                        'FIRST_LEG_ARR_TIME': 'str',\
                                        'SECOND_LEG_DEP_TIME': 'str',\
                                        'SECOND_LEG_ARR_TIME': 'str',\
                                        'NEXT_BEST_SECOND_LEG_DEP_TIME': 'str',\
                                        'NEXT_BEST_SECOND_LEG_ARR_TIME': 'str',\
                                        'FIRST_LEG_PRED15': 'float',\
                                        'FIRST_LEG_PRED30': 'float',\
                                        'FIRST_LEG_PRED45': 'float',\
                                        'FIRST_LEG_PRED60': 'float',\
                                        'FIRST_LEG_PRED75': 'float',\
                                        'FIRST_LEG_PRED90': 'float',\
                                        'FIRST_LEG_PRED105': 'float',\
                                        'FIRST_LEG_PRED120': 'float'})
    
    df_next_best = df_next_best[df_next_best['SECOND_LEG_DATE'] != third_date.strip("'")]

    df_next_best.reset_index(drop=True, inplace=True)

    return df_next_best


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
                    FROM {db_name}
                    WHERE
                        Origin = {origin} AND
                        FlightDate = {flight_date}) AS start,
                    (SELECT
                        Origin,
                        OriginCityName AS OriginCity,
                        Dest,
                        DestCityName AS DestCity,
                        FlightDate AS Date
                    FROM {db_name}
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

    dummy_row_location = len(df_partition.index)
    df_partition.loc[dummy_row_location, :] = [None, None, None, None]

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


def getRiskColumnName(tc):
    ## assume a 30 minute buffer
    ## i.e., if a person has a 45 minute connection time, they will miss their connection if first leg is >15 mins late.
    ## best attempt at accounting for plane doors closing before flight departs
    ## this will differ by airport; adding specificity to this assumption is a subject for further research

    risk_col_names = ['FIRST_LEG_PRED15', 'FIRST_LEG_PRED30', 'FIRST_LEG_PRED45', 'FIRST_LEG_PRED60', 'FIRST_LEG_PRED75',\
                        'FIRST_LEG_PRED90', 'FIRST_LEG_PRED105', 'FIRST_LEG_PRED120']
    if tc >= 150:
        return risk_col_names[7]
    elif tc >= 135:
        return risk_col_names[6]
    elif tc >= 120:
        return risk_col_names[5]
    elif tc >= 105:
        return risk_col_names[4]
    elif tc >= 90:
        return risk_col_names[3]
    elif tc >= 75:
        return risk_col_names[2]
    elif tc >= 60:
        return risk_col_names[1]
    else:
        return risk_col_names[0]
