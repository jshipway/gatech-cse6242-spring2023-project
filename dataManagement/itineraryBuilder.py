def itineraryBuilder(conn, ORIG, DEST, DATE, tc):
    origin = "'" + ORIG + "'"
    destination = "'" + DEST + "'"
    flight_date = "'" + DATE + "'"

    sql = f'''SELECT
                start.Origin AS FIRST_LEG_ORIG,
                start.Dest AS FIRST_LEG_DEST,
                start.Date AS FIRST_LEG_DATE,
                start.FlightNum AS FIRST_LEG_FLIGHT_NUM,
                start.DepTime AS FIRST_LEG_DEP_TIME,
                start.ArrTime AS FIRST_LEG_ARR_TIME,
                finish.Origin AS SECOND_LEG_ORIGIN,
                finish.Dest AS SECOND_LEG_DEST,
                finish.Date AS SECOND_LEG_DATE,
                finish.FlightNum AS SECOND_LEG_FLIGHT_NUM,
                finish.DepTime AS SECOND_LEG_DEP_TIME,
                finish.ArrTime AS SECOND_LEG_ARR_TIME,
                CAST((finish.DepTime/100) AS int)*60 + finish.DepTime%100 -
                    CAST((start.ArrTime/100) AS int)*60 - start.ArrTime%100 AS CONNECT_TIME
            FROM 
                (SELECT
                    Origin,
                    Dest,
                    FlightDate AS Date,
                    Flight_Number_Operating_Airline AS FlightNum,
                    CAST(CRSDepTime AS int) AS DepTime,
                    CAST(CRSArrTime AS int) AS ArrTime
                FROM faa
                WHERE
                    Origin = {origin} AND
                    Dest <> {destination} AND
                    FlightDate = {flight_date}) AS start,
                (SELECT
                    Origin,
                    Dest,
                    FlightDate AS Date,
                    Flight_Number_Operating_Airline AS FlightNum,
                    CAST(CRSDepTime AS int) AS DepTime,
                    CAST(CRSArrTime AS int) AS ArrTime
                FROM faa
                WHERE
                    Origin <> {origin} AND
                    Dest = {destination} AND
                    FlightDate = {flight_date}) AS finish
            WHERE
                FIRST_LEG_DEST = SECOND_LEG_ORIGIN AND
                CONNECT_TIME > {tc}
        '''

    cursor = conn.execute(sql)
    return cursor.fetchall()