import csv
import sqlite3

def generateDatabase(read_filename: str, database_name: str):
    ## This function generates a new database from scratch.
    ## read_filename: the csv source file location to serve as basis for new db.
    ## database_name: the name of database, which will be '[database_name].db'.
    
    imported_csv = csv.reader(open(read_filename, encoding='utf8'))
    conn = sqlite3.connect(database_name)
    conn.execute(f'DROP TABLE IF EXISTS {database_name}')    

    for i, row in enumerate(imported_csv):
        if i == 0:
            colnames = ', '.join(row)
            sql = f'CREATE TABLE {database_name} ({colnames});'
            print(sql)
            conn.execute(sql)
            print("table should be created")
        else:
            sql_base = f'''INSERT INTO {database_name} VALUES ({('?,'*len(row))[:-1]});'''
            sql_data = tuple(row)
            conn.execute(sql_base, sql_data)
    conn.commit()
    conn.close()

    return None


def appendDatabase(read_filename: str, database_name: str):
    ## This function appends a new csv to an existing database.
    ## The two files must have identical schema.
    ## read_filename: the csv source file location to serve as basis for new db.
    ## database_name: the name of database, which will be '[database_name].db'

    return None
