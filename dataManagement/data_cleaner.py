# Imports
import pandas as pd
from pathlib import Path
import zipfile
from utils import coordinate_utils as cu


_ZIP_DATA_DIR_ = r'C:\Users\Paul\OneDrive - Georgia Institute of Technology\Grad_School\Courses\CSE6242\Projects\dataset\Marketing On-time Data 2018-19'
_CSV_NAME_BASE = 'On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_'
_COLS_OF_INTEREST = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',
                     'Marketing_Airline_Network', 'Operated_or_Branded_Code_Share_Partners',
                     'Operating_Airline ', 'Tail_Number', 'OriginAirportID', 'OriginCityMarketID',
                     'Origin', 'OriginCityName', 'OriginState', 'DestAirportID', 'DestCityMarketID',
                     'Dest', 'DestCityName', 'DestState', 'CRSDepTime', 'DepTime', 'DepTimeBlk',
                     'CRSArrTime', 'ArrTime', 'ArrTimeBlk', 'Cancelled', 'CancellationCode',
                     'Distance', 'DistanceGroup', 'CarrierDelay', 'NASDelay', 'LateAircraftDelay']


def extract_csv_from_zip(zip_file_path):
    if Path(zip_file_path).exists():
        data_month = str(zip_file_path).split('_Performance_')[1].replace('.zip', '')
        extract_dir = Path(Path(zip_file_path).parent, 'temp_extracted')
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True)

        with zipfile.ZipFile(str(zip_file_path), 'r') as file:
            file.extractall(str(extract_dir))

        csv_path = Path(extract_dir, _CSV_NAME_BASE + data_month + '.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=_COLS_OF_INTEREST)
        else:
            df = pd.DataFrame(columns=_COLS_OF_INTEREST)

        return df

    else:
        raise FileNotFoundError('Could not find the defined zip file: {}'.format(zip_file_path))


def clean_dataframe(df, prime_airport=None):
    if (prime_airport is not None) and (not df[df['Origin'] == prime_airport].empty):
        primary_df = df[df['Origin'] == prime_airport]  # Get rows with origin of interest
        all_valid_dest = list(primary_df['Dest'].unique())  # Get all unique destinations
        secondary_df = df[df['Origin'].isin(all_valid_dest)]  # Set those dests as origins too

        final_df = pd.concat([primary_df, secondary_df]).sort_values(by='DayofMonth')
    else:
        if prime_airport is not None:
            print('Could not find the defined airport in the dataset: {}'.format(prime_airport))
        final_df = df.copy()
    all_unq_airports = list(set(list(df['Origin'].unique()) + list(df['Dest'].unique())))
    busy_count = {'Airport': ['Dep', 'Arr']}
    for apt in all_unq_airports:
        if apt not in busy_count:
            busy_count[apt] = [0, 0]
        busy_count[apt][0] = (df.Origin.values == apt).sum()
        busy_count[apt][1] = (df.Dest.values == apt).sum()

    airport_size_df = pd.DataFrame(busy_count).reset_index(drop=True)
    airport_size_df.to_csv('supplemental_data/airport_size.csv')
    fixed_df = finalize_dataset(final_df, airport_size_df)

    return fixed_df


def finalize_dataset(df, airport_size_df):
    # This method is intended to finalize all the features needed for the ML process, as well as
    # any other columns needed for visualization
    airport_locs = pd.ExcelFile('supplemental_data/Airport_locations_v2.xlsx')\
        .parse(sheet_name='Airports in Aug2022 On tim Data')
    hubs = pd.read_excel('supplemental_data/Hubs.xlsx', skiprows=0, header=1)
    sub_df = df.copy()
    sub_df = sub_df.merge(airport_locs, left_on='Origin', right_on='Airport')
    sub_df = sub_df.rename(mapper={'FAA_Lat': 'Origin_Lat', 'FAA_Long': 'Origin_Lon', 'ST_zone_adj': 'Origin_TZ_Adjust'}, axis=1, errors='raise')
    sub_df.drop(['Airport', 'Airport_city', 'Airport_State', 'Time_Zone', 'TZ_adjust', 'TimeZone_nm'],
                inplace=True, axis=1)
    sub_df = sub_df.merge(airport_locs, left_on='Dest', right_on='Airport')
    sub_df = sub_df.rename(mapper={'FAA_Lat': 'Dest_Lat', 'FAA_Long': 'Dest_Lon', 'ST_zone_adj': 'Dest_TZ_Adjust'}, axis=1, errors='raise')
    sub_df.drop(['Airport', 'Airport_city', 'Airport_State', 'Time_Zone', 'TZ_adjust', 'TimeZone_nm'],
                inplace=True, axis=1)
    unnamed_cols = [i for i in sub_df.columns if 'unnamed' in i.lower()]
    sub_df.drop(unnamed_cols, inplace=True, axis=1)

    dist = []
    dirctn = []
    ap_size = {'orgn': [], 'dest': []}
    hub_stat = {'orgn': [], 'dest': []}
    for ndx, row in sub_df.iterrows():
        # Get hub status
        orgn = getattr(row, 'Origin')
        dest = getattr(row, 'Dest')
        carrier = getattr(row, 'Marketing_Airline_Network')
        if (carrier in hubs.columns) and (orgn in list(hubs['Airport'].unique())):
            hub_stat['orgn'].append(hubs[hubs['Airport'] == orgn][carrier].iloc[0])
        else:
            hub_stat['orgn'].append(0)  # If not present, is not a hub

        if (carrier in hubs.columns) and (dest in list(hubs['Airport'].unique())):
            hub_stat['dest'].append(hubs[hubs['Airport'] == dest][carrier].iloc[0])
        else:
            hub_stat['dest'].append(0)  # If not present, is not a hub

        if orgn in airport_size_df.columns:
            ap_size['orgn'].append(airport_size_df[orgn].iloc[0] + airport_size_df[orgn].iloc[1])
        else:
            ap_size['orgn'].append(None)

        if dest in airport_size_df.columns:
            ap_size['dest'].append(airport_size_df[dest].iloc[0] + airport_size_df[dest].iloc[1])
        else:
            ap_size['dest'].append(None)

        # Get distance and direction
        latlon0 = [getattr(row, 'Origin_Lat'), getattr(row, 'Origin_Lon')]
        latlon1 = [getattr(row, 'Dest_Lat'), getattr(row, 'Dest_Lon')]
        a, d = cu.get_direction_range(latlon0, latlon1)
        dist.append(d)
        dirctn.append(a)
    sub_df['StraightLineDistance_Miles'] = dist
    sub_df['Direction_Deg'] = dirctn
    sub_df['OriginHubStatus'] = hub_stat['orgn']
    sub_df['DestHubStatus'] = hub_stat['dest']
    sub_df['OriginTotalFlights'] = ap_size['orgn']
    sub_df['DestTotalFlights'] = ap_size['dest']

    return sub_df


def extract_all_files(zip_dir, base_dir, prime_airport=None):
    if Path(zip_dir).is_dir():
        for fle in Path(zip_dir).iterdir():
            if fle.suffix == '.zip':
                df = extract_csv_from_zip(fle)
                cleaned_df = clean_dataframe(df, prime_airport=prime_airport)
                output_cleaned_data_to_structure(cleaned_df, base_dir)


def output_cleaned_data_to_structure(df, base_dir):
    if df.empty:
        raise AttributeError('Will not output an empty DataFrame into the data structure.')

    year = df['Year'].iloc[0]
    month = df['Month'].iloc[0]
    if len(str(month)) == 1:
        month = '0{}'.format(month)
    filename = '{}{}_OnTimeData.csv'.format(year, month)

    dataset_dir = Path(base_dir, 'dataset')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    output_path = Path(dataset_dir, filename)
    df.to_csv(output_path, index=False)


def extract_parquet_to_csv(filepath):
    if (Path(filepath).suffix == '.parquet') and (Path(filepath).exists()):
        df = pd.read_parquet(str(filepath))
        return df


if __name__ == '__main__':
    _BASE_DIR = r'C:\Users\Paul\OneDrive - Georgia Institute of Technology\Grad_School\Courses\CSE6242\Projects\dataset'
    _PRIME_AIRPORT = 'SAN'  # San Diego
    extract_all_files(_ZIP_DATA_DIR_, _BASE_DIR, _PRIME_AIRPORT)

    EXTRACT_DIR = Path(_ZIP_DATA_DIR_, 'temp_extracted')
    for file in EXTRACT_DIR.iterdir():
        file.unlink(missing_ok=True)
    EXTRACT_DIR.rmdir()

    # DF = extract_parquet_to_csv(Path(_BASE_DIR, 'dataset', '201801_OnTimeData.parquet'))
    # DF.to_csv(str(Path(_BASE_DIR, 'test_csv_201801.csv')))
