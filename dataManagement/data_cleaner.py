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
                     'Distance', 'DistanceGroup', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay']

_REGION_MAP = {'AK': 'Alaska', 'AL': 'EastSouthCentral', 'AR': 'WestSouthCentral', 'AZ': 'MountainSouth',
               'CA': 'Pacific', 'CO': 'MountainSouth', 'CT': 'NewEngland', 'DE': 'MidAtlantic', 'DC': 'MidAtlantic',
               'FL': 'SouthAtlantic', 'GA': 'SouthAtlantic', 'HI': 'Hawaii', 'ID': 'MountainNorth',
               'IN': 'MidwestGreatLakes', 'IA': 'MidwestCentral', 'IL': 'MidwestGreatLakes', 'KS': 'MidwestCentral',
               'KY': 'EastSouthCentral', 'LA': 'WestSouthCentral', 'ME': 'NewEngland', 'MD': 'MidAtlantic',
               'MA': 'NewEngland', 'MI': 'MidwestGreatLakes', 'MN': 'MidwestCentral', 'MS': 'EastSouthCentral',
               'MO': 'MidwestCentral', 'MT': 'MountainNorth', 'NE': 'MidwestCentral', 'NV': 'MountainSouth',
               'NH': 'NewEngland', 'NJ': 'MidAtlantic', 'NM': 'MountainSouth', 'NY': 'MidAtlantic',
               'NC': 'SouthAtlantic', 'ND': 'MidwestCentral', 'OH': 'MidwestGreatLakes', 'OK': 'WestSouthCentral',
               'OR': 'Pacific', 'PA': 'MidAtlantic', 'PR': 'Caribbean', 'RI': 'NewEngland', 'SC': 'SouthAtlantic',
               'SD': 'MidwestCentral', 'TN': 'EastSouthCentral', 'TX': 'WestSouthCentral', 'UT': 'MountainSouth',
               'VA': 'SouthAtlantic', 'VT': 'NewEngland', 'VI': 'Caribbean', 'WA': 'Pacific', 'WV': 'SouthAtlantic',
               'WI': 'MidwestGreatLakes', 'WY': 'MountainNorth'}
_REGION_CAT_MAP = {'Alaska': 1, 'Hawaii': 2, 'Pacific': 3, 'MountainSouth': 4, 'MountainNorth': 5, 'MidwestCentral': 6,
                   'WestSouthCentral': 7, 'EastSouthCentral': 8, 'MidwestGreatLakes': 9, 'SouthAtlantic': 10,
                   'MidAtlantic': 11, 'NewEngland': 12, 'Caribbean': 13}


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
    if not Path('supplemental_data', 'airport_size.csv').exists():
        all_unq_airports = list(set(list(df['Origin'].unique()) + list(df['Dest'].unique())))
        busy_count = {'Airport': ['Dep', 'Arr']}
        for apt in all_unq_airports:
            if apt not in busy_count:
                busy_count[apt] = [0, 0]
            busy_count[apt][0] = (df.Origin.values == apt).sum()
            busy_count[apt][1] = (df.Dest.values == apt).sum()

        airport_size_df = pd.DataFrame(busy_count).reset_index(drop=True)
        airport_size_df.to_csv(str(Path('supplemental_data', 'airport_size.csv')))
    else:
        airport_size_df = pd.read_csv(str(Path('supplemental_data', 'airport_size.csv')))

    fixed_df = finalize_dataset(final_df, airport_size_df)

    return fixed_df


def get_hub_status(airport, carrier, hub_df):
    if carrier not in hub_df.columns:
        return 0
    else:
        subset = hub_df[(hub_df['Airport'] == airport)]
        if not subset.empty:
            return subset[carrier].iloc[0]
        else:
            return 0


def get_airport_size(airport, size_df):
    if airport in size_df.columns:
        return size_df[airport].iloc[0] + size_df[airport].iloc[1]
    else:
        return None


def get_region_category(state):
    if state in _REGION_MAP.keys():
        return _REGION_CAT_MAP[_REGION_MAP[state]]
    else:
        return None


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

    sub_df[['StraightLineDistance_Miles', 'Direction_Deg']] = sub_df.apply(
        lambda x: cu.get_direction_range([x['Origin_Lat'], x['Origin_Lon']], [x['Dest_Lat'], x['Dest_Lon']]), axis=1)
    sub_df['OriginTotalFlights'] = sub_df.apply(lambda x: get_airport_size(x['Origin'], airport_size_df), axis=1)
    sub_df['DestTotalFlights'] = sub_df.apply(lambda x: get_airport_size(x['Dest'], airport_size_df), axis=1)
    sub_df['OriginGeographicCentrality'] = sub_df.apply(
        lambda x: cu.get_geographic_centrality_category(x['Origin_Lat'], x['Origin_Lon']), axis=1)
    sub_df['DestGeographicCentrality'] = sub_df.apply(
        lambda x: cu.get_geographic_centrality_category(x['Dest_Lat'], x['Dest_Lon']), axis=1)

    return sub_df


def extract_all_files(zip_dir, base_dir, prime_airport=None):
    if Path(zip_dir).is_dir():
        for fle in Path(zip_dir).iterdir():
            if fle.suffix == '.zip':
                print('\nProcessing file: {}\n'.format(fle))
                df = extract_csv_from_zip(fle)
                print('\n\tCleaning...')
                cleaned_df = clean_dataframe(df, prime_airport=prime_airport)
                print('\n\tOutputting...')
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
    # UNCOMMENT TO RUN ----------------------------------------------
    _BASE_DIR = input('Input base directory that the downloaded dataset exists in: ')
    # _BASE_DIR = r'C:\Users\Paul\OneDrive - Georgia Institute of Technology\Grad_School\Courses\CSE6242\Projects\dataset'
    _PRIME_AIRPORT = input('Input the primary airport of interest for the modeling and visualization. '
                           'This will be the origin airport: ')
    # _PRIME_AIRPORT = 'SAN'  # San Diego
    extract_all_files(_ZIP_DATA_DIR_, _BASE_DIR, _PRIME_AIRPORT)

    EXTRACT_DIR = Path(_ZIP_DATA_DIR_, 'temp_extracted')
    for file in EXTRACT_DIR.iterdir():
        file.unlink(missing_ok=True)
    EXTRACT_DIR.rmdir()
    # ---------------------------------------------------------------

    # DF = extract_parquet_to_csv(Path(_BASE_DIR, 'dataset', '201801_OnTimeData.parquet'))
    # DF.to_csv(str(Path(_BASE_DIR, 'test_csv_201801.csv')))

    # from plotly import graph_objects as go
    # color_scheme = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b',
    #                 '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
    #
    # df = pd.read_excel(r'C:\Users\Paul\OneDrive - Georgia Institute of Technology\Grad_School\Courses\CSE6242\Projects\repository\gatech-cse6242-spring2023-project\dataManagement\supplemental_data\Airport_locations_v2.xlsx')
    # x = []
    # y = []
    # catList = []
    # namelist = []
    # for ndx, row in df.iterrows():
    #     lat = getattr(row, 'FAA_Lat')
    #     lon = getattr(row, 'FAA_Long')
    #     name = getattr(row, 'Airport')
    #     state = getattr(row, 'Airport_State')
    #
    #     cat = get_region_category(state)
    #     if cat is None:
    #         continue
    #     x.append(lon)
    #     y.append(lat)
    #     catList.append(color_scheme[(cat-1)%10])
    #     namelist.append(name)
    #
    # data = [go.Scatter(x=x, y=y, mode='markers', marker=dict(color=catList))]
    # fig = go.Figure(data=data)
    # fig.show()
