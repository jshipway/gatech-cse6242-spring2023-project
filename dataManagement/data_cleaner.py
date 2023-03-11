# Imports
import pandas as pd
from pathlib import Path
import zipfile

_ZIP_DATA_DIR_ = r'C:\Users\pfarmer8-gtri\Box\Paul_Farmer\Grad School\Material\CSE6242 - Data and Visual Analytics\Project\Marketing On-time Data 2018-19'
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
        extract_dir = Path(Path(zip_file_path).parent, 'data_extracted')
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

    fixed_df = finalize_dataset(final_df)

    return fixed_df


def finalize_dataset(df):
    # This method is intended to finalize all the features needed for the ML process, as well as
    # any other columns needed for visualization

    # TODO: Actually make this do something lol
    return df.copy()


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
    filename = '{}{}_OnTimeData.parquet'.format(year, month)

    dataset_dir = Path(base_dir, 'dataset')
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    output_path = Path(dataset_dir, filename)
    df.to_parquet(output_path, index=False)


def extract_parquet_to_csv(filepath):
    if (Path(filepath).suffix == '.parquet') and (Path(filepath).exists()):
        df = pd.read_parquet(str(filepath))
        return df


if __name__ == '__main__':
    _BASE_DIR = r'C:\Users\pfarmer8-gtri\Box\Paul_Farmer\Grad School\Material\CSE6242 - Data and Visual Analytics\Project'
    _PRIME_AIRPORT = 'SAN'  # San Diego
    extract_all_files(_ZIP_DATA_DIR_, _BASE_DIR, _PRIME_AIRPORT)

    # NOTE: Delete extracted CSV files
    DF = extract_parquet_to_csv(Path(_BASE_DIR, 'dataset', '201801_OnTimeData.parquet'))
    DF.to_csv(str(Path(_BASE_DIR, 'test_csv_201801.csv')))
