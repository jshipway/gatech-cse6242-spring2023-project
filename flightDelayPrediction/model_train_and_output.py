# Imports
from pathlib import Path
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def merge_datasets(dataset_dir, data_fn_pattern='OnTimeData', merge_start_date='08/2018', merge_stop_date='12/2019'):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise NotADirectoryError(
            'Could not find the defined directory for the dataset. Received {}.'.format(str(dataset_dir)))

    concat_df = pd.DataFrame()

    from_year = int(merge_start_date.split('/')[1])
    to_year = int(merge_stop_date.split('/')[1])
    from_month = int(merge_start_date.split('/')[0])
    to_month = int(merge_stop_date.split('/')[0])

    for file in dataset_dir.iterdir():
        file_year = int(str(file)[0:4])
        file_month = int(str(file)[4:6])
        if (file_year < from_year) or (file_year == from_year and file_month < from_month) or \
                (file_year > to_year) or (file_year == to_year and file_month > to_month):
            continue
        if data_fn_pattern in str(file):
            print('File used: {}'.format(str(file)))
            if file.suffix == '.csv':
                df = pd.read_csv(str(file))
            elif file.suffix == 'xlsx':
                df = pd.read_excel(str(file))
            else:
                continue

            concat_df = pd.concat([concat_df, df]).reset_index(drop=True)

    return concat_df


def delay_categorization(in_df, delay_cats=['CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay'],
                         delay_mins=[15, 30, 45, 60, 75, 90, 105, 120]):
    df = in_df.copy()

    df['TotalDelay'] = 0
    for d in delay_cats:
        if d in df.columns:
            df['TotalDelay'] += df[d]

    df['NoDelay'] = (df['TotalDelay'].isna()).astype(int)
    df['AnyDelay'] = (~df['TotalDelay'].isna()).astype(int)

    result_cols = []
    for d_min in delay_mins:
        df['MoreThan{}'.format(d_min)] = (~df['TotalDelay'].isna() & (df['TotalDelay'] >= d_min)).astype(int)
        result_cols.append('MoreThan{}'.format(d_min))

    return df, result_cols


def fit_model(train_set, test_set, predictor_cols, result_col, n_trees=100, **kwargs):
    model = rfc(n_estimators=n_trees, class_weight='balanced', **kwargs)
    model.fit(train_set[predictor_cols], train_set[result_col])
    pred = model.predict(test_set[predictor_cols])

    scores = score_model(pred, test_set[result_col])

    return model, scores


def _category_score(p, a, tp, tn, fp, fn):
    if p == 1:
        if a == 1:
            tp += 1
        elif a == 0:
            fp += 1
    elif p == 0:
        if a == 0:
            tn += 1
        elif a == 1:
            fn += 1

    return tp, tn, fp, fn


def _probability_score(p, a, tp, tn, fp, fn):
    if p[1] < 0.5:
        if a == 0:
            tn += 1
        elif a == 1:
            fn += 1
    elif p[1] >= 0.5:
        if a == 1:
            tp += 1
        elif a == 0:
            fp += 1

    return tp, tn, fp, fn


def score_model(pred, act, score_type='category'):
    if score_type not in ['category', 'probability']:
        raise AttributeError('Expected either scoring categories or scoring probability.')

    tp, tn, fp, fn = 0, 0, 0, 0

    for p, a in zip(pred, act):
        if score_type == 'category':
            tp, tn, fp, fn = _category_score(p, a, tp, tn, fp, fn)
        elif score_type == 'probability':
            tp, tn, fp, fn = _probability_score(p, a, tp, tn, fp, fn)

    if (tp + fp + tn + fn) > 0:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
    else:
        accuracy = None

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = None

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
        fnr = fn / (tp + fn)
    else:
        recall = None
        fnr = None

    if not isinstance(precision, type(None)) and not isinstance(recall, type(None)):
        f1 = 2 / ((1 / precision) + (1 / recall))
    else:
        f1 = None

    scores = {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'accuracy': accuracy,
              'precision': precision, 'recall': recall, 'fnr': fnr, 'f1': f1}
    return scores


def prepare_dataset_for_model(dataset, predictor_cols, result_cols, test_pct=0.15,
                              app_data_month=12, app_data_year=2019):
    le = LabelEncoder()  # Category encoder to values
    ss = StandardScaler()  # Scaler

    standard_cats = ['DepTimeBlk', 'ArrTimeBlk', 'Marketing_Airline_Network', 'Origin', 'Dest']
    for cat in standard_cats:
        if cat in predictor_cols:
            dataset[cat] = le.fit_transform(dataset[cat])

    model_dataset = pd.DataFrame(ss.fit_transform(dataset[predictor_cols]), columns=predictor_cols)
    model_dataset[result_cols] = dataset[result_cols]
    model_dataset['Year'] = dataset['Year']
    model_dataset['Month_Unchanged'] = dataset['Month']
    model_dataset.dropna(axis=0, inplace=True)

    app_data = model_dataset[(model_dataset['Month_Unchanged'] == app_data_month) &
                             (model_dataset['Year'] == app_data_year)]
    model_dataset_noapp = model_dataset[~((model_dataset['Month_Unchanged'] == app_data_month) &
                                          (model_dataset['Year'] == app_data_year))]

    model_train_set, model_test_set = train_test_split(model_dataset_noapp, test_size=test_pct)

    return model_train_set, model_test_set, app_data


def train_all_models(train_set, test_set, app_data, predictor_cols, result_cols, save_model=False, **kwargs):
    info_save = {'delay_model': [], 'features': [], 'feature_importance': [], 'tp': [], 'fn': [],
                 'tn': [], 'fp': [], 'accuracy': [], 'precision': [], 'recall': [], 'fnr': [], 'f1': [],
                 'app_tp': [], 'app_fn': [], 'app_tn': [], 'app_fp': [], 'app_accuracy': [],
                 'app_precision': [], 'app_recall': [], 'app_fnr': [], 'app_f1': []}
    if save_model:
        if not Path('saved_models').exists():
            Path('saved_models').mkdir(parents=True)

    for result in result_cols:
        print('Modeling for {}'.format(result))
        model, scores = fit_model(train_set, test_set, predictor_cols, result, **kwargs)

        pred = model.predict_proba(app_data[predictor_cols])
        app_data['Prediction_{}'.format(result)] = [i[1] for i in pred]

        app_scores = score_model(pred, test_set[result], score_type='probability')

        if save_model:
            with open(str(Path('saved_models', 'Model_{}.mdl'.format(result))), 'wb') as file:
                pickle.dump(model, file)

        info_save['delay_model'].append(result)
        info_save['features'].append(list(model.feature_names_in_))
        info_save['feature_importance'].append(list(model.feature_importances_))
        info_save['tp'].append(scores['tp'])
        info_save['fn'].append(scores['fn'])
        info_save['tn'].append(scores['tn'])
        info_save['fp'].append(scores['fp'])
        info_save['accuracy'].append(scores['accuracy'])
        info_save['precision'].append(scores['precision'])
        info_save['recall'].append(scores['recall'])
        info_save['fnr'].append(scores['fnr'])
        info_save['f1'].append(scores['f1'])
        info_save['app_tp'].append(app_scores['tp'])
        info_save['app_fn'].append(app_scores['fn'])
        info_save['app_tn'].append(app_scores['tn'])
        info_save['app_fp'].append(app_scores['fp'])
        info_save['app_accuracy'].append(app_scores['accuracy'])
        info_save['app_precision'].append(app_scores['precision'])
        info_save['app_recall'].append(app_scores['recall'])
        info_save['app_fnr'].append(app_scores['fnr'])
        info_save['app_f1'].append(app_scores['f1'])

    save_df = pd.DataFrame(info_save)
    return save_df, app_data


def run(dataset_dir, app_month=12, app_year=2019, save_model=False, test_pct=0.15, **kwargs):
    predictor_cols = ['Quarter', 'Month', 'DayOfWeek', 'Marketing_Airline_Network', 'DepTimeBlk',
                      'ArrTimeBlk', 'StraightLineDistance_Miles', 'Direction_Deg', 'OriginPocketCat',
                      'DestPocketCat', 'OriginHubStatus', 'DestHubStatus', 'OriginTotalFlights',
                      'DestTotalFlights', 'OriginGeographicCentrality', 'DestGeographicCentrality', 'Origin',
                      'Dest']

    print('Merging Datasets')
    dataset = merge_datasets(dataset_dir)
    print('Categorizing Delay')
    dataset, result_cols = delay_categorization(dataset)
    print('Preparing Datasets')
    train_set, test_set, app_data = prepare_dataset_for_model(dataset, predictor_cols, result_cols, test_pct=test_pct,
                                                              app_data_month=app_month, app_data_year=app_year)

    print('Training Models')
    save_df, output_app_data = train_all_models(train_set, test_set, app_data, predictor_cols, result_cols,
                                                save_model=save_model, **kwargs)

    print('Outputting Results')
    if not Path('model_run_output').exists():
        Path('model_run_output').mkdir(parents=True)

    save_df.to_csv(str(Path('model_run_output', 'saved_model_info.csv')), index=False)
    output_app_data.to_csv(str(Path('model_run_output', 'AppData_w_Predictions.csv')), index=False)


if __name__ == '__main__':
    DATASET_DIR = input('Input the directory that contains all the data for training, testing, and outputting '
                        'from the models: ')

    APP_MONTH = int(input('What month do you want to use for predicting? '
                          'This month will be the one used in the visualization. (1-12): '))
    APP_YEAR = int(input('What year do you want to use for predicting?'
                         'This year will be the one used in the visualization. (Full year, i.e. 2019): '))

    save_models = input('Do you wish to output the models as a .mdl file (pickle file)? (Y or N): ')
    if save_models.upper() == 'Y':
        SAVE_MODEL = True
    else:
        SAVE_MODEL = False

    print('Running model training, validation, and prediction!\n\n')
    run(DATASET_DIR, app_month=APP_MONTH, app_year=APP_YEAR, save_model=SAVE_MODEL)
