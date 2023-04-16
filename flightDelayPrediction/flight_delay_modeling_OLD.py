# Imports
from pathlib import Path
import pandas as pd
import time
import pickle
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

_DATASET_DIR_ = r'C:\Users\Paul\OneDrive - Georgia Institute of Technology\Grad_School\Courses\CSE6242\Projects\dataset\dataset'


def merge_datasets(dataset_dir, data_fn_pattern):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise NotADirectoryError(
            'Could not find the defined directory for the dataset. Received {}.'.format(str(dataset_dir)))

    concat_df = pd.DataFrame()
    for file in dataset_dir.iterdir():
        if any([i in str(file) for i in ['201801', '201802', '201803', '201804', '201805', '201806', '201807']]):
            continue  # Skip it
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


def delay_categorization(in_df):
    df = in_df.copy()
    df['TotalDelay'] = df['CarrierDelay'] + df['WeatherDelay'] + df['NASDelay'] + df['LateAircraftDelay']
    df['NoDelay'] = (df['TotalDelay'].isna()).astype(int)
    df['AnyDelay'] = (~df['TotalDelay'].isna()).astype(int)
    df['MoreThan15'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 15)).astype(int)
    df['MoreThan30'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 30)).astype(int)
    df['MoreThan45'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 45)).astype(int)
    df['MoreThan60'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 60)).astype(int)
    df['MoreThan75'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 75)).astype(int)
    df['MoreThan90'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 90)).astype(int)
    df['MoreThan105'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 105)).astype(int)
    df['MoreThan120'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 120)).astype(int)
    df['MoreThan150'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 150)).astype(int)
    df['MoreThan180'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 180)).astype(int)
    df['MoreThan210'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 210)).astype(int)
    df['MoreThan240'] = (~df['TotalDelay'].isna() & (df['TotalDelay'] > 240)).astype(int)
    return df


def fit_model(train_set, test_set, predictor_cols, result_cols, n_trees=100, **kwargs):
    st = time.time()
    model15 = rfc(n_estimators=n_trees, class_weight='balanced', **kwargs)
    # model15 = knn(n_neighbors=10)
    # model15 = svc(gamma='scale', class_weight='balanced')
    model15.fit(train_set[predictor_cols], train_set[result_cols])
    # pred = model15.predict_proba(test_set[predictor_cols])
    #
    # for p, a in zip(pred, test_set['MoreThan15']):
    #     if p[0] > cutoff:
    pred = model15.predict(test_set[predictor_cols])
    print('Evaluating model')
    tp, tn, fp, fn = 0, 0, 0, 0
    for p, a in zip(pred, test_set[result_cols]):
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

    print('True Positive: {}'.format(tp))
    print('False Negative: {}'.format(fn))
    print('True Negative: {}'.format(tn))
    print('False Positive: {}'.format(fp))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 / ((1 / precision) + (1 / recall))

    pos_score = tp / (tp + fp)
    neg_score = tn / (tn + fn)
    print()
    print('Positive Score: {}'.format(pos_score))
    print('Negative Score: {}'.format(neg_score))
    print()
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))
    scores = [tp, fn, tn, fp]

    print('\n\n')
    print(round(time.time() - st, 3) / 60, 'min')
    return model15, scores


if __name__ == '__main__':
    st_tot = time.time()
    predictor_cols = ['Quarter', 'Month', 'DayOfWeek', 'Marketing_Airline_Network', 'DepTimeBlk',
                      'ArrTimeBlk', 'StraightLineDistance_Miles', 'Direction_Deg', 'OriginPocketCat',
                      'DestPocketCat', 'OriginHubStatus', 'DestHubStatus', 'OriginTotalFlights',
                      'DestTotalFlights', 'OriginGeographicCentrality', 'DestGeographicCentrality', 'Origin_Cat',
                      'Dest_Cat']

    result_cols = ['MoreThan30', 'MoreThan45', 'MoreThan60', 'MoreThan75', 'MoreThan90', 'MoreThan105', 'MoreThan120']

    # merged = merge_datasets(_DATASET_DIR_, data_fn_pattern='OnTimeData')
    # merged.to_csv(str(Path(_DATASET_DIR_, 'merged_data.csv')))
    # merged = pd.read_csv(str(Path(_DATASET_DIR_, 'merged_data.csv')))
    # merged_cat = delay_categorization(merged)
    # merged_cat.to_csv(str(Path(_DATASET_DIR_, 'merged_data_ACTUAL.csv')))
    merged_cat = pd.read_csv(str(Path(_DATASET_DIR_, 'merged_data_ACTUAL.csv')))
    saved_merge = merged_cat.copy()
    pred_df = pd.read_csv('App_Data_w_Predictions.csv')
    # dec_df = merged_cat[(merged_cat['Month'] == 12) & (merged_cat['Year'] == 2019)]
    # all_cols = list(dec_df.columns)
    # dec_df = dec_df[dec_df['Cancelled'] == 0]
    # dec_df.dropna(axis=0, inplace=True)
    # final_df = dec_df.copy()
    # final_df['Prediction_30min'] = pred_df['Prediction_MoreThan30_1'].tolist()
    # final_df['Prediction_45min'] = pred_df['Prediction_MoreThan45_1'].tolist()
    # final_df['Prediction_60min'] = pred_df['Prediction_MoreThan60_1'].tolist()
    # final_df['Prediction_75min'] = pred_df['Prediction_MoreThan75_1'].tolist()
    # final_df['Prediction_90min'] = pred_df['Prediction_MoreThan90_1'].tolist()
    # final_df['Prediction_105min'] = pred_df['Prediction_MoreThan105_1'].tolist()
    # final_df['Prediction_120min'] = pred_df['Prediction_MoreThan120_1'].tolist()
    # final_df.to_csv('FINAL.csv', index=False)

    # UNCOMMENT FOR MODELING

    # merged_cat_portion = merged_cat.sample(frac=0.2)
    # merged_cat_portion.to_csv(str(Path(_DATASET_DIR_, 'merged_data_sub2.csv')))
    # merged_cat = pd.read_csv(str(Path(_DATASET_DIR_, 'merged_data_sub2.csv')))
    # print('Dataset size: {}'.format(len(merged_cat)))
    # raise ValueError('Break')
    # m_desc = merged_cat.describe()
    # m_desc.to_csv('desc.csv')
    # merged_cat = merged_cat.sample(frac=0.5)
    predictor_cols = ['Quarter', 'Month', 'DayOfWeek', 'Marketing_Airline_Network', 'DepTimeBlk',
                      'ArrTimeBlk', 'StraightLineDistance_Miles', 'Direction_Deg', 'OriginPocketCat',
                      'DestPocketCat', 'OriginHubStatus', 'DestHubStatus', 'OriginTotalFlights',
                      'DestTotalFlights', 'OriginGeographicCentrality', 'DestGeographicCentrality', 'Origin_Cat',
                      'Dest_Cat']
    result_cols = ['MoreThan15', 'MoreThan30', 'MoreThan45', 'MoreThan60', 'MoreThan75', 'MoreThan90', 'MoreThan105',
                   'MoreThan120']
    # result_cols = ['MoreThan15']

    le = LabelEncoder()
    ss = StandardScaler()
    cutoff = 0.75

    merged_cat['DepTimeBlk'] = le.fit_transform(merged_cat['DepTimeBlk'])
    merged_cat['ArrTimeBlk'] = le.fit_transform(merged_cat['ArrTimeBlk'])
    merged_cat['Marketing_Airline_Network'] = le.fit_transform(merged_cat['Marketing_Airline_Network'])
    merged_cat['Origin_Cat'] = le.fit_transform(merged_cat['Origin'])
    merged_cat['Dest_Cat'] = le.fit_transform(merged_cat['Dest'])

    print('Prepping model')
    # Model 15
    merged_data = merged_cat[predictor_cols]
    merged_data = pd.DataFrame(ss.fit_transform(merged_data), columns=predictor_cols)
    merged_data[result_cols] = merged_cat[result_cols]
    merged_data['Year'] = merged_cat['Year']
    merged_data['Month_Unchanged'] = merged_cat['Month']
    merged_data.dropna(axis=0, inplace=True)

    app_data = merged_data[(merged_data['Month_Unchanged'] == 12) & (merged_data['Year'] == 2019)]
    merged_data_less_app = merged_data[~((merged_data['Month_Unchanged'] == 12) & (merged_data['Year'] == 2019))]

    train_set, test_set = train_test_split(merged_data_less_app, test_size=0.1)

    # print('Making model')
    # Random Forest
    saved_info = {'prediction': [], 'm1_features': [], 'm1_feature_importance': [], 'm2_features': [],
                  'm2_feature_importance': [], 'tp': [], 'fn': [], 'tn': [], 'fp': []}

    for result in result_cols:
        print('\nModeling for {}'.format(result))
        res_col = result
        model1, _ = fit_model(train_set, test_set, predictor_cols, res_col, max_depth=20, min_samples_leaf=0.2)
        model2, _ = fit_model(train_set, test_set, predictor_cols, res_col, max_depth=25, min_samples_leaf=0.0001)
        pred1 = model1.predict_proba(app_data[predictor_cols])
        pred2 = model2.predict_proba(app_data[predictor_cols])
        rel_pred = []

        tp, fn, tn, fp = 0, 0, 0, 0
        for a, p1, p2 in zip(app_data[result], pred1, pred2):
            if p1[1] <= 0.5:
                rel_pred.append(p1[1])  # Use model 1 prediction
                if a == 0:
                    tn += 1
                elif a == 1:
                    fn += 1
            else:
                rel_pred.append(p2[1])  # Use model 2 prediction
                if p2[1] <= 0.5:
                    if a == 0:
                        tn += 1
                    elif a == 1:
                        fn += 1
                elif p2[1] > 0.5:
                    if a == 1:
                        tp += 1
                    elif a == 0:
                        fp += 1

        saved_info['prediction'].append(result)
        saved_info['m1_features'].append(model1.feature_names_in_)
        saved_info['m1_feature_importance'].append(model1.feature_importances_)
        saved_info['m2_features'].append(model2.feature_names_in_)
        saved_info['m2_feature_importance'].append(model2.feature_importances_)
        saved_info['tp'].append(tp)
        saved_info['fn'].append(fn)
        saved_info['tn'].append(tn)
        saved_info['fp'].append(fp)

        # app_data['Prediction_{}_0'.format(result)] = [i[0] for i in pred1]
        # app_data['Prediction_{}_1'.format(result)] = [i[1] for i in pred2]
        app_data['Prediction_{}_1'.format(result)] = rel_pred
        app_data.to_csv('temp_app_data_{}_complete.csv'.format(result))

    saved_df = pd.DataFrame(saved_info)
    saved_df.to_csv('saved_info_test.csv', index=False)

    #
    # app_data.to_csv('App_Data_w_Predictions.csv')
    #
    # en_tot = time.time()
    # totl = round((en_tot - st_tot)/60, 3)
    # print('Total Time: {} min'.format(totl))
    # model3 = fit_model(train_set, test_set, predictor_cols, result_cols, max_depth=10, min_samples_leaf=0.002)
    #
    # with open('model15_50pctmodel_small0002.pickle', 'wb') as file:
    #     pickle.dump(model3, file)

    full_app_data = saved_merge.iloc[app_data.index]
    full_app_data['Prediction_15min'] = app_data['Prediction_MoreThan15_1'].tolist()
    full_app_data['Prediction_30min'] = pred_df['Prediction_MoreThan30_1'].tolist()
    full_app_data['Prediction_45min'] = pred_df['Prediction_MoreThan45_1'].tolist()
    full_app_data['Prediction_60min'] = pred_df['Prediction_MoreThan60_1'].tolist()
    full_app_data['Prediction_75min'] = pred_df['Prediction_MoreThan75_1'].tolist()
    full_app_data['Prediction_90min'] = pred_df['Prediction_MoreThan90_1'].tolist()
    full_app_data['Prediction_105min'] = pred_df['Prediction_MoreThan105_1'].tolist()
    full_app_data['Prediction_120min'] = pred_df['Prediction_MoreThan120_1'].tolist()

    print(full_app_data.head())
    full_app_data.to_csv('improvement_test.csv', index=False)
