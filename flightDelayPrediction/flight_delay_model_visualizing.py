# Imports
from pathlib import Path
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
from plotly.offline import plot as plt
import random

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


def plot_visuals(scores_df):
    acc_traces = []
    f1_traces = []
    prec_traces = []
    spec_traces = []

    for samp_unq in scores_df['min_samples_leaf'].unique():
        option_color = 'rgb({},{},{})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        chunk = scores_df[scores_df['min_samples_leaf'] == samp_unq]

        acc_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['accuracy'],
            xaxis='x1', yaxis='y1',
            mode='lines+markers',
            line=dict(color=option_color),
            marker=dict(color=option_color),
            name='Accuracy: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))
        acc_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['fnr'],
            xaxis='x1', yaxis='y1',
            mode='lines+markers',
            line=dict(color=option_color, dash='dash'),
            marker=dict(color=option_color),
            name='False Negative Rate: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))

        f1_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['f1'],
            mode='lines+markers',
            line=dict(color=option_color),
            marker=dict(color=option_color),
            name='F1 Score: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))
        f1_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['fnr'],
            xaxis='x1', yaxis='y1',
            mode='lines+markers',
            line=dict(color=option_color, dash='dash'),
            marker=dict(color=option_color),
            name='False Negative Rate: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))

        prec_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['precision'],
            mode='lines+markers',
            line=dict(color=option_color),
            marker=dict(color=option_color),
            name='Precision: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))
        prec_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['fnr'],
            xaxis='x1', yaxis='y1',
            mode='lines+markers',
            line=dict(color=option_color, dash='dash'),
            marker=dict(color=option_color),
            name='False Negative Rate: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))

        spec_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['specificity'],
            mode='lines+markers',
            line=dict(color=option_color),
            marker=dict(color=option_color),
            name='Specificity: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))
        spec_traces.append(go.Scatter(
            x=chunk['max_depth'],
            y=chunk['fnr'],
            xaxis='x1', yaxis='y1',
            mode='lines+markers',
            line=dict(color=option_color, dash='dash'),
            marker=dict(color=option_color),
            name='False Negative Rate: {} % Sample / Leaf'.format(samp_unq * 100.),
            legendgroup=samp_unq
        ))

    acc_layout = dict(title='Accuracy vs False Negative Rate', xaxis=dict(domain=[0, 1], title='Max Depth #'),
                      yaxis=dict(domain=[0, 1], title='Score'))
    f1_layout = dict(title='F1 Scores vs False Negative Rate', xaxis=dict(domain=[0, 1], title='Max Depth #'),
                     yaxis=dict(domain=[0, 1], title='Score'))
    prec_layout = dict(title='Precision vs False Negative Rate', xaxis=dict(domain=[0, 1], title='Max Depth #'),
                       yaxis=dict(domain=[0, 1], title='Score'))
    spec_layout = dict(title='Specificity vs False Negative Rate', xaxis=dict(domain=[0, 1], title='Max Depth #'),
                       yaxis=dict(domain=[0, 1], title='Score'))

    fig = go.Figure(data=acc_traces, layout=acc_layout)
    plt(fig, filename='accuracy_viz.html')

    fig = go.Figure(data=f1_traces, layout=f1_layout)
    plt(fig, filename='f1_viz.html')

    fig = go.Figure(data=prec_traces, layout=prec_layout)
    plt(fig, filename='precision_viz.html')

    fig = go.Figure(data=spec_traces, layout=spec_layout)
    plt(fig, filename='specificity_viz.html')


def score_models(model1, model2, pos_or_neg, test_set, predictor_cols, result_col):
    pred1 = model1.predict(test_set[predictor_cols])
    pred2 = model2.predict(test_set[predictor_cols])

    tp, fn, tn, fp = 0, 0, 0, 0
    for a, p1, p2 in zip(test_set[result_col], pred1, pred2):

        if p1 == pos_or_neg:
            if (a == pos_or_neg) and (pos_or_neg == 1):
                tp += 1
            elif (a == pos_or_neg) and (pos_or_neg == 0):
                tn += 1
            elif (a != pos_or_neg) and (pos_or_neg == 1):
                fp += 1
            elif (a != pos_or_neg) and (pos_or_neg == 0):
                fn += 1
        elif p2 == 1:
            if a == 1:
                tp += 1
            elif a == 0:
                fp += 1
        elif p2 == 0:
            if a == 0:
                tn += 1
            elif a == 1:
                fn += 1

    return [tp, fn, tn, fp]


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
    # merged_cat = pd.read_csv(str(Path(_DATASET_DIR_, 'merged_data_ACTUAL.csv')))
    # saved_merge = merged_cat.copy()
    # pred_df = pd.read_csv('App_Data_w_Predictions.csv')
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
    # ==============================================================================
    merged_cat = pd.read_csv(str(Path(_DATASET_DIR_, 'merged_data_sub2.csv')))
    print('Dataset size: {}'.format(len(merged_cat)))
    # raise ValueError('Break')
    # m_desc = merged_cat.describe()
    # m_desc.to_csv('desc.csv')
    # merged_cat = merged_cat.sample(frac=0.5)
    predictor_cols = ['Quarter', 'Month', 'DayOfWeek', 'Marketing_Airline_Network', 'DepTimeBlk',
                      'ArrTimeBlk', 'StraightLineDistance_Miles', 'Direction_Deg', 'OriginPocketCat',
                      'DestPocketCat', 'OriginHubStatus', 'DestHubStatus', 'OriginTotalFlights',
                      'DestTotalFlights', 'OriginGeographicCentrality', 'DestGeographicCentrality', 'Origin_Cat',
                      'Dest_Cat']
    # result_cols = ['MoreThan30', 'MoreThan45', 'MoreThan60', 'MoreThan75', 'MoreThan90', 'MoreThan105', 'MoreThan120']
    result_cols = ['MoreThan15']
    # ==========================================================================

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

    # app_data = merged_data[(merged_data['Month_Unchanged'] == 12) & (merged_data['Year'] == 2019)]
    # merged_data_less_app = merged_data[~((merged_data['Month_Unchanged'] == 12) & (merged_data['Year'] == 2019))]

    train_set, test_set = train_test_split(merged_data, test_size=0.25)
    print('Train len: {}'.format(len(train_set)))
    print('Test len: {}'.format(len(test_set)))

    parameters = {'n_estimators': [int(x) for x in np.linspace(50, 500, 10)],
                  'max_features': [None, 'sqrt', 'log2'],
                  'max_depth': [5, 50],
                  'min_samples_leaf': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                  'oob_score': [False, True],
                  'class_weight': ['balanced', None]}
    model = rfc()
    clf = GridSearchCV(model, parameters)
    clf.fit(train_set[predictor_cols], train_set[result_cols[0]])
    results = clf.cv_results_
    print('\n\n-------------------------------')
    for k, v in results.items():
        print(k)
        print(v)
        print()

    # print('Making model')
    # Random Forest
    # saved_info = {'max_depth': [], 'min_samples_leaf': [], 'tp': [], 'fn': [], 'tn': [], 'fp': []}
    # model1, output1 = fit_model(train_set, test_set, predictor_cols, result_cols[0], max_depth=30, min_samples_leaf=0.000001)
    # model2, output2 = fit_model(train_set, test_set, predictor_cols, result_cols[0], max_depth=25, min_samples_leaf=0.0001)
    # model3, output3 = fit_model(train_set, test_set, predictor_cols, result_cols[0], max_depth=20, min_samples_leaf=0.000001)
    # model4, output4 = fit_model(train_set, test_set, predictor_cols, result_cols[0], max_depth=20, min_samples_leaf=0.2)

    # with open('model1.pickle', 'wb') as file:
    #     pickle.dump(model1, file)
    # with open('model2.pickle', 'wb') as file:
    #     pickle.dump(model2, file)
    # with open('model3.pickle', 'wb') as file:
    #     pickle.dump(model3, file)
    # with open('model4.pickle', 'wb') as file:
    #     pickle.dump(model4, file)
    #
    # with open('model_scores.txt', 'w') as file:
    #     file.write('Model 1: \n')
    #     file.write(str(output1))
    #     file.write('Model 2: \n')
    #     file.write(str(output2))
    #     file.write('Model 3: \n')
    #     file.write(str(output3))
    #     file.write('Model 4: \n')
    #     file.write(str(output4))
    #
    # test_set.to_csv('TEST_SET.csv', index=False)

    # pred1 = model1.predict(test_set[predictor_cols])  # Model for negatives
    # pred2 = model2.predict(test_set[predictor_cols])
    # print(len(pred1))
    # print(len(pred2))
    # print('Evaluating model')
    # tp, tn, fp, fn = 0, 0, 0, 0
    # for p1, p2, a in zip(pred1, pred2, test_set[result_cols[0]]):
    #
    #     if p1 == 0:
    #         if a == 1:
    #             fn += 1
    #         elif a == 0:
    #             tn += 1
    #     elif p2 == 1:
    #         if a == 1:
    #             tp += 1
    #         elif a == 0:
    #             fp += 1
    #     elif p2 == 0:
    #         if a == 0:
    #             tn += 1
    #         elif a == 1:
    #             fn += 1
    # print('Negative Model Priority!!\n')
    # print('True Positive: {}'.format(tp))
    # print('False Negative: {}'.format(fn))
    # print('True Negative: {}'.format(tn))
    # print('False Positive: {}'.format(fp))
    #
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 / ((1 / precision) + (1 / recall))
    # accuracy = (tp + tn) / (tp + fp + tn + fn)
    #
    # pos_score = tp / (tp + fp)
    # neg_score = tn / (tn + fn)
    # print()
    # print('Positive Score: {}'.format(pos_score))
    # print('Negative Score: {}'.format(neg_score))
    # print()
    # print('Precision: {}'.format(precision))
    # print('Recall: {}'.format(recall))
    # print('F1: {}'.format(f1))
    # print('Accuracy: {}'.format(accuracy))
    #
    # tp, tn, fp, fn = 0, 0, 0, 0
    # for p1, p2, a in zip(pred1, pred2, test_set[result_cols[0]]):
    #
    #     if p2 == 1:
    #         if a == 1:
    #             tp += 1
    #         elif a == 0:
    #             fn += 1
    #     elif p1 == 1:
    #         if a == 1:
    #             tp += 1
    #         elif a == 0:
    #             fp += 1
    #     elif p1 == 0:
    #         if a == 0:
    #             tn += 1
    #         elif a == 1:
    #             fn += 1
    # print('----------------')
    # print('Positive Model Priority!!\n')
    # print('True Positive: {}'.format(tp))
    # print('False Negative: {}'.format(fn))
    # print('True Negative: {}'.format(tn))
    # print('False Positive: {}'.format(fp))
    #
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 / ((1 / precision) + (1 / recall))
    # accuracy = (tp + tn) / (tp + fp + tn + fn)
    #
    # pos_score = tp / (tp + fp)
    # neg_score = tn / (tn + fn)
    # print()
    # print('Positive Score: {}'.format(pos_score))
    # print('Negative Score: {}'.format(neg_score))
    # print()
    # print('Precision: {}'.format(precision))
    # print('Recall: {}'.format(recall))
    # print('F1: {}'.format(f1))
    # print('Accuracy: {}'.format(accuracy))

    # for depth in [5, 8, 10, 12, 15, 18, 20, 25, 30]:
    #     for min_samp in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]:
    #         print('Depth: {}, Min Samp Rate: {}'.format(depth, min_samp))
    #         _, output = fit_model(train_set, test_set, predictor_cols, result_cols[0], max_depth=depth, min_samples_leaf=min_samp)
    #         saved_info['max_depth'].append(depth)
    #         saved_info['min_samples_leaf'].append(min_samp)
    #         saved_info['tp'].append(output[0])
    #         saved_info['fn'].append(output[1])
    #         saved_info['tn'].append(output[2])
    #         saved_info['fp'].append(output[3])

    # scores_df = pd.DataFrame(saved_info)
    # scores_df.to_csv('raw_scores.csv', index=False)
    #
    # scores_df['accuracy'] = (scores_df['tp'] + scores_df['tn']) / (scores_df['tp'] + scores_df['fn'] + scores_df['tn'] + scores_df['fp'])
    # scores_df['precision'] = scores_df['tp'] / (scores_df['tp'] + scores_df['fp'])
    # scores_df['recall'] = scores_df['tp'] / (scores_df['tp'] + scores_df['fn'])
    # scores_df['specificity'] = scores_df['tn'] / (scores_df['tn'] + scores_df['fp'])
    # scores_df['fnr'] = scores_df['fn'] / (scores_df['fn'] + scores_df['tp'])
    # scores_df['f1'] = 2 * scores_df['precision'] * scores_df['recall'] / (scores_df['precision'] + scores_df['recall'])
    #
    # scores_df.to_csv('model_scores_viz_df.csv', index=False)
    # =====================================================================================

    # scores_df = pd.read_csv('model_scores_viz_df.csv')
    # plot_visuals(scores_df)

    # for result in result_cols:
    #     print('\nModeling for {}'.format(result))
    #     res_col = result
    #     model, _ = fit_model(train_set, test_set, predictor_cols, res_col)
    #     pred = model.predict_proba(app_data[predictor_cols])
    #     app_data['Prediction_{}_0'.format(result)] = [i[0] for i in pred]
    #     app_data['Prediction_{}_1'.format(result)] = [i[1] for i in pred]
    #     app_data.to_csv('temp_app_data_{}_complete.csv'.format(result))
    #

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

    # full_app_data = saved_merge.iloc[app_data.index]
    # full_app_data['Prediction_15min'] = app_data['Prediction_MoreThan15_1'].tolist()
    # full_app_data['Prediction_30min'] = pred_df['Prediction_MoreThan30_1'].tolist()
    # full_app_data['Prediction_45min'] = pred_df['Prediction_MoreThan45_1'].tolist()
    # full_app_data['Prediction_60min'] = pred_df['Prediction_MoreThan60_1'].tolist()
    # full_app_data['Prediction_75min'] = pred_df['Prediction_MoreThan75_1'].tolist()
    # full_app_data['Prediction_90min'] = pred_df['Prediction_MoreThan90_1'].tolist()
    # full_app_data['Prediction_105min'] = pred_df['Prediction_MoreThan105_1'].tolist()
    # full_app_data['Prediction_120min'] = pred_df['Prediction_MoreThan120_1'].tolist()
    #
    # print(full_app_data.head())
    # full_app_data.to_csv('FINAL3.csv', index=False)

    # test_set = pd.read_csv('TEST_SET.csv')
    #
    # with open('model1.pickle', 'rb') as file:
    #     model1 = pickle.load(file)
    # with open('model2.pickle', 'rb') as file:
    #     model2 = pickle.load(file)
    # with open('model3.pickle', 'rb') as file:
    #     model3 = pickle.load(file)
    # with open('model4.pickle', 'rb') as file:
    #     model4 = pickle.load(file)
    #
    # saved_scores = {'model1': [], 'model2': [], 'pos_neg': [], 'tp': [], 'fn': [], 'tn': [], 'fp': []}
    #
    # for m1 in [(model1, 'm1'), (model2, 'm2'), (model3, 'm3'), (model4, 'm4')]:
    #     for m2 in [(model1, 'm1'), (model2, 'm2'), (model3, 'm3'), (model4, 'm4')]:
    #         if m1[1] == m2[1]:
    #             continue
    #         for pos_neg in [0, 1]:
    #             out = score_models(m1[0], m2[0], pos_neg, test_set, predictor_cols, result_cols[0])
    #             saved_scores['model1'].append(m1[1])
    #             saved_scores['model2'].append(m2[1])
    #             saved_scores['pos_neg'].append(pos_neg)
    #             saved_scores['tp'].append(out[0])
    #             saved_scores['fn'].append(out[1])
    #             saved_scores['tn'].append(out[2])
    #             saved_scores['fp'].append(out[3])
    #
    # scores_df = pd.DataFrame(saved_scores)
    # scores_df['accuracy'] = (scores_df['tp'] + scores_df['tn']) / (scores_df['tp'] + scores_df['fn'] + scores_df['tn'] + scores_df['fp'])
    # scores_df['precision'] = scores_df['tp'] / (scores_df['tp'] + scores_df['fp'])
    # scores_df['recall'] = scores_df['tp'] / (scores_df['tp'] + scores_df['fn'])
    # scores_df['specificity'] = scores_df['tn'] / (scores_df['tn'] + scores_df['fp'])
    # scores_df['fnr'] = scores_df['fn'] / (scores_df['fn'] + scores_df['tp'])
    # scores_df['f1'] = 2 * scores_df['precision'] * scores_df['recall'] / (scores_df['precision'] + scores_df['recall'])
    #
    # scores_df.to_csv('model_combo_scores.csv', index=False)
