# Imports
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path
import pickle
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


def knn_evaluation(train_set, test_set, predictor_cols, result_col):
    print('\nKNN Evaluation\n')
    knn_output = dict()
    knn_output['predictor_columns'] = predictor_cols
    for n in [3, 5, 8, 10, 12, 15, 20]:
        for alg in ['auto', 'ball_tree', 'kd_tree']:
            print('N: {}, ALG: {}'.format(n, alg))
            key = '{}_{}'.format(alg, n)
            start = time.time()
            model = KNeighborsClassifier(n_neighbors=n, algorithm=alg)
            model.fit(train_set[predictor_cols], train_set[result_col])
            pred = model.predict(test_set[predictor_cols])
            end = time.time()
            score = score_model(pred, test_set[result_col])
            total_time = round(end - start, 3)
            knn_output[key] = {'score': score, 'time': total_time}

    print(knn_output)

    with open('model_results/knn_results.json', 'w') as file:
        json.dump(knn_output, file, indent=2)


def svm_evaluation(train_set, test_set, predictor_cols, result_col):
    print('\nSVM Evaluation\n')
    svm_output = dict()
    svm_output['predictor_columns'] = predictor_cols

    for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            print('C: {}, KERNEL: {}'.format(c, kernel))
            key = '{}_{}'.format(kernel, c)
            start = time.time()
            model = SVC(C=c, kernel=kernel, max_iter=800)
            model.fit(train_set[predictor_cols], train_set[result_col])
            pred = model.predict(test_set[predictor_cols])
            end = time.time()
            score = score_model(pred, test_set[result_col])
            total_time = round(end - start, 3)
            svm_output[key] = {'score': score, 'time': total_time}

    print(svm_output)

    with open('model_results/svm_results.json', 'w') as file:
        json.dump(svm_output, file, indent=2)


def nn_evaluation(train_set, test_set, predictor_cols, result_col):
    print('\nNeural Network Evaluation\n')
    nn_output = dict()
    nn_output['predictor_columns'] = predictor_cols

    for hl in [10, 15, 20, 30, 50, 100]:
        for act in ['identity', 'logistic']:
            for solv in ['lbfgs']:
                print('HL: {}, ACTIVATION: {}, SOLVER: {}'.format(hl, act, solv))
                key = '{}_{}_{}'.format(act, solv, hl)
                start = time.time()
                model = MLPClassifier(hidden_layer_sizes=hl, activation=act, solver=solv)
                model.fit(train_set[predictor_cols], train_set[result_col])
                pred = model.predict(test_set[predictor_cols])
                end = time.time()
                score = score_model(pred, test_set[result_col])
                total_time = round(end - start, 3)
                nn_output[key] = {'score': score, 'time': total_time}

    print(nn_output)

    with open('model_results/nn_results.json', 'w') as file:
        json.dump(nn_output, file, indent=2)


def rf_evaluation(train_set, test_set, predictor_cols, result_col):
    print('\nRandom Forest Evaluation\n')
    rf_output = dict()
    rf_output['predictor_columns'] = predictor_cols

    for trees in [50, 100, 150, 200]:
        for crit in ['gini', 'entropy']:
            print('TREES: {}, CRITERION: {}'.format(trees, crit))
            key = '{}_{}'.format(crit, trees)
            start = time.time()
            model = RandomForestClassifier(n_estimators=trees, criterion=crit)
            model.fit(train_set[predictor_cols], train_set[result_col])
            pred = model.predict(test_set[predictor_cols])
            end = time.time()
            score = score_model(pred, test_set[result_col])
            total_time = round(end - start, 3)
            rf_output[key] = {'score': score, 'time': total_time}

    print(rf_output)

    with open('model_results/rf_results.json', 'w') as file:
        json.dump(rf_output, file, indent=2)


def linear_evaluation(train_set, test_set, predictor_cols, result_col):
    print('\nLinear Evaluation\n')
    lin_output = dict()
    lin_output['predictor_columns'] = predictor_cols

    start = time.time()
    model = KNeighborsClassifier()
    model.fit(train_set[predictor_cols], train_set[result_col])
    pred = model.predict(test_set[predictor_cols])
    end = time.time()
    score = score_model(pred, test_set[result_col])
    total_time = round(end - start, 3)
    lin_output['output'] = {'score': score, 'time': total_time}

    print(lin_output)

    with open('model_results/linear_results.json', 'w') as file:
        json.dump(lin_output, file, indent=2)
    return lin_output


def logistic_evaluation(train_set, test_set, predictor_cols, result_col):
    print('\nLogistic Evaluation\n')
    log_output = dict()
    log_output['predictor_columns'] = predictor_cols
    for c in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for pen in ['l2', 'none']:
            print('C: {}, PEN: {}'.format(c, pen))
            key = '{}_{}'.format(c, pen)
            start = time.time()
            model = LogisticRegression(C=c, penalty=pen)
            model.fit(train_set[predictor_cols], train_set[result_col])
            pred = model.predict_proba(test_set[predictor_cols])
            end = time.time()
            score = score_model(pred, test_set[result_col])
            total_time = round(end - start, 3)
            log_output[key] = {'score': score, 'time': total_time}

    print(log_output)

    with open('model_results/logistic_results.json', 'w') as file:
        json.dump(log_output, file, indent=2)
    return log_output


def score_model(pred, act):
    tp, fn, tn, fp = 0, 0, 0, 0

    for p, a in zip(pred, act):

        try:
            p = p[1]
        except:
            pass

        if p < 0.5:
            if a == 0:
                tn += 1
            elif a == 1:
                fn += 1
        elif p >= 0.5:
            if a == 0:
                fp += 1
            elif a == 1:
                tp += 1

    acc = (tp + tn) / (tp + fp + tn + fn)
    if tp + fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = None
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = None
    if fn + fp > 0:
        fnr = fn / (fn + fp)
    else:
        fnr = None
    if not isinstance(prec, type(None)) and not isinstance(recall, type(None)):
        f1 = (2 * prec * recall) / (prec + recall)
    else:
        f1 = None
    out_dict = {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
                'accuracy': acc, 'precision': prec, 'recall': recall,
                'fnr': fnr, 'f1': f1}
    return out_dict


def security_delay_explore(df):
    out = {'airport': [], 'sec_delays': [], 'tot_flights': [], 'percent': []}
    for airport in df['ORIGIN'].unique():
        airport_df = df[df['ORIGIN'] == airport]
        tot_origins = len(airport_df)
        tot_sec_delay = len(airport_df[airport_df['SECURITY_DELAY'] > 0])
        out['airport'].append(airport)
        out['sec_delays'].append(tot_sec_delay)
        out['tot_flights'].append(tot_origins)
        out['percent'].append(round(100.*(tot_sec_delay/tot_origins), 4))

    pd.DataFrame(out).to_csv('dataset/security_delay_exploration.csv', index=False)


def sanity_check(df):
    total_flights = len(df)
    # df['RELEVANT_DELAY'] = df['LATE_AIRCRAFT_DELAY'] + df['CARRIER_DELAY']
    # df['RELEVANT_DELAY_BOOL'] = [int(i >= 15) for i in
    #                              df['RELEVANT_DELAY']]  # Delay counts with 15+ minutes
    delay_flights = len(df[df['RELEVANT_DELAY_BOOL'] == True])
    print('Total Flights: {}'.format(total_flights))
    print('Delayed Flights: {}'.format(delay_flights))

    delay_pct = 100.*(delay_flights / total_flights)
    nondelay_pct = 100.*((total_flights - delay_flights) / total_flights)

    print('Delayed %: {}'.format(delay_pct))
    print('Non-Delayed %: {}'.format(nondelay_pct))


def explore_knn_model(train_set, test_set, predictor_cols, result_col):
    train_set.dropna(axis=0, inplace=True)
    test_set.dropna(axis=0, inplace=True)

    model = KNeighborsClassifier(n_neighbors=12, algorithm='auto')
    model.fit(train_set[predictor_cols], train_set[result_col])

    pred = model.predict_proba(test_set[predictor_cols])

    x = []
    y1 = []
    y2 = []
    for whatever in range(100):
        sum_score = 0
        act_pred = []
        for p, j in zip(pred, test_set[result_col]):
            if p[0] < whatever/100.:
                v = 1
            else:
                v = 0
            sum_score += (abs(v - j))
            act_pred.append(v)
        score = round(sum_score / len(pred), 5)
        # print('Val: ', i/100.)
        # print('KNN MODEL SCORE: {}\n'.format(score))

        act_delay = []
        no_delay = []
        for i, j in zip(act_pred, test_set[result_col]):
            if int(j) == 0:
                no_delay.append(abs(int(i) - int(j)))
            else:
                act_delay.append(abs(int(i) - int(j)))

        x.append(whatever/100.)
        y1.append(round(1 - (sum(no_delay) / len(no_delay)), 5))
        y2.append(round(1 - (sum(act_delay) / len(act_delay)), 5))
    data = [
        go.Scatter(
            x=x, y=y1, mode='lines',
        ),
        go.Scatter(
            x=x, y=y2, mode='lines',
        ),
        go.Scatter(
            x=x, y=[i+j for i, j in zip(y1, y2)], mode='lines'
        )
    ]
    fig = go.Figure(data=data)
    fig.show()

        # print('Total No-Delays (Actual): {}'.format(len(no_delay)))
        # print('No-Delays (Actual) Score: {}'.format(round(1 - (sum(no_delay) / len(no_delay)), 5)))
        # print('Total Delays (Actual): {}'.format(len(act_delay)))
        # print('Delays (Actual) Score: {}'.format(round(1 - (sum(act_delay) / len(act_delay)), 5)))


def predictor_comparison(df, predictor_cols, result_col):
    fig = make_subplots(rows=2, cols=4, subplot_titles=predictor_cols)
    for i, p in enumerate(predictor_cols):
        fig.add_trace(
            go.Scatter(
                y=df[result_col].iloc[::20], x=df[p].iloc[::20], mode='markers', marker=dict(size=3)
            ),
            row=math.ceil((i + 1) / 4), col=(i % 4) + 1
        )

    fig.show()


if __name__ == '__main__':
    test_pct = 0.2

    data_fn = r'C:\Users\Paul\OneDrive - Georgia Institute of Technology\Grad_School\Courses\CSE6242\Projects\scripts\dataset/T_ONTIME_REPORTING.csv'
    df = pd.read_csv(data_fn)

    predictor_cols = ['MONTH', 'DAY_OF_WEEK', 'DAY', 'YEAR', 'OP_UNIQUE_CARRIER_CAT',
                      'ORIGIN_CAT', 'DEST_CAT', 'DISTANCE', 'DEP_TIME_BLK_CAT', 'DEP_TIME_LOC_MULT']
    # result_col = 'RELEVANT_DELAY'
    result_col = 'RELEVANT_DELAY_BOOL'

    df['RELEVANT_DELAY'] = df['LATE_AIRCRAFT_DELAY'] + df['CARRIER_DELAY']
    df['RELEVANT_DELAY_BOOL'] = [int(i >= 15) for i in
                                 df['RELEVANT_DELAY']]  # Delay counts with 15+ minutes
    df['DAY'] = [int(i.split('/')[1]) for i in df['FL_DATE']]
    df['YEAR'] = [int(i.split('/')[-1].split(' ')[0]) for i in df['FL_DATE']]
    le = LabelEncoder()
    df['ORIGIN_CAT'] = le.fit_transform(df['ORIGIN'])
    df['DEST_CAT'] = le.fit_transform(df['DEST'])
    df['DEP_TIME_BLK_CAT'] = le.fit_transform(df['DEP_TIME_BLK'])
    df['DEP_TIME_LOC_MULT'] = df['DEP_TIME_BLK_CAT'] * df['ORIGIN_CAT']
    df['OP_UNIQUE_CARRIER_CAT'] = le.fit_transform(df['OP_UNIQUE_CARRIER'])

    rel_df = df[predictor_cols + [result_col]]
    scaler = StandardScaler()
    dataset = pd.DataFrame(scaler.fit_transform(rel_df[predictor_cols]), columns=predictor_cols)

    dataset[result_col] = rel_df[result_col]
    train_set, test_set = train_test_split(dataset, test_size=test_pct)
    train_set_unscaled, test_set_unscaled = train_test_split(rel_df, test_size=test_pct)

    # KNN Evaluation
    # knn_evaluation(train_set_unscaled, test_set_unscaled, predictor_cols, result_col)
    # explore_knn_model(train_set, test_set, predictor_cols, result_col)

    # KNN Evaluation
    if not Path('model_results', 'knn_results.json').exists():
        knn_out = knn_evaluation(train_set_unscaled, test_set_unscaled, predictor_cols, result_col)
    else:
        with open('model_results/knn_results.json', 'r') as file:
            knn_out = json.load(file)
    # explore_knn_model(train_set, test_set, predictor_cols, result_col)

    # # SVM Evaluation
    # svm_evaluation(train_set, test_set, predictor_cols, result_col)

    # Linear Regression Evaluation
    if not Path('model_results', 'linear_results.json').exists():
        lin_out = linear_evaluation(train_set, test_set, predictor_cols, result_col)
    else:
        with open('model_results/linear_results.json', 'r') as file:
            lin_out = json.load(file)

    # Logistic Regression Evaluation
    if not Path('model_results', 'logistic_results.json').exists():
        log_out = logistic_evaluation(train_set, test_set, predictor_cols, result_col)
    else:
        with open('model_results/logistic_results.json', 'r') as file:
            log_out = json.load(file)

    # Random Forest Evaluation
    # rf_evaluation(train_set, test_set, predictor_cols, result_col)
    if not Path('model_results', 'rf_results.json').exists():
        rf_out = rf_evaluation(train_set, test_set, predictor_cols, result_col)
    else:
        with open('model_results/rf_results.json', 'r') as file:
            rf_out = json.load(file)

        # Neural Network Evaluation
        # nn_evaluation(train_set, test_set, predictor_cols, result_col)
    if not Path('model_results', 'nn_results.json').exists():
        nn_out = nn_evaluation(train_set, test_set, predictor_cols, result_col)
    else:
        with open('model_results/nn_results.json', 'r') as file:
            nn_out = json.load(file)

    score_compare = dict(model=[], tp=[], fn=[], tn=[], fp=[], accuracy=[], precision=[], recall=[], fnr=[], f1=[])

    for k, v in knn_out.items():
        if 'predictor' in k:
            continue
        score_compare['model'].append('KNN_{}'.format(k))
        score_compare['tp'].append(v['score']['tp'])
        score_compare['fn'].append(v['score']['fn'])
        score_compare['tn'].append(v['score']['tn'])
        score_compare['fp'].append(v['score']['fp'])
        score_compare['accuracy'].append(v['score']['accuracy'])
        score_compare['precision'].append(v['score']['precision'])
        score_compare['recall'].append(v['score']['recall'])
        score_compare['fnr'].append(v['score']['fnr'])
        score_compare['f1'].append(v['score']['f1'])

    for k, v in lin_out.items():
        if 'predictor' in k:
            continue
        score_compare['model'].append('LIN_{}'.format(k))
        score_compare['tp'].append(v['score']['tp'])
        score_compare['fn'].append(v['score']['fn'])
        score_compare['tn'].append(v['score']['tn'])
        score_compare['fp'].append(v['score']['fp'])
        score_compare['accuracy'].append(v['score']['accuracy'])
        score_compare['precision'].append(v['score']['precision'])
        score_compare['recall'].append(v['score']['recall'])
        score_compare['fnr'].append(v['score']['fnr'])
        score_compare['f1'].append(v['score']['f1'])

    for k, v in log_out.items():
        if 'predictor' in k:
            continue
        score_compare['model'].append('LOG_{}'.format(k))
        score_compare['tp'].append(v['score']['tp'])
        score_compare['fn'].append(v['score']['fn'])
        score_compare['tn'].append(v['score']['tn'])
        score_compare['fp'].append(v['score']['fp'])
        score_compare['accuracy'].append(v['score']['accuracy'])
        score_compare['precision'].append(v['score']['precision'])
        score_compare['recall'].append(v['score']['recall'])
        score_compare['fnr'].append(v['score']['fnr'])
        score_compare['f1'].append(v['score']['f1'])

    for k, v in rf_out.items():
        if 'predictor' in k:
            continue
        score_compare['model'].append('RF_{}'.format(k))
        score_compare['tp'].append(v['score']['tp'])
        score_compare['fn'].append(v['score']['fn'])
        score_compare['tn'].append(v['score']['tn'])
        score_compare['fp'].append(v['score']['fp'])
        score_compare['accuracy'].append(v['score']['accuracy'])
        score_compare['precision'].append(v['score']['precision'])
        score_compare['recall'].append(v['score']['recall'])
        score_compare['fnr'].append(v['score']['fnr'])
        score_compare['f1'].append(v['score']['f1'])

    for k, v in nn_out.items():
        if 'predictor' in k:
            continue
        score_compare['model'].append('NN_{}'.format(k))
        score_compare['tp'].append(v['score']['tp'])
        score_compare['fn'].append(v['score']['fn'])
        score_compare['tn'].append(v['score']['tn'])
        score_compare['fp'].append(v['score']['fp'])
        score_compare['accuracy'].append(v['score']['accuracy'])
        score_compare['precision'].append(v['score']['precision'])
        score_compare['recall'].append(v['score']['recall'])
        score_compare['fnr'].append(v['score']['fnr'])
        score_compare['f1'].append(v['score']['f1'])

    out_df = pd.DataFrame(score_compare)
    out_df.to_csv('model_results/overall_score_comparison.csv')
    # Security Delay Exploration
    # security_delay_explore(df)
    # sanity_check(test_set)

    # predictor_comparison(rel_df, ['DAY_OF_WEEK', 'OP_UNIQUE_CARRIER_CAT', 'ORIGIN_CAT', 'DEST_CAT',
    #                               'DISTANCE', 'DEP_TIME_BLK_CAT', 'DEP_TIME_LOC_MULT'], result_col)


