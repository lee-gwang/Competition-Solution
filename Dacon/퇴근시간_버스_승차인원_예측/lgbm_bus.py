# Library 예시
import os
# handling
import pandas as pd
import numpy as np
import random
import gc
import tqdm
# visualization
#import matplotlib.pyplot as plt
# prevent overfit
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# model
import lightgbm as lgb
from dacon_function import dacon_processing as dp
SEED=42


###########################
# BASIC SETTING
###########################
dp.seed_everything(SEED)
TARGET = '18~20_ride'

###########################
# DATA LOAD
###########################
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
sub = pd.read_csv('./data/submission_sample.csv')

print('data load complete')

###########################
# Final features list
###########################
excluded_features = ['id', 'date', 'in_out', TARGET,'day', 'week', 'weekday', 'lat_long']

ride_take = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff',
             '6~8_ride', '6~9_ride', '6~10_ride', '6~8_takeoff', '6~9_takeoff', '6~10_takeoff']

tr, te = dp.base_preprocessing(train_df, test_df)
tr, te = dp.lat_long_create(tr, te)
tr, te = dp.feature_combine(tr, te)

tr, te = dp.day_agg(tr, te, merge_columns=['day'], columns=ride_take, aggs=['mean'])
tr, te = dp.sub_day_agg(tr, te, merge_columns=['bus_route_id', 'station_code', 'station_lat_long'], date_columns=['day'], columns=ride_take, aggs=['mean'])
tr, te = dp.sub_day_agg(tr, te, merge_columns=['bus_route_id', 'station_code', 'station_name', 'station_lat_long'], date_columns=['day'], columns=ride_take, aggs=['quantile'])

###########################
# transform categorical
###########################

category_features = ['bus_route_id', 'station_code', 'station_name', 'station_name2', 'station_lat_long', 'station_lat_long2', 'bus_route_id_station_code',
                     'bus_route_id_station_lat_long']
tr, te = dp.frequency_encoding(tr, te, category_features)

###########################
# Model
###########################
def make_predictions(model, tr_df, tt_df, features_columns, target, params, category_feature=[''], NFOLDS=4,
                     oof_save=False, clip=999, SEED=SEED):
    X, y = tr_df[features_columns], tr_df[target]
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(tr_df))
    pred = np.zeros(len(tt_df))
    fi_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(kf.split(X)):
        print('Fold:', fold_)
        tr_data = lgb.Dataset(X.loc[trn_idx], label=y[trn_idx].clip(0, clip))
        vl_data = lgb.Dataset(X.loc[val_idx], label=y[val_idx])
        if model == 'lgb':
            estimator = lgb.train(params, tr_data, valid_sets=[tr_data, vl_data], verbose_eval=500)
            fi_df = pd.concat([fi_df, pd.DataFrame(sorted(zip(estimator.feature_importance(), features_columns)),
                                                   columns=['Value', 'Feature'])])

        oof[val_idx] = estimator.predict(X.loc[val_idx])
        pred += estimator.predict(tt_df[features_columns]) / NFOLDS
        del estimator
        gc.collect()

    oof = np.where(oof > 0, oof, 0)
    pred = np.where(pred > 0, pred, 0)

    if oof_save:
        if model == 'lgb':
            np.save('./submission/oof_lgb.npy', oof)
            np.save('./submission/pred_lgb.npy', pred)
        elif model == 'cat':
            np.save('./submission/oof_cat.npy', oof)
            np.save('./submission/pred_cat.npy', pred)

    tt_df[target] = pred
    print('OOF RMSE:', rmse(y, oof))

    try:
        fi_df = fi_df.groupby('Feature').mean().reset_index().sort_values('Value')
    except:
        pass

    return tt_df[['id', target]], fi_df
#########
lgb_params = {
        'objective':'regression',
        'boosting_type':'gbdt',
        'metric':'rmse',
        'n_jobs':-1,
        'learning_rate':0.003,
        'num_leaves': 700,
        'max_depth':-1,
        'min_child_weight':5,
        'colsample_bytree': 0.3,
        'subsample':0.7,
        'n_estimators':50000,
        'gamma':0,
        'reg_lambda':0.05,
        'reg_alpha':0.05,
        'verbose':-1,
        'seed': SEED,
        'early_stopping_rounds':50
    }
tr, te = dp.remove_outlier(tr, te, category_features)
tr, te = dp.category_transform(tr, te, category_features)

features_columns = [col for col in tr.columns if col not in excluded_features]
test_predictions, fi = make_predictions('lgb', tr, te, features_columns, TARGET, lgb_params, category_feature=category_features, NFOLDS=5, oof_save=True)

###########################
# submission
###########################
test_predictions.to_csv('./submission/lgb_model.csv', index=False)





