import pandas as pd
import numpy as np
import random
import gc
import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

"""
Useful function
"""

class dacon_processing():

    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    def rmse(y_true, y_pred):
        return np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)


    def df_copy(tr_df, te_df):
        tr = tr_df.copy();
        te = te_df.copy()
        return tr, te


    def base_preprocessing(tr_df, te_df):

        tr, te = df_copy(tr_df, te_df)
        """
        like category reduction
        """
        tr['bus_route_id'] = tr['bus_route_id'].apply(lambda x: str(x)[:-4]).astype(int) # 클러스터링 개념으로 생각하면될듯, 카테고리좀 줄인것
        te['bus_route_id'] = te['bus_route_id'].apply(lambda x: str(x)[:-4]).astype(int)
        tr['station_name2'] = tr['station_name'].apply(lambda x: str(x)[:2]) # 정류장명 앞부분 두글자만 사용해서 라벨링
        te['station_name2'] = te['station_name'].apply(lambda x: str(x)[:2])
        tr['station_name'] = tr['station_name'].apply(lambda x: x.replace(' ', '')) # 띄어쓰기 없애고 정류장명 라벨링
        te['station_name'] = te['station_name'].apply(lambda x: x.replace(' ', ''))

        le = LabelEncoder().fit(pd.concat([tr['station_name'], te['station_name']]))
        le2 = LabelEncoder().fit(pd.concat([tr['station_name2'], te['station_name2']]))

        for df in [tr, te]:
            """
            date feature
            """
            df['day'] = pd.to_datetime(df['date']).dt.day
            df['week'] = pd.to_datetime(df['date']).dt.week
            df['weekday'] = pd.to_datetime(df['date']).dt.weekday

            """
            label encoding
            """
            df['station_name'] = le.transform(df['station_name'])
            df['station_name2'] = le2.transform(df['station_name2'])

            """
            feature binning(cv 상당히 올랐지만 결국 나중 오버피팅)
            """
            df['6~8_ride'] = df[['6~7_ride', '7~8_ride']].sum(1)
            df['6~9_ride'] = df[['6~7_ride', '7~8_ride', '8~9_ride']].sum(1)
            df['6~10_ride'] = df[['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride']].sum(1)
            df['6~8_takeoff'] = df[['6~7_takeoff', '7~8_takeoff']].sum(1)
            df['6~9_takeoff'] = df[['6~7_takeoff', '7~8_takeoff', '8~9_takeoff']].sum(1)
            df['6~10_takeoff'] = df[['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff']].sum(1)
        te['day'] = te['day'] + 30 # train과 test의 day가 다르기에(각 9월, 10월) 서로 다르게 day feature를 입력

        return tr, te


    def lat_long_create(tr_df, te_df):
        """
        비슷한 위도 경도에 있는(반올림을통해) 정류장들 묶어주기위함
       """
        tr, te = df_copy(tr_df, te_df)
        tr['lat_long'] = np.round(tr['latitude'], 2).astype(str) + np.round(tr['longitude'], 2).astype(str)
        te['lat_long'] = np.round(te['latitude'], 2).astype(str) + np.round(te['longitude'], 2).astype(str)
        le = LabelEncoder().fit(pd.concat([tr['lat_long'], te['lat_long']]))
        tr['station_lat_long'] = le.transform(tr['lat_long'])
        te['station_lat_long'] = le.transform(te['lat_long'])

        tr['lat_long'] = np.round(tr['latitude'], 3).astype(str) + np.round(tr['longitude'], 2).astype(str)
        te['lat_long'] = np.round(te['latitude'], 3).astype(str) + np.round(te['longitude'], 2).astype(str)
        le = LabelEncoder().fit(pd.concat([tr['lat_long'], te['lat_long']]))
        tr['station_lat_long2'] = le.transform(tr['lat_long'])
        te['station_lat_long2'] = le.transform(te['lat_long'])

        return tr, te


    def feature_combine(tr_df, te_df):
        """
        .astype(str) +'_' + .astype(str) categorical feature
        """
        tr, te = df_copy(tr_df, te_df)
        for df in [tr, te]:
            df['bus_route_id_station_code'] = ((df['bus_route_id']).astype(str) + (df['station_code']).astype(str)).astype(
                'category')
            df['bus_route_id_station_lat_long'] = ((df['bus_route_id']).astype(str) + (df['station_lat_long']).astype(str)).astype('category')

        return tr, te


    def category_transform(tr_df, te_df, columns):
        """
        transform categorical feature for lgbm model
        """
        tr, te = df_copy(tr_df, te_df)
        for df in [tr, te]:
            df[columns] = df[columns].astype(str).astype('category')

        return tr, te


    def frequency_encoding(tr_df, te_df, columns, normalize=False):
        """
        count categorical feature
        """
        tr, te = df_copy(tr_df, te_df)
        for col in columns:
            if not normalize:
                freq_encode = pd.concat([tr[col], te[col]]).value_counts()
                tr[col + '_fq_enc'] = tr[col].map(freq_encode)
                te[col + '_fq_enc'] = te[col].map(freq_encode)
            else:
                freq_encode = pd.concat([tr[col], te[col]]).value_counts(normalize=True)
                tr[col + '_fq_enc_nor'] = tr[col].map(freq_encode)
                te[col + '_fq_enc_nor'] = te[col].map(freq_encode)
        return tr, te


    def remove_outlier(tr_df, te_df, columns):
        """
        train과 test 서로 없는 피쳐는 지우려는 것것
        """
        tr te = df_copy(tr_df, te_df)
        for col in columns:
            tr[col] = np.where(tr[col].isin(te[col]), tr[col], 0)
            te[col] = np.where(te[col].isin(tr[col]), te[col], 0)
        return tr, te


    def day_agg(tr_df, te_df, merge_columns, columns, aggs=['mean']):
        """
        aggregation feature 1
        """
        tr, te = df_copy(tr_df, te_df)
        for merge_column in merge_columns:
            for col in columns:
                for agg in aggs:
                    valid = pd.concat([tr[[merge_column, col]], te[[merge_column, col]]])
                    new_cn = merge_column + '_' + agg + '_' + col
                    if agg == 'quantile':
                        valid = valid.groupby(merge_column)[col].quantile(0.8).reset_index().rename(columns={col: new_cn})
                    else:
                        valid = valid.groupby(merge_column)[col].agg([agg]).reset_index().rename(columns={agg: new_cn})
                    valid.index = valid[merge_column].tolist()
                    valid = valid[new_cn].to_dict()

                    tr[new_cn] = tr[merge_column].map(valid)
                    te[new_cn] = te[merge_column].map(valid)
        return tr, te


    def sub_day_agg(tr_df, te_df, merge_columns, date_columns, columns, aggs=['mean']):
        """
        aggregation feature 2, after combine feature
        """
        tr, te = df_copy(tr_df, te_df)
        for merge_column in merge_columns:
            for date in date_columns:
                tr['mc_date'] = tr[merge_column].astype(str) + '_' + tr[date].astype(str)
                te['mc_date'] = te[merge_column].astype(str) + '_' + te[date].astype(str)
                for col in columns:
                    for agg in aggs:
                        valid = pd.concat([tr[['mc_date', col]], te[['mc_date', col]]])
                        new_cn = merge_column + '_' + date + '_' + col + '_' + agg
                        if agg == 'quantile':
                            valid = valid.groupby('mc_date')[col].quantile(0.8).reset_index().rename(columns={col: new_cn})
                        else:
                            valid = valid.groupby('mc_date')[col].agg([agg]).reset_index().rename(columns={agg: new_cn})
                        valid.index = valid['mc_date'].tolist()
                        valid = valid[new_cn].to_dict()

                        tr[new_cn] = tr['mc_date'].map(valid)
                        te[new_cn] = te['mc_date'].map(valid)
        tr = tr.drop(columns=['mc_date'])
        te = te.drop(columns=['mc_date'])
        return tr, te
