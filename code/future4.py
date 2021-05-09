import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBRegressor
from xgboost import XGBRegressor
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from keras.utils import np_utils


df_train = pd.read_csv('./input/sales_train.csv')
df_shops = pd.read_csv("../input/shops.csv")
df_items = pd.read_csv("../input/items.csv")
df_item_categories = pd.read_csv("../input/item_categories.csv")
df_test = pd.read_csv('../input/test.csv')


df_train.head()

df_train.info()

df_train.describe()
df_train.isnull().sum()
df_test.isna().sum()

import pandas_profiling as pp
pp.ProfileReport(df_train)


df_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)

df_train.head()

df_train['date'] = pd.to_datetime(df_train['date'], dayfirst=True)
df_train['date'] = df_train['date'].apply(lambda x: x.s('%Y-%m'))

df = df_train.groupby(['date', 'shop_id','item_id']).sum()
df = df.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day',fill_value=0)
df.reset_index(inplace=True)
df.head().T


df_test = pd.merge(df_test, on=['shop_id', 'item_id'], how='left')
df_test.drop(["ID", '2013-01'], axis=1, inplace=True)
