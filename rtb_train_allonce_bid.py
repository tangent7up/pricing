# %pyspark

from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType,StringType,IntegerType
from pyspark.mllib.util import MLUtils
from pyspark.ml.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
from scipy.sparse.csr import csr_matrix
from scipy.sparse import vstack,hstack
from scipy.stats import norm

import os
import pickle
import datetime
import time
import json
from utils import *
from conf_train import *


def load_dict_df(path):
    try:
        d = spark.read.parquet(path)
        d = dict(d.rdd.map(lambda x: (x['_1'],x['_2'])).collect())
    except:
        d = dict()
    return d

# 行转换
def RTF(t,columns_dict, columns_dict_len, add = []):
    """
    生成一行libsvm格式的字符串
    """
    target=t
    def rowTransform(row):
        new_row = {}
        tar = None
        ind=0
        for column in catagory:
            index = columns_dict[column].get(row[column],0)
            if index != 0: new_row[ind + index-1] = 1
            ind += columns_dict_len[column]
        for column in quality:
            words=row[column]
            if words is not None:
                for w in words:
                    index=columns_dict[column].get(w,None)
                    if index != None:
                        new_row[ind + index] = 1
            ind += columns_dict_len[column]
        keys = sorted(new_row.keys())
        for column in target:
            keys.append(row[column])
        for a in add:
            keys.append(row[a])
        return keys
    return rowTransform


def parse_feature_data(onerow):
    try:
        result=[]
        for eachcol in catagory+quality:
            if eachcol in catagory:
                value=onerow[eachcol]
                if value != None:
                    result.append(eachcol + '_:_' + myStr(value))
            if eachcol in quality:
                value=onerow[eachcol]
                if value != [] and value is not None:
                    for v in value:
                        result.append(eachcol + '_:_' + v)
        return result
    except:
        return None

# 一些字段转换
def field_transf(r):
    try:
        r_=r.asDict()
        r_['osv']=r['osv'].split('.')[0]
        r_['model']=r['model'].split(',')[0]
        r_['data']=[i['value'] for i in r['data'][-1]['segment']]
        a=datetime.datetime.strptime(r['drt'],'%Y-%m-%d+%H:%M:%S')
        r_['minute']=str(a.minute)
        r_['weekday']=str(a.weekday())
        r_['time_zone']=str(a.hour)
        r_.pop('drt')
        return Row(**r_)
    except:
        return None

def myStr(string):
    if type(string)==unicode:
        return string.encode('utf-8')
    else:
        return str(string)
## path 在本地可以跑，但是在s3，不知道可不可以。

def load_dict(path,name):
    try:
        with open(path + name,'r') as file:
            dict_old = pickle.load(file)
        assert(type(dict_old) == dict)
    except:
        dict_old = dict()
    return dict_old


auc_dict = load_dict(path,'models/auc_dict')

bid_col=['drt', 'ad.adid', 'ad.axid', 'paytype', 'product',
'serv', 'ad.ad_pkg', 'ad.dspchn', 'app.app_pkg',
'app.name', 'app.adxslotid', 'app.slotid', 'app.adplace_width', 'app.adplace_height',
'app.rtbchn', 'app.app', 'app.dsslotid', 'network.carrier',
'network.nettype', 'geo.country', 'geo.regionc', 'geo.cityc', 'geo.lat',
'geo.lon', 'device.brand', 'device.model', 'device.osv',
'device.screen_width', 'device.screen_height', 'device.dv',
'user.yob', 'user.gender', 'user.edu', 'user.spec_usrtag', 'user.data',
'content.keyword', 'content.cclabel', 'content.cat1', 'content.cat2', 'content.author',
'content.cont_id', 'content.program', 'bid.bidprice']

win_col=['click_active' ,'has_active', 'has_click', 'drt', 'ad.adid', 'ad.axid', 'paytype', 'product',
'serv', 'ad.ad_pkg', 'ad.dspchn', 'app.app_pkg',
'app.name', 'app.adxslotid', 'app.slotid', 'app.adplace_width', 'app.adplace_height',
'app.rtbchn', 'app.app', 'app.dsslotid', 'network.carrier',
'network.nettype', 'geo.country', 'geo.regionc', 'geo.cityc', 'geo.lat',
'geo.lon', 'device.brand', 'device.model', 'device.osv',
'device.screen_width', 'device.screen_height', 'device.dv',
'user.yob', 'user.gender', 'user.edu', 'user.spec_usrtag', 'user.data',
'content.keyword', 'content.cclabel', 'content.cat1', 'content.cat2', 'content.author',
'content.cont_id', 'content.program', 'bid.bidprice', 'bid.winprice']



df1 = spark.read.parquet(bid_path)
### adid 对应的 bidprice 分布检测
# df1.createOrReplaceTempView('df1')
# spark.sql('select ad.adid as adid, ad.axid as axid, mean(bid.bidprice) as bidprice from df1 group by ad.adid, ad.axid order by ad.adid').show(100)
df2 = spark.read.parquet(bid_session_path)

df1 = df1.where(df1.ad.axid.isin(ad)).select(df1.ad.paytype.alias("ad_paytype"),*bid_col)
# df2 = df2.withColumn('click_active', when((df2.has_click == 0) & (df2.has_active == 1), 0).otherwise(df2.has_active))
df2_ = df2.select(df2.ad.paytype.alias("ad_paytype"),*bid_col)

df_bid = df1.union(df2_)
df_bid = df_bid.rdd.map(field_transf).filter(lambda x: x is not None).toDF().cache()
count = df_bid.count()
if count > 5000000:
    df_bid = df_bid.sample(False, 4000000.0/count, seed = 0).count()

catagory=['weekday','time_zone','minute','paytype', 'product',
'serv', 'ad.ad_pkg', 'ad.dspchn', 'app.app_pkg',
'app.name', 'app.adxslotid', 'app.slotid', 'app.adplace_width', 'app.adplace_height',
'app.rtbchn', 'app.app', 'app.dsslotid', 'network.carrier',
'network.nettype', 'geo.country', 'geo.regionc', 'geo.cityc', 'geo.lat',
'geo.lon', 'device.brand', 'device.model', 'device.osv',
'device.screen_width', 'device.screen_height', 'device.dv',
'user.yob', 'user.gender', 'user.edu',
'content.cat1', 'content.cat2', 'content.author',
'content.cont_id', 'content.program']
quality=['user.spec_usrtag', 'user.data', 'content.keyword', 'content.cclabel']

catagory=[i.split(".")[-1] for i in catagory]
quality=[i.split(".")[-1] for i in quality]

print("得出所有field的可能值")
time_a = time.time()
# 得出所有field的可能值，使用 field_:_value 的方式存放。
new_columns=[]
columns_dict = {}
distinct_words=df_bid.select(catagory+quality).rdd.flatMap(parse_feature_data).filter(lambda x: x is not None).distinct().collect()
for column in catagory+quality:
    columns_dict[column]=[]
    columns_dict[column] += [word.split("_:_")[1] for word in distinct_words if word.split("_:_")[0]==column]
print 'time: ', time.time()-time_a


print("生成onehot后新的列")
time_a = time.time()
# new_columns是每个field的字段值组成的新feature
# columns_dict 是每个field的onehot字典
# columns_dict_len 是每个field的不同值数
columns_dict_len={}
for column in catagory:
    unique=columns_dict[column]
    columns_dict[column]=dict(zip(unique,range(1,1+len(unique))))
    columns_dict_len[column]=len(unique)
    new_columns+=[column+"_:_"+myStr(c) for c in unique]
for column in quality:
    unique=[i for i in list(set(columns_dict[column])) if i!=""]
    columns_dict[column]=dict(zip(unique,range(len(unique))))
    columns_dict_len[column]=len(unique)
    new_columns+=[column+"_:_"+myStr(c) for c in unique]
print 'time: ', time.time()-time_a


print("生成sparse matrix")
time_a = time.time()
sparse=df_bid.rdd.map(RTF(['bidprice'], columns_dict, columns_dict_len)).filter(lambda x: x is not None).collect()
y_bid = [row[-1] for row in sparse]
cols = [row[:-1] for row in sparse]
l = len(sparse)
del(df1,df2,df2_,df_bid)
del(sparse)
rows = [[num]*len(row) for num,row in enumerate(cols)]
cols = [i for a in cols for i in a]
rows = [i for a in rows for i in a]
data = [1] * len(cols)
X_bid = csr_matrix((data, (rows, cols)), shape=(l, len(new_columns)))
del(cols,rows,data)

bid_columns_information = dict()
bid_columns_information['time'] = time.time()
bid_columns_information['new_columns'] = new_columns
bid_columns_information['columns_dict_len'] = columns_dict_len
bid_columns_information['columns_dict'] = columns_dict
# encode_json = json.dumps(columns_information)
# spark.createDataFrame([(encode_json,)]).coalesce(1).write.mode("overwrite").text(path + 'models/'+str(axid)+'/columns_information')
print 'time: ', time.time()-time_a

print('训练bidprice模型')
time_a = time.time()
X_train = X_bid
X_train[:,-1] = X_train[:,-1].todense()
y_train = np.array(y_bid)

#winprice 'bidprice','has_active','click_active'
xgblinear = xgb.XGBRegressor(nthread=6, objective = 'reg:linear', learning_rate=0.08,n_estimators=100,max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
xgblinear.fit(X_train, y_train)
print 'time: ', time.time()-time_a

try:
    os.mkdir('models')
except:
    pass
with open(path + 'models/bid_columns_information','w') as file:
    pickle.dump(bid_columns_information,file)
with open(path + 'models/bid_model','w') as file:
    pickle.dump(xgblinear,file)

try:
    os.mkdir('constants')
except:
    pass

bid_constant_correct = {114333:1}
with open(path + 'constants/bid_constant_correct', 'w') as file:
    json.dump(bid_constant_correct,file)

model_match = {114333:42256}
with open(path + 'constants/model_match', 'w') as file:
    json.dump(model_match,file)
