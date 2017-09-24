# -*- coding: utf-8 -*-

from pyspark import SparkContext
sc = SparkContext()
from pyspark.sql import SQLContext
spark = SQLContext(sc)

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
    result=[]
    for eachcol in catagory+quality:
        if eachcol in catagory:
            value=onerow[eachcol]
            if value != None:
                if onerow["has_active"] == 1:
                    result += [eachcol + '_:_' + myStr(value)]*3
                if onerow["click_active"] == 1:
                    result += [eachcol + '_:_' + myStr(value)]*6
                result.append(eachcol + '_:_' + myStr(value))
        if eachcol in quality:
            value=onerow[eachcol]
            if value != [] and value is not None:
                for v in value:
                    if onerow["has_active"] == 1:
                        result += [eachcol + '_:_' + v]*3
                    if onerow["click_active"] == 1:
                        result += [eachcol + '_:_' + v]*6
                    result.append(eachcol + '_:_' + v)
    return result

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

# session_dict 记录每个广告 session条目数
# has_click_dict_old 记录每个广告 has_click 条目数
# last_train_time 记录 每个广告上次训练的时间
session_dict_old = load_dict_df(path+'models/session_dict')
has_click_dict_old = load_dict_df(path+'models/has_click_dict')
last_train_time = load_dict_df(path+'models/last_train_time')
axid_dict = load_dict_df(path+'models/axid_dict_df')
adid_dict = load_dict_df(path+'models/adid_dict_df')


df2 = spark.read.parquet(session_path)
df3 = spark.read.parquet(session_old_path)
df2.createOrReplaceTempView("df")
tmp = spark.sql("select ad.axid,count(1) as num from df where has_active = 1 group by ad.axid order by num desc")
has_click_dict = dict([(i['axid'],i['num']) for i in tmp.rdd.collect()])
tmp = spark.sql("select ad.axid,count(1) as num from df group by ad.axid order by num desc")
session_dict = dict([(i['axid'],i['num']) for i in tmp.rdd.collect()])
tmp = spark.sql("select distinct ad.axid,ad.adid from df where ad.adid != 0 order by adid")
id_tuples = [(row["axid"],row["adid"]) for row in tmp.rdd.collect()]

# axid 和 adid的互相查询字典 更新。
for tup in id_tuples:
    axid_dict[tup[0]] = tup[1]
    oldv = adid_dict.get(tup[1],[])
    oldv.append(tup[0])
    adid_dict[tup[1]] = oldv

if auto_search_ad:
    ad = [tup[0] for tup in has_click_dict.iteritems() if tup[1]>50]

auc_dict = load_dict(path,'models/auc_dict')

if renew_bid_constand:
    # bid_constant_dict  = load_dict(path,'models/bid_constant_dict')
    with open(path + 'models/bid_model','r') as file:
        xgblinear = pickle.load(file)
    with open(path + 'models/bid_columns_information','r') as file:
        bid_columns_information = pickle.load(file)
    bid_columns_dict_len = bid_columns_information['columns_dict_len']
    bid_columns_dict = bid_columns_information['columns_dict']
    bid_new_columns = bid_columns_information['new_columns']

ad = [i for i in ad if session_dict.get(i,0) > 0]
# 只有session数量增加了5%，或者12小时没训练过且点击有增加，才会进行训练，选出这样的广告 ad_to_train
ad_to_train = [i for i in ad if (float(session_dict[i])-float(session_dict_old.get(i,0)))/float(session_dict[i])>0.05 or\
    (time.time()-last_train_time.get(i,0) > 3600*12 and has_click_dict[i] > has_click_dict_old.get(i,0))]

# 更新旧的记录,并储存
for i in ad_to_train:
    session_dict_old[i] = session_dict[i]
    has_click_dict_old[i] = has_click_dict[i]
    last_train_time[i] = time.time()

spark.createDataFrame(session_dict_old.items()).coalesce(1).write.mode("overwrite").format("parquet").save(path +'models/session_dict')
spark.createDataFrame(has_click_dict_old.items()).coalesce(1).write.mode("overwrite").format("parquet").save(path +'models/has_click_dict')
spark.createDataFrame(last_train_time.items()).coalesce(1).write.mode("overwrite").format("parquet").save(path +'models/last_train_time')
spark.createDataFrame(adid_dict.items()).coalesce(1).write.mode("overwrite").format("parquet").save(path +'models/adid_dict_df')
spark.createDataFrame(axid_dict.items()).coalesce(1).write.mode("overwrite").format("parquet").save(path +'models/axid_dict_df')
try:
    spark.createDataFrame([(str(i),) for i in ad_to_train]).coalesce(1).write.mode("overwrite").text(path + 'models/ad_to_train')
except:
    pass
    #空的时候会报错


df2 = df2.where(df2.ad.axid.isin(ad))
df2 = df2.withColumn('click_active', when((df2.has_click == 0) & (df2.has_active == 1), 0).otherwise(df2.has_active))
df2.createOrReplaceTempView("df")
tmp = spark.sql("select ad.axid,count(1) as num from df where click_active = 1 group by ad.axid order by num desc")
click_active_dict = dict([(i['axid'],i['num']) for i in tmp.rdd.collect()])



# df2_ = df2.select(df2.ad.paytype.alias("ad_paytype"),*bid_col)
df3 = df3.filter(df3.has_active==1)
df3 = df3.withColumn('click_active', when((df3.has_click == 0) & (df3.has_active == 1), 0).otherwise(df3.has_active))


df_win_1 = df2.select(df2.ad.paytype.alias("ad_paytype"),*win_col)
df_win_2 = df3.select(df3.ad.paytype.alias("ad_paytype"),*win_col)
df_win = df_win_2.union(df_win_1)
df_win = df_win.rdd.map(field_transf).filter(lambda x: x is not None).toDF().cache()

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

for axid in ad_to_train:
    print 'axid: ',axid
    # df_bid_ = df_bid.filter(df_bid.axid == axid)
    df_win_ = df_win.filter(df_win.axid == axid).cache()
    count = df_win_.count()
    if count > 250000:
        df_win_ = df_win_.sampleBy("has_active", fractions={0: 200000./count, 1: 1.0}, seed=0)
        sample_percentage = 200000./count
    else:
        sample_percentage = 1.

    print("得出所有field的可能值")
    time_a = time.time()
    # 得出所有field的可能值，使用 field_:_value 的方式存放。
    new_columns=[]
    columns_dict = {}
    # distinct_words=df_win_.select(catagory+quality).rdd.flatMap(parse_feature_data).distinct().collect()
    distinct_words=df_win_.rdd.flatMap(parse_feature_data).map(lambda x: (x,1)).countByKey()
    distinct_words = sorted(distinct_words.iteritems(),key = lambda x: x[1],reverse = True)
    distinct_words = distinct_words[:20000]
    print "截断位置热度:", distinct_words[-1][1]
    distinct_words = [i[0] for i in distinct_words]
    for column in catagory+quality:
        columns_dict[column]=[]
        columns_dict[column] += [word.split("_:_")[1] for word in distinct_words if word.split("_:_")[0]==column]
    print 'time: ', time.time()-time_a, '\n'

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
    print 'time: ', time.time()-time_a, '\n'

    print("生成sparse matrix")
    time_a = time.time()
    sparse=df_win_.rdd.map(RTF(['winprice'],columns_dict, columns_dict_len, ['bidprice','has_active','click_active'])).collect()
    y_win = [row[-4:] for row in sparse]
    y_win = np.array(y_win)
    cols = [row[:-4] for row in sparse]
    rows = [[num]*len(row) for num,row in enumerate(cols)]
    cols = [i for a in cols for i in a]
    rows = [i for a in rows for i in a]
    data = [1] * len(cols)
    X_win = csr_matrix((data, (rows, cols)), shape=(len(sparse), len(new_columns)))
    del(rows,cols,data)

    columns_information = dict()
    columns_information['time'] = time.time()
    columns_information['new_columns'] = new_columns
    columns_information['columns_dict_len'] = columns_dict_len
    columns_information['columns_dict'] = columns_dict
    # encode_json = json.dumps(columns_information)
    # spark.createDataFrame([(encode_json,)]).coalesce(1).write.mode("overwrite").text(path + 'models/'+str(axid)+'/columns_information')
    print 'time: ', time.time()-time_a, '\n'

    if renew_bid_constand:
        print('训练bidprice模型')
        sparse=df_win_.rdd.map(RTF(['bidprice'],bid_columns_dict, bid_columns_dict_len)).collect()
        y_bid = [row[-1] for row in sparse]
        cols = [row[:-1] for row in sparse]
        rows = [[num]*len(row) for num,row in enumerate(cols)]
        cols = [i for a in cols for i in a]
        rows = [i for a in rows for i in a]
        data = [1] * len(cols)
        X_bid = csr_matrix((data, (rows, cols)), shape=(len(sparse), len(bid_new_columns)))

        time_a = time.time()
        X_train = X_bid
        X_train[:,-1] = X_train[:,-1].todense()
        y_train = y_bid

        with open(path + 'models/bid_model','r') as file:
            xgblinear = pickle.load(file)

        #winprice 'bidprice','has_active','click_active'
        y_pred_win_train = xgblinear.predict(X_train)
        bid_average = np.average(y_pred_win_train)
        print 'time: ', time.time()-time_a, '\n'

    print('训练cvr模型')

    shape = X_win.shape
    print 'X_train.shape: ', shape
    print 'np.sum(y_win[:,-1]): ', np.sum(y_win[:,-1])

    time_a = time.time()
    X_train = X_win
    X_train[:,-1]=X_train[:,-1].todense()

    y_train = y_win[:,-2:]
    X, y = my_DataSplit(X_train, y_train,2)
    X[1][:,-1] = X[1][:,-1].todense()
    X[0][:,-1] = X[0][:,-1].todense()
    gbdt_lr_test = GBDT_LR()
    gbdt_lr_test.fit(X[0], np.array(y[0][:,0]).flatten(), False)
    y_pred_test_cvr = gbdt_lr_test.predict(X[1])
    try:
        test_auc = roc_auc_score(np.array(y[1][:,1]).flatten(), y_pred_test_cvr)/2
    except:
        test_auc = 0.25

    gbdt_lr_test.fit(X[1], np.array(y[1][:,0]).flatten(), False)
    y_pred_test_cvr = gbdt_lr_test.predict(X[0])
    try:
        test_auc += roc_auc_score(np.array(y[0][:,1]).flatten(), y_pred_test_cvr)/2
    except:
        test_auc += 0.25
    print 'test_auc', test_auc

    auc_dict[axid] = test_auc

    y_train = np.array(y_win[:,-2]).flatten()

    y_train_add = np.array(y_win[:,-1]).flatten()  # 接下来五行用来调整权重
    X_train_add = vstack(list(X_win[y_train_add ==1])*5)
    y_train_add = np.ones(int(np.sum(y_train_add)*5))
    y_train = np.concatenate((y_train,y_train_add))
    X_train = vstack((X_train, X_train_add))

    # y_train_add = np.array(X_win[:,-2].todense()).flatten()  # 接下来五行用来调整权重
    # X_train_add = vstack(list(X_win[:,:-3][y_train_add ==1])*9)
    # y_train_add = np.ones(int(np.sum(y_train_add)*9))
    # y_train = np.concatenate((y_train,y_train_add))
    # X_train = vstack((X_train, X_train_add))

    gbdt_lr = GBDT_LR()
    y_pred_train_cvr = gbdt_lr.fit(X_train, y_train, True)

    discount = click_active_dict[axid] / np.sum(y_pred_train_cvr) / sample_percentage  #实际没有预测的高，打个折。

    if renew_bid_constand:
        bid_constant = bid_average / click_active_dict[axid] * X_win.shape[0]
    print 'time: ', time.time()-time_a, '\n'

    model_information = dict()
    model_information['time'] = datetime.datetime.now()
    model_information['gbdt_lr'] = gbdt_lr
    model_information['discount'] = discount
    model_information['col_num'] = shape[1]
    model_information['bid_constant'] = bid_constant
    model_information['test_auc'] = test_auc
    try:
        os.mkdir(path + 'models/ads/')
    except:
        pass
    try:
        os.mkdir(path + 'models/ads/'+str(axid))
    except:
        pass
    with open(path + 'models/ads/'+str(axid)+'/model_information','w') as file:
        pickle.dump(model_information,file)
    with open(path + 'models/ads/'+str(axid)+'/columns_information','w') as file:
        pickle.dump(columns_information,file)


encode_json = json.dumps(axid_dict)
spark.createDataFrame([(encode_json,)]).coalesce(1).write.mode("overwrite").text(path + 'models/axid_dict')

encode_json = json.dumps(adid_dict)
spark.createDataFrame([(encode_json,)]).coalesce(1).write.mode("overwrite").text(path + 'models/adid_dict')

with open(path +'models/auc_dict','w') as file:
    pickle.dump(auc_dict,file)
