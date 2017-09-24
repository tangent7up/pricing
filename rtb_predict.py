from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType,StringType,IntegerType
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
from scipy.sparse.csr import csr_matrix
from scipy.sparse import vstack,hstack
import numpy as np

import datetime
import math
import os
import pickle
from utils import *
import time
import json

from conf_predict import *

def field_transf(r):
    # r_=r.asDict()
    r_ = r.copy()
    r_['osv']=r['osv'].split('.')[0]
    r_['model']=r['model'].split(',')[0]
    r_['data']=[i['value'] for i in r['data'][-1]['segment']]
    a=datetime.datetime.strptime(r['drt'],'%Y-%m-%d+%H:%M:%S')
    r_['minute']=str(a.minute)
    r_['weekday']=str(a.weekday())
    r_['time_zone']=str(a.hour)
    r_.pop('drt')
    return r_ # Row(**r_)

def RTF_target(row, catagory, quality, columns_dict, columns_dict_len):
    new_row = []
    ind=1
    for column in catagory:
        index = columns_dict[column].get(row[column],0)
        if index != 0: new_row.append(ind + index-1)
        ind += columns_dict_len[column]
    for column in quality:
        words=row[column]
        if words is not None:
            for w in words:
                index=columns_dict[column].get(w,None)
                if index != None:
                    new_row.append(ind + index)
        ind += columns_dict_len[column]
    new_row = sorted(new_row)
    return new_row

def getdict(d,l):
    if l == []:
        return d
    else:
        return getdict(d[l[0]],l[1:])

def get_cvr_bid(my_row_):
    axid = my_row_["axid"]
    adid = my_row_["adid"]

    my_row_['ad_paytype'] = my_row['ad']['paytype']
    my_row_2 = field_transf(my_row_)

    # 如果模型还没生成，或者还没成熟，那么改用本adid的其他axid的模型来预测。
    # if axid not in ad_predict:
    #     same_adid = [i for i in adid_dict[axid_dict[axid]] if i in ad_predict]
    #     if same_adid == []: # 连同一个adid的axid都不存在模型的情况
    #         use_axid = sorted(auc_dict.items(),key = lambda x: x[1],reverse = True)[0][0]
    #     else: # 能找到同一个adid的其他axid的模型
    #         use_axid = sorted(same_adid,key = lambda x: auc_dict[x],reverse = True)[0]

    try:
        usd_axid = model_match[axid]
    except:
        if axid not in ad_models + ad_old_models:
            use_axid = axid_dict.get(axid,ad_models[0])
            if use_axid not in ad_models + ad_old_models: use_axid = ad_models[0]
        else: use_axid = axid

    new_row = RTF_target(my_row_2, catagory, quality, model_dict[use_axid]['columns_dict'], model_dict[use_axid]['columns_dict_len'])
    cols = np.array(new_row) - 1
    rows = np.zeros(len(cols))
    data = np.ones(len(cols))
    my_row_3 = csr_matrix((data, (rows, cols)), shape=(1, model_dict[use_axid]['col_num']))
    my_row_3[:,-1]=my_row_3[-1,-1] # 最后一个元素如果是0，稀疏矩阵不会存，下一步进xgboost会报错，所以要先存一下最后一个数

    cvr = model_dict[use_axid]['gbdt_lr'].predict(my_row_3)[0] * model_dict[use_axid]['discount'] # 这句耗时占 95% 以上
    bid = cvr * model_dict[use_axid]['bid_constant'] * bid_constant_correct.get(axid,1)
    return cvr,bid

if __name__ == "__main__":

    model_dict = dict() # 用来存放各个模型的信息
    for axid in ad_models:
        with open('models/ads/'+str(axid)+'/model_information','r') as file1:
            with open('models/ads/'+str(axid)+'/columns_information','r') as file2:
                model_information = pickle.load(file1)
                columns_information = pickle.load(file2)
        model_dict[axid] = dict(columns_information, **model_information)

    for axid in ad_old_models:
        with open('old_models/ads/'+str(axid)+'/model_information','r') as file1:
            with open('old_models/ads/'+str(axid)+'/columns_information','r') as file2:
                model_information = pickle.load(file1)
                columns_information = pickle.load(file2)
        model_dict[axid] = dict(columns_information, **model_information)

    encode_json = get_txt(path + 'models/adid_dict')
    adid_dict = json.loads(encode_json[0])
    adid_dict = dict([(int(tup[0]),tup[1]) for tup in adid_dict.iteritems()])
    encode_json = get_txt(path + 'models/axid_dict')
    axid_dict = json.loads(encode_json[0])
    axid_dict = dict([(int(tup[0]),tup[1]) for tup in axid_dict.iteritems()])

    with open(path + 'constants/bid_constant_correct', 'r') as file:
        bid_constant_correct = json.load(file)
        bid_constant_correct = dict([(int(tup[0]),tup[1]) for tup in bid_constant_correct.iteritems()])

    with open(path + 'constants/model_match', 'r') as file:
        model_match = json.load(file)
        model_match = dict([(int(tup[0]),tup[1]) for tup in model_match.iteritems()])


    predict_col=['drt', 'ad.adid', 'ad.axid', 'paytype', 'product',
    'serv', 'ad.ad_pkg', 'ad.dspchn', 'app.app_pkg',
    'app.name', 'app.adxslotid', 'app.slotid', 'app.adplace_width', 'app.adplace_height',
    'app.rtbchn', 'app.app', 'app.dsslotid', 'network.carrier',
    'network.nettype', 'geo.country', 'geo.regionc', 'geo.cityc', 'geo.lat',
    'geo.lon', 'device.brand', 'device.model', 'device.osv',
    'device.screen_width', 'device.screen_height', 'device.dv',
    'user.yob', 'user.gender', 'user.edu', 'user.spec_usrtag', 'user.data',
    'content.keyword', 'content.cclabel', 'content.cat1', 'content.cat2', 'content.author',
    'content.cont_id', 'content.program']


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

    # 下面两行是模拟取数的动作，实际上不是这样取的。
    # df_bid_train = spark.read.load('bid')
    df_bid_train = spark.read.load('/data/bid/producttype=*/date=2017-09-15')
    my_rows = df_bid_train.take(10000)

    # 下面开始循环
    now = time.time()
    for my_row in my_rows:
        my_row_ = dict([(i.split('.')[-1],getdict(my_row,i.split('.'))) for i in predict_col]) # 根据predict_col 筛选预测需要的字段

        cvr, bid = get_cvr_bid(my_row_) # 就是这个预测方法
        print cvr, bid
    print time.time() - now
