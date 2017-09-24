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
from pyspark.mllib.regression import LabeledPoint

import numpy as np

import datetime
import math
import os
import pickle
from utils import *
import time
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from conf_predict import *

def load_dict_df(path):
    try:
        d = spark.read.parquet(path)
        d = dict(d.rdd.map(lambda x: (x['_1'],x['_2'])).collect())
    except:
        d = dict()
    return d


if __name__ == "__main__":
    ad_list_ = os.listdir(path + 'models/ads')
    ad_list = []
    for a in ad_list_:
        try:
            int(a[:6])
            assert(len(a)>5)
            ad_list.append(int(a))
        except:
            pass
    ad_list_all = []
    for a in ad_list_:
        try:
            int(a[:5])
            ad_list_all.append(int(a))
        except:
            pass
    del(ad_list_)

    encode_json = get_txt(path + 'models/adid_dict')
    adid_dict = json.loads(encode_json[0])
    adid_dict = dict([(int(tup[0]),tup[1]) for tup in adid_dict.iteritems()])
    encode_json = get_txt(path + 'models/axid_dict')
    axid_dict = json.loads(encode_json[0])
    axid_dict = dict([(int(tup[0]),tup[1]) for tup in axid_dict.iteritems()])

    session_dict = load_dict_df(path+'models/session_dict')
    has_active_dict = load_dict_df(path+'models/has_click_dict')
    last_train_time = load_dict_df(path+'models/last_train_time')
    axid_dict = load_dict_df(path+'models/axid_dict_df')
    adid_dict = load_dict_df(path+'models/adid_dict_df')

    with open(path + 'models/auc_dict','r') as file:
        auc_dict = pickle.load(file)
    with open(path + 'constants/bid_constant_correct', 'r') as file:
        bid_constant_correct = json.load(file)
        bid_constant_correct = dict([(int(tup[0]),tup[1]) for tup in bid_constant_correct.iteritems()])


    model_dict = dict()
    for axid in ad_list_all:
        with open('models/ads/'+str(axid)+'/model_information','r') as file1:
            with open('models/ads/'+str(axid)+'/columns_information','r') as file2:
                model_information = pickle.load(file1)
                columns_information = pickle.load(file2)
        model_dict[axid] = dict(columns_information, **model_information)

    names = spark.read.option("header", "true").option("escape", "\"").csv("youmi_spot.spot.csv")
    names.createOrReplaceTempView('spot')
    names2 = names.select("adid","remark")
    name_dict = dict(names.rdd.map(lambda x: (str(x["spotid"]),x["remark"])).collect())

    rows = []
    for axid in ad_list:
        adid = axid_dict[axid]
        r = Row(axid = axid, remark = name_dict[str(axid)], test_cvr = float(auc_dict[axid]), has_active = has_active_dict[axid], session = session_dict[axid], cols = len(model_dict[axid]['new_columns']),\
            update_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_train_time[axid]+3600*8)),
            zadid = adid, zadid_test_cvr = float(auc_dict[adid]), zadid_has_active = has_active_dict[adid], zadid_session = session_dict[adid], zadid_cols = len(model_dict[adid]['new_columns']),\
            zaxid_update_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_train_time[adid]+3600*8)))
        rows.append(r)

    df = spark.createDataFrame(rows)
    df.coalesce(1).write.mode("overwrite").format("parquet").save(path +'models/table')
    df.show(100)
    pd_df = df.toPandas()
    pd_df.to_csv(path +'models/table.csv')

    # df.write.format("jdbc").\
    #     options(url="jdbc:mysql://172.31.20.40:3306/youmi_rtb_report", \
    #     dbtable="jtan_cvr_rtb", user="youmi_db_emr_ad", \
    #     driver="com.mysql.jdbc.Driver", password="zPc^9aq3!aM&").save(mode="overwrite")




    # 生成新旧 adid 的报表
    ad_list_ = os.listdir(path + 'old_models/ads')
    ad_list = []
    for a in ad_list_:
        try:
            int(a[:5])
            ad_list.append(int(a))
        except:
            pass
    del(ad_list_)

    session_dict = load_dict_df(path+'old_models/session_dict')
    has_active_dict = load_dict_df(path+'old_models/has_click_dict')
    last_train_time = load_dict_df(path+'old_models/last_train_time')

    with open(path + 'old_models/auc_dict','r') as file:
        auc_dict = pickle.load(file)

    model_dict = dict()
    for adid in ad_list:
        with open('old_models/ads/'+str(adid)+'/model_information','r') as file1:
            with open('old_models/ads/'+str(adid)+'/columns_information','r') as file2:
                model_information = pickle.load(file1)
                columns_information = pickle.load(file2)
        model_dict[adid] = dict(columns_information, **model_information)

    names = spark.read.option("header", "true").option("escape", "\"").csv("youmi_spot.spot.csv")
    # names.createOrReplaceTempView('spot')
    # names2 = names.select("adid","remark")
    name_dict = dict(names.rdd.map(lambda x: (str(x["adid"]),x["remark"])).collect())

    rows = []
    for adid in ad_list:
        r = Row(adid = adid, remark = name_dict[str(adid)], test_cvr = float(auc_dict[adid]), has_active = has_active_dict[adid], session = session_dict[adid], cols = len(model_dict[adid]['new_columns']),\
            update_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_train_time[adid]+3600*8)),time = 'old')
        rows.append(r)

    ad_list_ = os.listdir(path + 'models/ads')
    ad_list = []
    for a in ad_list_:
        try:
            assert(len(a) ==5)
            ad_list.append(int(a))
        except:
            pass
    del(ad_list_)

    session_dict = load_dict_df(path+'models/session_dict')
    has_active_dict = load_dict_df(path+'models/has_click_dict')
    last_train_time = load_dict_df(path+'models/last_train_time')

    with open(path + 'models/auc_dict','r') as file:
        auc_dict = pickle.load(file)

    model_dict = dict()
    for adid in ad_list:
        with open('models/ads/'+str(adid)+'/model_information','r') as file1:
            with open('models/ads/'+str(adid)+'/columns_information','r') as file2:
                model_information = pickle.load(file1)
                columns_information = pickle.load(file2)
        model_dict[adid] = dict(columns_information, **model_information)

    names = spark.read.option("header", "true").option("escape", "\"").csv("youmi_spot.spot.csv")
    # names.createOrReplaceTempView('spot')
    # names2 = names.select("adid","remark")
    name_dict = dict(names.rdd.map(lambda x: (str(x["adid"]),x["remark"])).collect())

    for adid in ad_list:
        r = Row(adid = adid, remark = name_dict[str(adid)], test_cvr = float(auc_dict[adid]), has_active = has_active_dict[adid], session = session_dict[adid], cols = len(model_dict[adid]['new_columns']),\
            update_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_train_time[adid]+3600*8)), time = 'new')
        rows.append(r)
    rows = sorted(rows,key = lambda x: x["adid"], reverse = True)

    df = spark.createDataFrame(rows)
    df.show(100)
    pd_df = df.toPandas()
    pd_df.to_csv(path +'models/adid_table.csv')
