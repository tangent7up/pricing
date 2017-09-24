# -*- coding:utf-8 -*-
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from scipy.sparse.csr import csr_matrix
from scipy.sparse import vstack,hstack
import numpy as np

from itertools import combinations
import datetime
import math
import os
import pickle


def solve_monotonous(mon_fun,exp_value = 0.,left_limit=-10000.,right_limit=10000.):
    L = mon_fun(left_limit) - exp_value
    R = mon_fun(right_limit) - exp_value
    if np.sign(L) == np.sign(R):
        return 'error'
    if np.abs(right_limit - left_limit) < 0.000001:
        return (right_limit + left_limit)/2
    M = mon_fun(left_limit/2 + right_limit/2) - exp_value
    if np.sign(M) == np.sign(L):
        return solve_monotonous(mon_fun,exp_value,left_limit/2 + right_limit/2,right_limit)
    else:
        return solve_monotonous(mon_fun,exp_value,left_limit,left_limit/2 + right_limit/2)

def myStr(string):
    if type(string)==unicode:
        return string.encode('utf-8')
    else:
        return str(string)

def getFiles(path,contain="",startwith=""):
    filepaths=[]
    files = os.listdir(path)
    for f in files:
        if(os.path.isdir(os.path.join(path,f))):
            filepaths+=getFiles(os.path.join(path,f),contain,startwith)
        elif(f.startswith(".") or contain not in f or not f.startswith(startwith)):
            continue
        else:
            filepaths.append(os.path.join(path,f))
    return filepaths

def get_txt(dirpath):
    for filepath in getFiles(dirpath,startwith="part"):
        with open(filepath , 'r') as file:
            lines = file.readlines()
            lines = [i.strip() for i in lines]
    return lines




def load_svm(dirs,header=None):
    """
    return a set contain clicks
    """
    for dirpath in dirs:
        get_head = True
        for filepath in getFiles(dirpath,startwith="part"):
            if get_head:
                try:
                    X, y = load_svmlight_file(filepath)
                    get_head = False
                except Exception as e:
                    pass
            else:
                try:
                    X_, y_ = load_svmlight_file(filepath)
                    X = vstack((X,X_))
                    y = np.concatenate((y,y_))
                except Exception as e:
                    continue
    return X, y

def my_DataSplit(X, y, num, seed=20):
    if num == 1: return [X], [y]
    else:
        X1, X2, y1, y2 = train_test_split(X, y, test_size=1./num, random_state=seed)
        X_out, y_out = my_DataSplit(X1, y1, num-1, seed+1)
        X_out.append(X2)
        y_out.append(y2)
    return X_out, y_out

def get_featurename(new_columns,x):
    if type(x)==list or type(x)==tuple:
        for i in x:
            print new_columns[i]
    else: print new_columns[x]

def to_sparse_matrix(X,cols):
    r = np.array([[i]*len(j) for i,j in enumerate(X)]).flatten()
    c = (X-1).flatten()
    data = np.array([1]*len(r))
    shape = (len(X),cols)
    del(X)
    X = csr_matrix((data, (r, c)), shape)
    return X


class OneHot:
    def __init__(self,keyMapDict = None):
        self.keyMapDict = keyMapDict

    def fit_transform(self,X):
        rows, cols = X.shape
        self.cum_count = np.zeros((1, cols), dtype=np.int32)
        # field_values = [0] * colskeyMapDict
        for j in range(cols):
            # field_values[j] = set(np.unique(X_train_leaves[:, j]))
            if j == 0:
                self.cum_count[0][j] = len(np.unique(X[:, j]))
            else:
                self.cum_count[0][j] = len(np.unique(X[:, j])) + self.cum_count[0][j-1]

        self.keyMapDict = [0] * cols
        for j in range(cols):
            self.keyMapDict[j] = dict()
            if j == 0:
                initial_index = 1
            else:
                initial_index = self.cum_count[0][j-1]+1
            for i in range(rows):
                if X[i, j] not in self.keyMapDict[j]:
                    self.keyMapDict[j][X[i, j]] = initial_index
                    X[i, j] = initial_index
                    initial_index = initial_index + 1
                else:
                    X[i, j] = self.keyMapDict[j][X[i, j]]
        return X

    def transform(self,X):
        rows, cols = X.shape
        for j in range(cols):
            if j == 0:
                initial_index = 1
            else:
                initial_index = self.cum_count[0][j-1]+1
            for i in range(rows):
                if X[i, j] in self.keyMapDict[j]:
                    X[i, j] = self.keyMapDict[j][X[i, j]]
        return X


def predict_one(table_score,average,row,weight):
    indexs = row.indices[row.data==1.]
    feature_tuples = list(combinations(indexs, 2))
    score = 0
    w1 = 0
    w2 = 0
    for ind in indexs:
        s = table_score.get(ind,average)
        w = 1
        score += s * w * weight
        w1 += w * weight
    for tup in feature_tuples:
        s = table_score.get(tup,average)
        w = 1
        score += s * w
        w2 += w
    final_score = score / (w1 + w2)
    return final_score

def predict_one_2(table_score,average,row,weight):
    indexs = row.indices[row.data==1.]
    feature_tuples = list(combinations(indexs, 2))
    score = 0
    w1 = 0
    w2 = 0
    for ind in indexs:
        s = table_score.get(ind,average)
        w = np.abs(average-s)+0.000001
        score += s * w * weight
        w1 += w * weight
    for tup in feature_tuples:
        s = table_score.get(tup,average)
        w = np.abs(average-s)+0.000001
        score += s * w
        w2 += w
    final_score = score / (w1 + w2)
    return final_score

def predict_one_3(table_score,average,row,weight):
    indexs = row.indices[row.data==1.]
    feature_tuples = list(combinations(indexs, 2))
    score = 0
    w1 = 0
    w2 = 0
    a = np.log(average)
    for ind in indexs:
        s = table_score.get(ind,average)
        score += (np.log(s) - a) * weight
    for tup in feature_tuples:
        s = table_score.get(tup,average)
        score += np.log(s) - a
    final_score = score
    return final_score

class Dictionary:
    def __init__(self,a=3):
        self.average = 0
        self.a = a

    def fit(self,X,y,cross_remove_columns = set()):
        self.table_dict=dict()
        for i in range(len(y)):
            # one_dim dict
            indexs = X[i].indices[X[i].data==1.]
            for ind in indexs:
                origin_value = self.table_dict.get(ind,[0,0])
                origin_value[1]+=1
                if y[i] == 1.: origin_value[0]+=1
                self.table_dict[ind] = origin_value
            # remove too much value field
            indexs = [ind for ind in indexs if ind not in cross_remove_columns]
            feature_tuples = list(combinations(indexs,2))
            # feature_tuples = list(combinations(X_train[i].indices[X_train[i].data==1.],2))
            # two_dim dict
            for tup in feature_tuples:
                origin_value = self.table_dict.get(tup,[0,0])
                origin_value[1]+=1
                if y[i] == 1.: origin_value[0]+=1
                self.table_dict[tup] = origin_value
        ones_num = len(y[y==1])
        all_num = len(y)
        self.average = average = float(ones_num) / all_num
        self.gamma = float(self.a) / len(y)
        self.table_score=dict((i[0],(i[1][0] + self.gamma * ones_num)/(i[1][1] + self.gamma * all_num)) for i in self.table_dict.iteritems())

    def predict(self,X_test,predict_fun = predict_one, weight=0):
        return np.array([predict_fun(self.table_score,self.average,row,weight) for row in X_test])



# class GBDT_LR:
#     def __init__(self,nthread=4, learning_rate=0.08, n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5,n_jobs=-1, C=0.1, penalty='l1'):
#         self.nthread = nthread
#         self.learning_rate = learning_rate
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.gamma = gamma
#         self.subsample = subsample
#         self.colsample_bytree = colsample_bytree
#         self.n_jobs = n_jobs
#         self.C = C
#         self.penalty = penalty
#         self.xgb = xgb.XGBClassifier(self.nthread, self.learning_rate, self.n_estimators, self.max_depth, self.gamma, self.subsample, self.colsample_bytree)
#         self.lr = LogisticRegression(self.n_jobs, self.C, self.penalty)
#
#     def fit(self,X,y,pred = False):
#         self.xgb.fit(X,y)
#         self.onehot = OneHot()
#         X_gbdt = self.onehot.fit_transform(self.xgb.apply(X))
#         self.cols = np.max(X_gbdt[:,-1])
#         X_ext = hstack([X, to_sparse_matrix(X_gbdt,self.cols)])
#         self.lr.fit(X_ext, y)
#         if pred:
#             return self.lr.predict_proba(X_ext)[:, 1]
#
#     def predict(self,X_test):
#         return self.lr.predict_proba(hstack([X_test, to_sparse_matrix(self.onehot.transform(self.xgb.apply(X_test)),self.cols)]))[:, 1]


class GBDT_LR:
    def __init__(self):
        self.xgb = xgb.XGBClassifier(nthread=1, learning_rate=0.08, n_estimators=10, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
        self.lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')

    def fit(self,X,y,pred = False):
        self.xgb.fit(X,y)
        self.onehot = OneHot()
        X_gbdt = self.onehot.fit_transform(self.xgb.apply(X))
        self.cols = np.max(X_gbdt[:,-1])
        X_ext = hstack([X, to_sparse_matrix(X_gbdt,self.cols)])
        self.lr.fit(X_ext, y)
        if pred:
            return self.lr.predict_proba(X_ext)[:, 1]

    def predict(self,X_test):
        return self.lr.predict_proba(hstack([X_test, to_sparse_matrix(self.onehot.transform(self.xgb.apply(X_test)),self.cols)]))[:, 1]
