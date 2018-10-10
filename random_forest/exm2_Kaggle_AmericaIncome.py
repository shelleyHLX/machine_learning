# coding: utf-8

import pandas as pd  # load csv's (pd.read_csv)
import numpy as np   # math (lin. algebra)

import sklearn as skl   # machine learning
from sklearn.ensemble import RandomForestClassifier
#from plotnine import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def get_train_data():
    train_path = "adult.data"
    test_path = 'adult.test'
    columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus','Occupation','Relationship','Race','Sex',
               'CapitalGain','CapitalLoss','HoursPerWeek','Country','Income']
    df_train_set = pd.read_csv(train_path, names=columns)
    print(df_train_set.head())
    return df_train_set

def get_test_data():
    test_path = 'adult.test'
    columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus','Occupation','Relationship','Race','Sex',
               'CapitalGain','CapitalLoss','HoursPerWeek','Country','Income']
    df_test_set = pd.read_csv(test_path, names=columns)
    print(df_test_set.head())
    return df_test_set




