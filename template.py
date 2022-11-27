#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/soultaffy/oss2

import sys
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_dataset(dataset_path):
        #To-Do: csv 파일에서 데이터 가져오기
  data_df = pd.read_csv(dataset_path)
  return data_df

def dataset_stat(dataset_df):	
        #To-Do: 데이터셋의 스탯 반환
  n_feats = dataset_df.shape[1]
  n_class0 = dataset_df[dataset_df.columns[0]].count()
  n_class1 = dataset_df[dataset_df.columns[1]].count()  
  n_stats = (n_feats, n_class0, n_class1)
  return n_stats

def split_dataset(dataset_df, testset_size):
	#To-Do: 스플릿 train data, test data, train label, test label
  dataset_df.groupby("target").size()
  X = dataset_df.drop(columns="target", axis=1)
  y = dataset_df["target"]
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testset_size, random_state=2)

  split_df = (x_train, x_test, y_train, y_test)
  return split_df

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Decision tree model 이용 training & predict
  dt_cls = DecisionTreeClassifier()
  dt_cls.fit(x_train, y_train)
  dt_predict_x = dt_cls.predict(x_test)
  dt_predict_y = dt_cls.predict(y_test)

  acc_dt = metrics.accuracy_score(y_test, dt_predict_x)
  prec_dt = metrics.precision_score(y_test, dt_predict_x)
  recall_dt = metrics.recall_score(y_test, dt_predict_x)

  perf_dt = (acc_dt, prec_dt, recall_dt)
  return perf_dt

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Random forest model 이용 training & predict
  rf_cls = RandomForestClassifier()
  rf_cls.fit(x_train, y_train)
  rf_predict_x = rf_cls.predict(x_test)
  rf_predict_y = rf_cls.predict(y_test)

  acc_rf = metrics.accuracy_score(y_test, rf_predict_y)
  prec_rf = metrics.precision_score(y_test, rf_predict_y)
  recall_rf = metrics.recall_score(y_test, rf_predict_y)

  perf_rf = (acc_rf, prec_rf, recall_rf)
  return perf_rf

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: SVM model 이용 training & predict
  svm_cls = SVC()
  svm_cls.fit(x_train, y_train)
  svm_predict_x = svm_cls.predict(x_test)
  svm_predict_y = svm_cls.predict(y_test)
  
  #acc_svm = metrics.accuracy_score(y_test, svm_predict_y)
  #prec_svm = metrics.precision_score(y_test, svm_predict_y)
  #recall_svm = metrics.recall_score(y_test, svm_predict_y)

  # Pipeline
  svm_pipe = make_pipeline(StandardScaler(), SVC())
  svm_pipe.fit(x_train, y_train)
  svm_pipe_predict_x = svm_pipe.predict(x_test)
  svm_pipe_predict_y = svm_pipe.predict(y_test)

  acc_svm = metrics.accuracy_score(y_test, svm_pipe_predict_y)
  prec_svm = metrics.precision_score(y_test, svm_pipe_predict_y)
  recall_svm = metrics.recall_score(y_test, svm_pipe_predict_y)

  perf_svm = (acc_svm, prec_svm, recall_svm)
  return perf_svm

#=======================================

def print_performances(acc, prec, recall):
  #Do not modify this function!
  print ("Accuracy: ", acc)
  print ("Precision: ", prec)
  print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)  
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
