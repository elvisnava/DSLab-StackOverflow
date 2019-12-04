from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import sys
import utils

SPLIT = 0.8

# Load data
df_read = pd.read_csv("baseline_data.csv")

# split in train and test
num_events = len(np.unique(df_read["decision_time"].values, return_counts=True)[1])
print("overall, in the data there are ", num_events, "question-answer events")
print("on average, for each event there are ", len(df_read)//num_events, " open questions")
cutoff = int(num_events * SPLIT)
df_train = df_read[df_read["decision_time"]<cutoff]
df_test = df_read[df_read["decision_time"]>=cutoff]

# Prepare training set
X_train = df_train.drop(['label', 'decision_time', 'question_id', 'Unnamed: 0', 'question_age'], axis=1) 
features = X_train.columns.tolist()
X_train = np.asarray(X_train)
Y_train = df_train['label'].values
G_train = df_train['decision_time'].values
# print(sorted(np.unique(G_train//100)))

# Prepare testing set
X_test = df_test.drop(['label', 'decision_time', 'question_id', 'Unnamed: 0',  'question_age'], axis=1) # df_test[["questionage"]] #
X_test = np.asarray(X_test)
Y_test = df_test['label'].values
G_test = df_test['decision_time'].values
# print(sorted(np.unique(G_test//100)))
assert(len(X_train)==len(Y_train))

print("Size of training set: ", len(Y_train), " Test set:", len(Y_test))
class_counts = np.unique(Y_train, return_counts=True)[1]
print("Class imbalance: 1:", class_counts[0]//class_counts[1])

## GRID SEARCH
# for DEPTH in [20,40,80,120]:
#     for WEIGHT in [20,40]:
#         for N_ESTIMATORS in [30,60,100]:
# PARAMS
N_ESTIMATORS=130
DEPTH= 20
WEIGHT = 10
print("------------")
print("PARAMS:", DEPTH, WEIGHT, N_ESTIMATORS)

# Train RF
X_train, Y_train, G_train = utils.shuffle_3(X_train, Y_train, G_train)
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=DEPTH, class_weight={0:1, 1:WEIGHT})
clf.fit(X_train,Y_train)

# investigate features
sorted_importance = np.argsort(clf.feature_importances_)
print("Features sorted by their importance for prediction:")
print(np.asarray(features)[sorted_importance])
print(clf.feature_importances_[sorted_importance])

probs_train = clf.predict_proba(X_train)
score, _ = utils.mrr(probs_train[:,1], G_train, Y_train)
print("Training MRR:", score)

probs_test = clf.predict_proba(X_test)
score, ranks = utils.mrr(probs_test[:,1], G_test, Y_test)
print("Testing MRR score:", score)

pred_targets = probs_test[:,1]
chance_mrr = utils.mrr3(out_probs=np.random.permutation(pred_targets), grouped_queries=G_test, ground_truth=Y_test)
mrr_score = utils.mrr3(out_probs=pred_targets, grouped_queries=G_test, ground_truth=Y_test)
print ("test chance mrr", chance_mrr, "test mrr utils", mrr_score)

df_test_gt = df_test
df_test_gt["rank"] = ranks.tolist()
print(df_test_gt.head())

# SAVING TEST FEATURES
df_test_gt.to_csv("ranks_features.csv")
