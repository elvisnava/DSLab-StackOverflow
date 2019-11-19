from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import sys
from utils import split_inds, mrr, shuffle_3
# from sklearn.utils import shuffle

# SPLIT VERSION:
# split such that the test set consists of completely different users: "user"
# split randomly, such that test and train might contain question-answer pairs of the same user: "mixed"
# split such that completly different users from a completely different time frame are selected: "time"
SPLIT_MODE = "time"
SPLIT = 0.7
train_data = "data" # "data_14_1-14_5" # "data/"
test_data = "data_2015" # "data_2016" 
# data_2016: wrong processig - no user features
# data_2016_2: with user features, but some do not have user features because to early
# data 2016_3: the ones without the ones with no user features

# Load data
dataframes = []
for file in os.listdir(train_data):
    if file[0]=="." or "example" in file:
        continue
    df_read = pd.read_csv(os.path.join(train_data,file), index_col="id")
    # print("Successfully loaded ", file)
    dataframes.append(df_read)

if SPLIT_MODE=="user":
    lengths = [len(d) for d in dataframes]
    nr_data = sum(lengths)*SPLIT
    inds = np.random.permutation(len(lengths))
    dataframes_train = []
    summed_data = 0
    k=0
    while summed_data < nr_data:
        dataframes_train.append(dataframes[inds[k]])
        k+=1
        summed_data += lengths[inds[k]]
    dataframes_test = [dataframes[j] for j in inds[k:]]
    df_train = pd.concat(dataframes_train)
    df_test = pd.concat(dataframes_test)
    print("Number of users in train set:", len(dataframes_train))
    print("Number of users in test set:", len(dataframes_test))
    print("Number of samples including all answer-question pairs: Train:", len(df_train), " Test:", len(df_test))
elif SPLIT_MODE=="mixed":
    # Take completely random sample (same user might be in test and train set, for different answers)
    df = pd.concat(dataframes)
    # Split in train and tests - split by group (one answer-open_questiosn block must be in same part)
    df_grouped = df.groupby("decision_time")
    # df_train, df_test = split_groups(df_grouped)
    nr_groups = len(df_grouped)
    train_inds, test_inds = split_inds(nr_groups, split=SPLIT)
    df_train = pd.concat([ df_grouped.get_group(group) for i,group in enumerate(df_grouped.groups) if i in train_inds])
    df_test = pd.concat([ df_grouped.get_group(group) for i,group in enumerate(df_grouped.groups) if i in test_inds])
elif SPLIT_MODE=="time":
    df_train = pd.concat(dataframes)
    dataframes_test = []
    for file in os.listdir(test_data):
        if file[0]=="." or "example" in file:
            continue
        df_read = pd.read_csv(os.path.join(test_data,file), index_col="id")
        # print("Successfully loaded ", file)
        dataframes_test.append(df_read)
    df_test = pd.concat(dataframes_test)
    print("Number of users in train set:", len(dataframes))
    print("Number of users in test set:", len(dataframes_test))
    print("Number of samples including all answer-question pairs: Train:", len(df_train), " Test:", len(df_test))
else:
    print("ERROR: SPLIT MODE DOES NOT EXIST")
    sys.exit()

# Prepare training set
X_train = df_train.drop(['label', 'decision_time', 'reputation_user', 'reputation_asker'], axis=1) # df_train[["questionage"]] #
features = X_train.columns.tolist()
X_train = np.asarray(X_train)
Y_train = df_train['label'].values
G_train = df_train['decision_time'].values
# print(sorted(np.unique(G_train//100)))

# Prepare testing set
X_test = df_test.drop(['label', 'decision_time', 'reputation_user', 'reputation_asker'], axis=1) # df_test[["questionage"]] #
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
WEIGHT = 20
print("------------")
print("PARAMS:", DEPTH, WEIGHT, N_ESTIMATORS)

# Train RF
X_train, Y_train, G_train = shuffle_3(X_train, Y_train, G_train)
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=DEPTH, class_weight={0:1, 1:WEIGHT})
clf.fit(X_train,Y_train)

# investigate features
sorted_importance = np.argsort(clf.feature_importances_)
print("Features sorted by their importance for prediction:")
print(np.asarray(features)[sorted_importance])
print(clf.feature_importances_[sorted_importance])

probs_train = clf.predict_proba(X_train)
score, _ = mrr(probs_train[:,1], G_train, Y_train)
print("Training MRR:", score)

probs_test = clf.predict_proba(X_test)
score, ranks = mrr(probs_test[:,1], G_test, Y_test)
print("Testing MRR score:", score)

df_test_gt = df_test
df_test_gt["rank"] = ranks.tolist()
print(df_test_gt.head())

# SAVING TEST FEATURES
df_test_gt.to_csv("ranks_features.csv")
