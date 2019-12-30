import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from pipeline_utils import NamedColumnTransformer
import pandas as pd
import scipy.stats as st
import os
import pickle
from datetime import timedelta

import data
import data_utils

import gp_features
from gp_utils import *
from collections import defaultdict

def sample_open_questions(open_questions, random_from_cdf, gt_question):
    """
    sample open questions from the histogram
    param: open_questions: are all open questions at that point of time
    param: random_from_cdf: is the histogram distribution
    param: sample_size: how many to sample
    returns: open questions which were sampled
    """
    age_vals = open_questions["question_age"].values
    uni, counts = np.unique(random_from_cdf, return_counts=True)
    val_before = 0
    final_inds = []
    for r in range(len(uni)):
        val = uni[r]
        val_set = set(np.where(age_vals<val)[0]).intersection(np.where(age_vals>val_before)[0])
        val_before=val
        if len(val_set)>counts[r]:
            subset = np.random.choice(list(val_set), counts[r], replace=False)
        else:
            subset = list(val_set)
            if r<len(uni)-1:
                counts[r+1] += counts[r]-len(val_set)
            else: # last one reached
                nr_missing = counts[r]-len(val_set)
                val_set = np.where(age_vals>val)[0]
                if len(val_set)>nr_missing:
                    rand_of_leftover = np.random.choice(val_set, nr_missing, replace=False)
                    subset.extend(rand_of_leftover)
                else:
                    subset.extend(val_set)
        final_inds.extend(subset)
    manually = 0
    if gt_question not in final_inds:
        final_inds.append(gt_question)
        manually = 1
        print("manually added")
    final_inds_index = open_questions.index[final_inds]
    final_inds_index = sorted(final_inds_index)
    open_questions = open_questions.loc[final_inds_index]
    return open_questions, manually

# PARAMETER:
redo_database_dumps = False
redo_histogram = False
# parameters for suggested questions
hour_threshold_suggested_answer = 24
only_open_questions_suggestable = True 
filter_nan_asker_id = True
# output directory (must exist)
save_dir = "burel_data"
# number of negatives samples per positive
NR_NEG = 100

# paths for cached data
fp = "../cache"
all_events_file = os.path.join(fp, "gp/all_events.pickle")
cached_data_file = os.path.join(fp, "gp/cached_data.pickle")

if redo_database_dumps:
    all_events_dataframe = data_utils.all_answer_events_dataframe(start_time=None, end_time=None, time_delta_scores_after_post=time_delta_scores_after_posts, filter_empty_asker=filter_nan_asker, filter_empty_target_user=filter_nan_answerer)
    all_events_dataframe.to_pickle(all_events_file)

    cached_data = data.DataHandleCached()
    with open(cached_data_file, "wb") as f:
        pickle.dump(cached_data, f)
else:
    all_events_dataframe = pd.read_pickle(all_events_file)

    with open(cached_data_file, "rb") as f:
        cached_data = pickle.load(f)

# define data and feature handles
data_handle = data.Data()

feature_collection = gp_features.GP_Feature_Collection(
gp_features.GP_Features_affinity(),
gp_features.GP_Features_Question(),
gp_features.GP_Features_user())


# start and end of data
start_time = data_utils.make_datetime("01.01.2012 00:01")
end_time = data_utils.make_datetime("01.01.2017 00:01") # data_utils.make_datetime("01.03.2012 00:01")

all_feates_collector = list()
n_candidates_collector = list()

save_every = 300
q_a_pair_counter = 1

## Approximate questionage distribution
if redo_histogram:
    questionage_table = data_handle.query("SELECT a.id, (answercreationdate-CreationDate) as questionage FROM (SELECT parentid as Id, creationdate as answercreationdate FROM Posts WHERE PostTypeId=2) a LEFT JOIN Posts b ON a.Id=b.Id;")
    questionage_table["questionage"] = questionage_table["questionage"].dt.days +  (questionage_table["questionage"].dt.seconds)/(24*60*60)
    age_vals = questionage_table["questionage"].values
    age_vals = age_vals[age_vals>0]
    age_vals = age_vals[age_vals<100]
    hist, bins = np.histogram(age_vals, bins=500)
    bin_midpoints = bins[1:] # + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(100)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    with open("random_from_cdf.pickle", "wb") as outfile:
        pickle.dump(random_from_cdf, outfile)
else:
    with open("random_from_cdf.pickle", "rb") as outfile:
        random_from_cdf = pickle.load(outfile)


user_dic = defaultdict(int)


# START ITERATING THROUGH DATA
prev_answertime = start_time
for i, event in enumerate(data_utils.all_answer_events_iterator(timedelta(days=2), start_time=start_time, end_time=end_time)):
    if np.isnan(event.answerer_user_id) or np.isnan(event.asker_user_id):
         continue
    
    if i%100 ==0 :
        avg_candidates = np.mean(n_candidates_collector)
        print("Preptraining at {}| on average {} candidates in the last {} suggested_question_events".format(event.answer_date, avg_candidates, len(n_candidates_collector)))
        n_candidates_collector = list()
    
    # only add to data if user has answered more than five questions and the answer is more than 12 hours after the last one
    if user_dic[event.answerer_user_id] >=5 and event.answer_date> prev_answertime + timedelta(hours = 12):
        open_questions = get_suggestable_questions(event.answer_date, cached_data, only_open_questions_suggestable, hour_threshold_suggested_answer, filter_nan_asker_id)
        # add question age
        question_dates = [pd.Timestamp(x) for x in open_questions["question_date"].values]
        open_questions["question_age"] = [event.answer_date - question_event_time for question_event_time in question_dates]
        open_questions["question_age"] = (open_questions["question_age"].dt.days +  (open_questions["question_age"].dt.seconds)/(24*60*60))
        
        gt_ind = np.where(open_questions.question_id == event.question_id)[0]
        if len(open_questions) ==0 or len(gt_ind)==0:
            print("Warning: question already answered or For answer id {} (to question {}) there was not a single suggestable question".format(event.answer_id, event.question_id))
            continue
        if len(open_questions) <= NR_NEG:
            suggestable_questions = open_questions
            manually = 0
        else:
            print("sampling")
            suggestable_questions, manually = sample_open_questions(open_questions, random_from_cdf, gt_ind[0])
        
        assert(np.any(suggestable_questions.question_id == event.question_id))
           
        n_candidates_collector.append(len(suggestable_questions))
        
        # append to feature and label list
        feats = feature_collection.compute_features(event.answerer_user_id, suggestable_questions, event.answer_date)
        label = suggestable_questions.question_id.values == event.question_id
        
        # add some more information
        feats["question_id"] = suggestable_questions.question_id.values.tolist() # remember question ids
        feats["decision_time"] = q_a_pair_counter # for MRR need to remember groups
        feats["label"] = label.astype(int)
        feats["manually_added"] = manually
        feats["answer_date"] = pd.Series([event.answer_date for _ in range(len(feats))])

        all_feates_collector.append(feats)

        assert(np.sum(np.asarray(label).astype(int))==1)
        q_a_pair_counter+=1
        prev_answertime = event.answer_date

        # save inbetween and clear variables in order to backup
        if (q_a_pair_counter) % save_every==0:
            save_name = "feature_data_"+str((q_a_pair_counter+1)//save_every)+".csv"
            features_table = pd.concat(all_feates_collector, axis=0)
            features_table.to_csv(os.path.join(save_dir, save_name), index=False)
            print("Successfully saved data inbetween", save_name)
            del features_table 
            del all_feates_collector
            all_feates_collector = list()

    feature_collection.update_pos_event(event) # update features in any case
    user_dic[event.answerer_user_id] += 1

    
   

if (q_a_pair_counter) % save_every != 0: # last batch hasn't been saved
    save_name = "feature_data_"+str((q_a_pair_counter+1)//save_every + 1)+".csv"
    features_table = pd.concat(all_feates_collector, axis=0)
    features_table.to_csv(os.path.join(save_dir, save_name), index=False)
print("Successfully saved last data")
print("Finished")