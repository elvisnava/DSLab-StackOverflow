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

cached_data = data.DataHandleCached()
data_handle = data.Data()

feature_collection = gp_features.GP_Feature_Collection(
gp_features.GP_Features_affinity(),
gp_features.GP_Features_TTM(),
gp_features.GP_Features_Question(),
gp_features.GP_Features_user())

# parameters for suggested questions
hour_threshold_suggested_answer = 24
only_open_questions_suggestable = False 
filter_nan_asker_id = True

save_dir = "baseline_data"

start_time = None # data_utils.make_datetime("01.01.2012 00:01")
end_time = data_utils.make_datetime("01.01.2016 00:01") # data_utils.make_datetime("01.03.2012 00:01")

all_feates_collector = list()

n_candidates_collector = list()

save_every = 10000
q_a_pair_counter = 1

for i, event in enumerate(data_utils.all_answer_events_iterator(timedelta(days=2), start_time=start_time, end_time=end_time)):
    if np.isnan(event.answerer_user_id) or np.isnan(event.asker_user_id):
         continue
    
    if i%100 ==0 :
        avg_candidates = np.mean(n_candidates_collector)
        print("Preptraining at {}| on average {} candidates in the last {} suggested_question_events".format(event.answer_date, avg_candidates, len(n_candidates_collector)))
        n_candidates_collector = list()

    if is_user_answers_suggested_event(event, hour_threshold_suggested_answer):
        
        suggestable_questions = get_suggestable_questions(event.answer_date, cached_data, only_open_questions_suggestable, hour_threshold_suggested_answer, filter_nan_asker_id)
        if len(suggestable_questions) ==0:
            # warnings.warn("For answer id {} (to question {}) there was not a single suggestable question".format(event.answer_id, event.question_id))
            continue
            
        n_candidates_collector.append(len(suggestable_questions))
        
        # erst appenden wenn ueber einer bestimmten zeit?
        feats = feature_collection.compute_features(event.answerer_user_id, suggestable_questions, event.answer_date)
        label = suggestable_questions.question_id.values == event.question_id
        
        # add some more information
        feats["question_id"] = suggestable_questions.question_id.values.tolist() # remember question ids
        feats["decision_time"] = q_a_pair_counter # for MRR need to remember groups
        feats["label"] = label.astype(int)
        feats["answer_date"] = pd.Series([event.answer_date for _ in range(len(feats))])

        all_feates_collector.append(feats)

        assert(np.sum(np.asarray(label).astype(int))==1)
        

    feature_collection.update_pos_event(event) # update features in any case

    # save inbetween and clear variables in order to backup
    if (q_a_pair_counter+1) % save_every==0:
        save_name = "feature_data_"+str((q_a_pair_counter+1)//save_every)+".csv"
        features_table = pd.concat(all_feates_collector, axis=0)
        features_table.to_csv(os.path.join(save_dir, save_name), index=False)
        print("Successfully saved data inbetween", save_name)
        del features_table 
        del all_feates_collector
        all_feates_collector = list()

    
    q_a_pair_counter+=1

if (q_a_pair_counter+1) % save_every != 0: # last batch hasn't been saved
    save_name = "feature_data_"+str((q_a_pair_counter+1)//save_every + 1)+".csv"
    features_table = pd.concat(all_feates_collector, axis=0)
    features_table.to_csv(os.path.join(save_dir, save_name), index=False)
print("Successfully saved last data")
print("Finished")