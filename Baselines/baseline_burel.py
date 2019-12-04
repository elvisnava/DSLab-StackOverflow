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


cached_data = data.DataHandleCached()
data_handle = data.Data()

def is_user_answers_suggested_event(event):
    return event.question_age_at_answer <= timedelta(hours=hour_threshold_suggested_answer)

def get_suggestable_questions(time):
    open_questions = cached_data.open_questions_at_time(time)
    mask = (open_questions.question_date >= time - timedelta(hours=hour_threshold_suggested_answer))
    return open_questions[mask]

feature_collection = gp_features.GP_Feature_Collection(
gp_features.GP_Features_affinity(),
gp_features.GP_Features_Question(),
gp_features.GP_Features_user())

hour_threshold_suggested_answer = 24

start_time = None # data_utils.make_datetime("01.01.2012 00:01")
end_time = data_utils.make_datetime("01.01.2014 00:01") # data_utils.make_datetime("01.03.2012 00:01")

all_feates_collector = list()
all_label_collector = list() # list of 1d numpy arrays

n_candidates_collector = list()

q_a_pair_counter = 1

for i, event in enumerate(data_utils.all_answer_events_iterator(start_time=start_time, end_time=end_time)):
    if np.isnan(event.answerer_user_id) or np.isnan(event.asker_user_id):
         continue
    # print(q_a_pair_counter, " question_id", event.question_id, "answer_id", event.answer_id, "user_id", event.answerer_user_id, "asker_id:", event.asker_user_id)
    
    if i%100 ==0 :
        avg_candidates = np.mean(n_candidates_collector)
        print("Preptraining at {}| on average {} candidates in the last {} suggested_question_events".format(event.answer_date, avg_candidates, len(n_candidates_collector)))
        n_candidates_collector = list()

    if is_user_answers_suggested_event(event):
        
        suggestable_questions = get_suggestable_questions(event.answer_date)
        if len(suggestable_questions) ==0:
            # warnings.warn("For answer id {} (to question {}) there was not a single suggestable question".format(event.answer_id, event.question_id))
            continue
            
        n_candidates_collector.append(len(suggestable_questions))
        
        # erst appenden wenn ueber einer bestimmten zeit?
        feats = feature_collection.compute_features(event.answerer_user_id, suggestable_questions, event.answer_date)
        label = suggestable_questions.question_id.values == event.question_id
        if any(label):
            # print("suggested questions:", suggestable_questions.question_id.values)
            # add some more information
            feats["question_id"] = suggestable_questions.question_id.values.tolist() # remember question ids
            feats["decision_time"] = q_a_pair_counter # for MRR need to remember groups
            feats["label"] = label.astype(int)
            # TODO: add question-answer date such that later we can decide what is the train and what is the test set

            all_feates_collector.append(feats)
            # all_label_collector.append(label)
            # print(np.asarray(label).astype(int))
            # assert(np.sum(np.asarray(label).astype(int))==1)

            q_a_pair_counter+=1
        # print("(not in 24h)")
        
    # print("asker user id:", event.asker_user_id, "answerer userid:", event.answerer_user_id, "dict of users:", feature_collection.features[2].user_features)
    #     # todo: answerer id abspeichern
    # print("tags:", feature_collection.features[0].user_tags)
    # print("-----------------------------")

    feature_collection.update_pos_event(event) # update features in any case

    if q_a_pair_counter>100:
        break
    

features_table = pd.concat(all_feates_collector, axis=0)
features_table.to_csv("test2.csv", index=False)