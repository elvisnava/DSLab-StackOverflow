
import pandas
import data
import data_utils
import utils
import pandas as pd
import gp_features
from datetime import timedelta
import numpy as np
import warnings
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct


start_time_online_learning =  data_utils.make_datetime("01.01.2012 00:01")
hour_threshold_suggested_answer = 24

pretraining_cache_file = "../cache/gp/pretraining.pickle"
redo_pretraining = False

cached_data = data.DataHandleCached()
data_handle = data.Data()

def is_user_answers_suggested_event(event):
    return event.question_age_at_answer <= timedelta(hours=hour_threshold_suggested_answer)

def get_suggestable_questions(time):
    open_questions = cached_data.open_questions_at_time(time)
    mask = (open_questions.question_date >= time - timedelta(hours=hour_threshold_suggested_answer))
    return open_questions[mask]


def argmax_ucb(mu, sigma, beta):
    return np.argmax(mu + sigma * np.sqrt(beta))

def print_intermediate_info(info_dict, current_time):
    if len(info_dict['event_time'])==0:
        print("empty info dict")
        return

    last_n = 100
    avg_candidates = np.mean(np.array(info_dict["n_candidates"])[-last_n:])
    most_recent_time = info_dict['event_time'][-1]

    s = "{} | number of average candidates: {}".format(current_time, avg_candidates)
    print(s)


sigma = 1
beta = 0.4
n_preds = 5


def top_N_ucb(mu, sigma, beta=beta, n=n_preds):
    upper_bounds = mu + sigma * np.sqrt(beta)
    # ids = utils.get_ids_of_N_largest(upper_bounds, n)
    sorted_ids = np.argsort(-upper_bounds)[:n]
    return sorted_ids # first is actually the one with the highest prediction


all_features_collection_raw = gp_features.GP_Feature_Collection(
    gp_features.GP_Features_affinity(),
    # gp_features.GP_Features_TTM()),
    # gp_features.GP_Features_Question(),
    gp_features.GP_Features_user())


def pretrain_gp_ucp(feature_collection, start_time, end_time):

    all_feates_collector = list()
    all_label_collector = list() # list of 1d numpy arrays


    n_candidates_collector = list()

    for i, event in enumerate(data_utils.all_answer_events_iterator(start_time=start_time, end_time=end_time)):
        if i%100 ==0 :
            avg_candidates = np.mean(n_candidates_collector)
            print("Preptraining at {}| on average {} candidates in the last {} suggested_question_events".format(event.answer_date, avg_candidates, len(n_candidates_collector)))
            n_candidates_collector = list()

        if not is_user_answers_suggested_event(event):
            feature_collection.update_pos_event(event)
        else:
            suggestable_questions = get_suggestable_questions(event.answer_date)
            if len(suggestable_questions) ==0:
                warnings.warn("For answer id {} (to question {}) there was not a single suggestable question".format(event.answer_id, event.question_id))
                continue

            n_candidates_collector.append(len(suggestable_questions))

            feats = feature_collection.compute_features(event.answerer_user_id, suggestable_questions, event.answer_date)
            label = suggestable_questions.question_id.values == event.question_id

            all_feates_collector.append(feats)
            all_label_collector.append(label)

            # TODO I don't update the negative event here

            feature_collection.update_pos_event(event)

    all_feats = pd.concat(all_feates_collector, axis=0)
    all_label = np.concatenate(all_label_collector, axis=0).tolist()

    return feature_collection, (all_feats, all_label)

if redo_pretraining:
    pretraining_result = pretrain_gp_ucp(all_features_collection_raw, start_time=None, end_time=start_time_online_learning)
    with open(pretraining_cache_file, "wb") as f:
        pickle.dump(pretraining_result, f)
else:
    with open(pretraining_cache_file, "rb") as f:
        pretraining_result = pickle.load(f)


all_features_collection, (training_set_for_gp, observed_labels ) = pretraining_result

info_dict = {'answer_id': list(), 'event_time': list(), 'user_id': list(), 'n_candidates': list()}

for i, event in enumerate(data_utils.all_answer_events_iterator(data_handle, start_time=start_time_online_learning)):
    if i%1 ==0:
        print_intermediate_info(info_dict, event.answer_date)


    if not is_user_answers_suggested_event(event):
        # Don't just update the coupe, also add to the df as observation
        all_features_collection.update_pos_event(event)
        # TODO: add to what_algo_observed
    else:
        target_user_id = event.answerer_user_id
        actually_answered_id = event.question_id
        event_time = event.answer_date

        suggestable_questions = get_suggestable_questions(event.answer_date)
        if len(suggestable_questions) ==0:
            warnings.warn("For answer id {} (to question {}) there was not a single suggestable question".format(event.answer_id, event.question_id))
            continue

        # compute features
        features = all_features_collection.compute_features(target_user_id, suggestable_questions, event_time)
        # previous version: (I changed it because it is not necessary to give a list of target_user_id)
        # features = all_features_collection.compute_features(len(suggestable_questions)*[target_user_id], suggestable_questions, event_time)
        

        # # fit and predict with gaussian process
        gpr = GaussianProcessRegressor(kernel=DotProduct(), random_state=0).fit(training_set_for_gp.values, observed_labels)
        mu, sigma = gpr.predict(features, return_std=True)
        max_inds = argmax_ucb(mu, sigma, beta) # this is the indexes of the predicted question that the user will answer


        rank_of_true_question = -1

        # update feature database
        for rank, selected_id in enumerate(max_inds):
            actually_suggested_question = suggestable_questions.iloc[selected_id]

            if actually_suggested_question.question_id == actually_answered_id:
                # the suggested question is what was actually answered
                all_features_collection.update_pos_event(event)
                rank_of_true_question = rank
            else:
                # this suggested question was not answered
                all_features_collection.update_neg_event(event) # i think all features so far ignore this


        # update training_data for gaaussian process

        suggested_questions_features = features.iloc[max_inds]
        suggested_questions_label = (suggestable_questions.iloc[max_inds].question_id == actually_answered_id)

        training_set_for_gp = pd.concat([training_set_for_gp, features])
        observed_labels.extend(suggested_questions_label)

        # update info
        info_dict["answer_id"].append(event.answer_id)
        info_dict["event_time"].append(event_time)
        info_dict["user_id"].append(target_user_id)
        info_dict["n_candidates"].append(len(suggestable_questions))


    pass

    if i > 300:
        training_set_for_gp.to_csv("test.csv")
        break
