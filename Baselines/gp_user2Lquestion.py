
import pandas
import data
import data_utils
import utils
import pandas as pd
import gp_features
from datetime import timedelta
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct


start_time =  data_utils.make_datetime("01.01.2014 00:01")
hour_threshold_suggested_answer = 5

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


training_set_for_gp = pd.DataFrame()
observed_labels = []
mu = 0 # TODO
sigma = 1
beta = 0.4
n_preds = 5


def top_N_ucb(mu, sigma, beta=beta, n=n_preds):
    upper_bounds = mu + sigma * np.sqrt(beta)
    # ids = utils.get_ids_of_N_largest(upper_bounds, n)
    sorted_ids = np.argsort(-upper_bounds)[:n]
    return sorted_ids # first is actually the one with the highest prediction


all_features_collection = gp_features.GP_Feature_Collection(
    gp_features.GP_Features_affinity(),
    # gp_features.GP_Features_TTM()),
    # gp_features.GP_Features_Question(),
    gp_features.GP_Features_user())


for i, event in enumerate(data_utils.all_answer_events_iterator(data_handle, start_time=start_time)):

    if not is_user_answers_suggested_event(event):
        # Don't just update the coupe, also add to the df as observation
        all_features_collection.update_pos_event(event)
        # TODO: add to what_algo_observed
    else:
        target_user_id = event.answerer_user_id
        actually_answered_id = event.question_id
        event_time = event.answer_date

        suggestable_questions = get_suggestable_questions(event.answer_date)

        # compute features
        features = all_features_collection.compute_features(target_user_id, suggestable_questions, event_time)
        # previous version: (I changed it because it is not necessary to give a list of target_user_id)
        # features = all_features_collection.compute_features(len(suggestable_questions)*[target_user_id], suggestable_questions, event_time)
        

        # # fit and predict with gaussian process
        gpr = GaussianProcessRegressor(kernel=DotProduct(), random_state=0).fit(training_set_for_gp, observed_labels)
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


    pass

    if i > 300:
        training_set_for_gp.to_csv("test.csv")
        break
