
import pandas
import data
import data_utils
import pandas as pd
import gp_features
from datetime import timedelta

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

what_algo_observed = pd.DataFrame()
observed_labels = []
mu = 0 # TODO
sigma = 1

all_features_collection = gp_features.GP_Feature_Collection(gp_features.GP_Features_affinity(), gp_features.GP_Features_Question(), gp_features.GP_Features_user())
    # gp_features.GP_Features_TTM())


for i, event in enumerate(data_utils.all_answer_events_iterator(data_handle, start_time=start_time)):

    if not is_user_answers_suggested_event(event):
        # Don't just update the coupe, also add to the df as observation
        all_features_collection.update_event(event)
        # TODO: add to what_algo_observed
    else:
        target_user_id = event.answerer_user_id
        actually_answered_id = event.question_id
        event_time = event.answer_date

        suggestable_questions = get_suggestable_questions(event.answer_date)

        label = (suggestable_questions.question_id == actually_answered_id)
        # compute features
        features = all_features_collection.compute_features(target_user_id, suggestable_questions, event_time)
        # previous version: (I changed it because it is not necessary to give a list of target_user_id)
        # features = all_features_collection.compute_features(len(suggestable_questions)*[target_user_id], suggestable_questions, event_time)
        

        # # fit and predict with gaussian process
        # gpr = GaussianProcessRegressor(kernel=DotProduct(), random_state=0).fit(what_algo_observed, observed_labels)
        # mu, sigma = gp.predict(features, return_std=True)
        # max_ind = argmax_ucb(mu, sigma, beta) # this is the index of the predicted question that the user will answer

        # update features with new event (ONLY IF it is pos)?
        all_features_collection.update_event(event)

        # add features and labels to observed data
        what_algo_observed = pd.concat([what_algo_observed, features])
        observed_labels.extend(label) # this probably doesn't make sense
        


    if i > 300:
        what_algo_observed.to_csv("test.csv")
        break
