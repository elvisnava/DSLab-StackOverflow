
import pandas
import data
import data_utils
import pandas as pd
import gp_features
from datetime import timedelta

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




what_algo_observed = pd.DataFrame()

all_features = gp_features.GP_Feature_Collection() # TODO later all feature instances will go in the constructor here

for i, event in enumerate(data_utils.all_answer_events_iterator(data_handle, start_time=start_time)):

    if not is_user_answers_suggested_event(event):
        all_features.update_event(event)
    else:
        suggestable_questions = get_suggestable_questions(event.answer_date)



    # compute features for all young_open_question_at_the_time candidates using what_algo_observed

    # use GP to predict scores for all candidates

    # pick top N \mu + \sigma candidates

    # add candidates together with (candidate == actually_answered_id) as label to what_algo_observed list

    # fit GP on new what_algo_observed list

    if i > 300:
        break