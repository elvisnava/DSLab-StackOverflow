
import pandas
import data
import data_utils
import pandas as pd
from datetime import timedelta

start_time =  data_utils.make_datetime("01.01.2014 00:01")


def is_user_answers_suggested_event(event):
    return event.question_age_at_answer <= timedelta(hours=5)


cached_data = data.DataHandleCached()
data_handle = data.Data()

what_algo_observed = pd.DataFrame()

for i, event in enumerate(data_utils.all_answer_events_iterator(data_handle, start_time=start_time)):

    if is_user_answers_suggested_event(event):
        pass
    else:
        pass

    # compute features for all young_open_question_at_the_time candidates using what_algo_observed

    # use GP to predict scores for all candidates

    # pick top N \mu + \sigma candidates

    # add candidates together with (candidate == actually_answered_id) as label to what_algo_observed list

    # fit GP on new what_algo_observed list

    if i > 300:
        break