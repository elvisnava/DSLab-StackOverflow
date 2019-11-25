
import pandas
import data
import data_utils

cached_data = data.DataHandleCached()

what_algo_observed = pd.DataFrame()
for i, event in enumerate(data_utils.user_answers_young_question_event_iterator_with_candidates(cached_data, 5, start_time = data_utils.make_datetime("01.01.2014 00:01"))):
    (event_date, user_id, actually_answered_id, young_open_questions_at_the_time) = event

    # compute features for all young_open_question_at_the_time candidates using what_algo_observed

    # use GP to predict scores for all candidates

    # pick top N \mu + \sigma candidates

    # add candidates together with (candidate == actually_answered_id) as label to what_algo_observed list

    # fit GP on new what_algo_observed list

    if i > 300:
        break