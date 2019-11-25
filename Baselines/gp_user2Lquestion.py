

import data
import data_utils

cached_data = data.DataHandleCached()

for i, event in enumerate(data_utils.user_answers_young_question_event_iterator_with_candidates(cached_data, 5, start_time = data_utils.make_datetime("01.01.2014 00:01"))):
    (event_date, user_id, actually_answered_id, young_open_questions_at_the_time) = event
    if i > 300:
        break