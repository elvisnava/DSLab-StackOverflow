from data import Data
from datetime import datetime

import pandas as pd
from functools import reduce

from data_utils import Time_Binned_Features, make_datetime



training_questions_start_time = make_datetime("01.01.2015 00:00")
training_questions_end_time = make_datetime("01.06.2016 00:01")
testing_questions_start_time = make_datetime("01.06.2016 00:02")
testing_questions_end_time = make_datetime("31.12.2016 23:59")

n_feature_time_bins = 5


def get_user_data(db_access):
    date_now =  db_access.end

    date_string = str(date_now)

    basic_user_data = db_access.query("SELECT Id as User_Id, CreationDate, Reputation, UpVotes, DownVotes, date '{}' - CreationDate AS PlattformAge from Users ORDER BY Id".format(date_string)).set_index("user_id", drop=False)

    n_question_for_user = db_access.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as NumberQuestions from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {questionPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True).set_index("user_id")
    n_answers_for_user = db_access.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as  NumberAnswers from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {answerPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True).set_index("user_id")

    n_accepted_answers_query = """SELECT A.OwnerUserId as User_Id, count(A.Id) as NumberAcceptedAnswers from Posts Q LEFT JOIN Posts A on Q.AcceptedAnswerId = A.Id WHERE Q.AcceptedAnswerId IS NOT NULL AND A.OwnerUserId IS NOT NULL GROUP BY A.OwnerUserId ORDER BY A.OwnerUserId"""

    n_accepted_answers = db_access.query(n_accepted_answers_query).set_index("user_id")

    user_tags = (db_access.get_user_tags()[["user_id", "user_tags"]]).set_index("user_id")

    all_data_sources = [basic_user_data, n_question_for_user, n_answers_for_user, n_accepted_answers, user_tags]
    final = reduce(lambda a,b: a.join(b), all_data_sources)

    return final

db_access = Data(verbose=3)

user_features = Time_Binned_Features(db_access=db_access, gen_features_func=get_user_data, start_time=training_questions_start_time, end_time=testing_questions_end_time, n_bins=n_feature_time_bins, verbose=1)

u1 = user_features[datetime(year=2015, month=3, day=2)]
u2 = user_features["20.11.2016 13:03"]



# define times
# fit LDA
# training questions
# testing questions
# number of user feature buckets







# compute user features in thesse buckets




# compute question features global (double check)
# fit on LDA fit questions



# go through questions and get users. (all or just best answer)
# get and make the pairs, annotate where they come from