from functools import reduce


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


def make_pairs(question_stream,
               question_features_to_use_for_similarity,
               question_start_time,
               group_column_name,
               db_access
               ):
    # TODO
    # make sure questions are sorted by date

    # go through questions in tern
    # take all questions before/above the current one
    # assert that questions above are all before and below are all after
    # do similarity
    # get all answerers according to strategy

    # get actuall answerer
