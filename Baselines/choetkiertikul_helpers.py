from functools import reduce
import numpy as np
import utils
import pandas as pd
import sklearn
import time


def get_user_data(db_access):
    date_now = db_access.end

    date_string = str(date_now)

    basic_user_data = db_access.query("SELECT Id as User_Id, CreationDate, Reputation as _reputation_global, date '{}' - CreationDate AS PlattformAge from Users ORDER BY Id".format(date_string)).set_index("user_id", drop=False)

    n_question_for_user = db_access.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as NumberQuestions from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {questionPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True).set_index("user_id")
    n_answers_for_user = db_access.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as  NumberAnswers from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {answerPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True).set_index("user_id")

    n_accepted_answers_query = """SELECT A.OwnerUserId as User_Id, count(A.Id) as NumberAcceptedAnswers from Posts Q LEFT JOIN Posts A on Q.AcceptedAnswerId = A.Id WHERE Q.AcceptedAnswerId IS NOT NULL AND A.OwnerUserId IS NOT NULL GROUP BY A.OwnerUserId ORDER BY A.OwnerUserId"""

    n_accepted_answers = db_access.query(n_accepted_answers_query).set_index("user_id")

    query_votes = """SELECT P.OwnerUserId as user_id, sum((V.VoteTypeId = 2)::int) as upvotes, sum((V.VoteTypeId=3)::int) as downvotes
        FROM Votes as V join Posts P on V.PostId = P.Id 
        GROUP BY P.OwnerUserId
    """
    votes = db_access.query(query_votes).set_index("user_id")

    reputation = db_access.user_reputations().set_index("user_id")

    user_tags = (db_access.get_user_tags()[["user_id", "user_tags"]]).set_index("user_id")

    all_data_sources = [basic_user_data, n_question_for_user, n_answers_for_user, n_accepted_answers, user_tags, votes, reputation]
    final = reduce(lambda a,b: a.join(b), all_data_sources)

    return final


def make_pairs(question_stream, # dataframe with all questions (including testing)
               question_features_to_use_for_similarity, # name of columns to use for similarity
               question_start_time, # question
               group_column_name, # name of (in this case) prevalent topic column || None -> find similar context question among all questions
               answerers_strategy,
               n_candidate_questions,
               user_features, #  a Time_Binned_Features instance
               similarity_measure = "cosine"
               ):
    stats = dict(ignored_questions = 0, used_questions = 0, manually_added_users_unkown_at_last_intervall=0)
    print("Starting make pairs")

    sorted_ids = question_stream.creationdate.argsort()

    assert(np.all(sorted_ids == np.arange(len(question_stream))))

    id_of_first_question = question_stream.creationdate.searchsorted(question_start_time)

    assert(question_stream.creationdate[id_of_first_question] >= question_start_time)

    collector = list()

    for id in range(id_of_first_question, len(question_stream)):
        if id%10==0:
            frac_done = (id-id_of_first_question)/(len(question_stream)-id_of_first_question)
            print("Make Pairs at {:.1f} %".format(frac_done*100))


        current_target_question = question_stream.iloc[id]
        actuall_answerers = answerers_strategy.get_answerers_set(question_ids=[current_target_question.question_id], before_timepoint=None)
        if len(actuall_answerers) == 0:
            stats['ignored_questions'] += 1
            continue


        context_questions = question_stream.iloc[:(id-1), :]

        assert(current_target_question.question_id not in list(context_questions.question_id))


        if group_column_name is not None:
            target_group_id = current_target_question[group_column_name]
            context_questions_in_group = context_questions[context_questions[group_column_name] == target_group_id]
        else:
            context_questions_in_group = context_questions

        target_question_featues = np.array(current_target_question[question_features_to_use_for_similarity].values)[None, :]
        # TODO at the mmoment we take all similar questions (i.e. also unanswered ones)
        context_questions_features = context_questions_in_group[question_features_to_use_for_similarity].values
        closest_ids = np.squeeze(utils.get_closest_n(source_features=target_question_featues, context_features=context_questions_features, n=n_candidate_questions, metric=similarity_measure))

        selected_context_questions = context_questions_in_group.iloc[closest_ids]

        answerer_users_candidates = answerers_strategy.get_answerers_set(question_ids=selected_context_questions.question_id, before_timepoint=current_target_question.creationdate)

        incorrectly_picked_candidates, correctly_picked_candidates, manually_added_cadidates = utils.set_partitions(answerer_users_candidates, actuall_answerers)
        all_answerers = list(incorrectly_picked_candidates) + list(correctly_picked_candidates) + list(manually_added_cadidates)
        label = ([False]*len(incorrectly_picked_candidates) + [True]*len(correctly_picked_candidates) + [True]*len(manually_added_cadidates))
        type_of_answerer = (['incorrect_pick']*len(incorrectly_picked_candidates)) + (['correct_pick']*len(correctly_picked_candidates)) + (['manually_added']*len(manually_added_cadidates))
        df_dict = dict(answerer_id = all_answerers, type_of_answerer=type_of_answerer, label=label)
        df_dict.update(dict(current_target_question))
        df_dict['user_features_age'] = user_features.age_of_data(current_target_question.creationdate)
        question_with_answerers = pd.DataFrame(df_dict)

        # Now get the actuall features of the answerers
        all_user_features_this_time = user_features[current_target_question.creationdate]

        answerer_features_this_time = all_user_features_this_time.loc[all_answerers, :]

        if not np.all(np.isin(all_answerers, all_user_features_this_time.index)):
            stats['manually_added_users_unkown_at_last_intervall'] += 1


        finished_pairs = question_with_answerers.merge(answerer_features_this_time, left_on="answerer_id", right_index=True, suffixes=("_question", "_user"), how="left")
        collector.append(finished_pairs)

        stats['used_questions'] += 1

        # this is for breaking for debugging
        # if stats['used_questions'] >= 20:
        #     break


    all_samples = pd.concat(collector, axis=0)

    print("Making Pairs fun facts : {}".format(stats))

    return all_samples



    # go through questions in tern
    # take all questions before/above the current one
    # assert that questions above are all before and below are all after
    # do similarity
    # get all answerers according to strategy

    # get actuall answerer

def overview_score(y_true, y_hat, group, label=None):
    assert(y_hat.dtype==np.float)
    y_hat_bin = y_hat >=0.5


    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_hat_bin)
    prec = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_hat_bin)
    rec = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_hat_bin)
    fscore = 2* prec*rec / (prec+rec)


    hist, edges = np.histogram(y_hat, bins=[-0.1, 0.25, 0.75, 1])

    t0 = time.time()
    mrr_score, mrr_ranks = utils.multi_mrr(out_probs=y_hat, grouped_queries=group, ground_truth=y_true)
    mrr_time = time.time() - t0

    all_info = dict(accuracy=acc, precission = prec, recall = rec, fscore = fscore, prediction_values=hist, mrr_score = mrr_score, mrr_time=mrr_time, label=label)

    return all_info, mrr_ranks



def dataframe_to_xy(df, feature_cols):
    y = df["label"].values

    # df["noisy_label"] = y.astype(float) + 0.1 * np.random.rand(len(y))
    # _feature_cols = feature_cols + ["noisy_label"]
    # print("WARN >> added noisy label")
    df.loc[:, "plattformage_seconds"] = df.plattformage.dt.total_seconds()

    actuall_cols = set(df.columns)
    not_found_cols = set(feature_cols) - actuall_cols
    print("Didn't find columns {}".format(not_found_cols))
    assert(len(not_found_cols)==0)
    cols_that_didnt_get_picked = actuall_cols - set(feature_cols)

    print("Used features: {}".format(feature_cols))
    print("Columns that didn't get picked {}".format(cols_that_didnt_get_picked))

    X = df[feature_cols].values.astype(float)
    X = X.astype(float)
    return X, y