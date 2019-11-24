import numpy as np
import pandas as pd
import scipy
from datetime import date, datetime, timedelta
import re
import os
import pickle

from sklearn.pipeline import Pipeline, make_pipeline
from pipeline_utils import NamedColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pyltr

from data import Data, GetAnswerersStrategy
from data_utils import Time_Binned_Features, make_datetime
import features
import utils

import custom_lda

cache_dir = "../cache/"
load_question_features = True
print_ttm_topics = False
raw_question_features_path = os.path.join(cache_dir, "raw_question_features.pickle")
load_feat_pairs = True
train_qu_pairs_dataframe_path = os.path.join(cache_dir, "train_qu_pairs_dataframe.pickle")
train_qu_targets_path = os.path.join(cache_dir, "train_qu_targets.pickle")
train_qu_qids_path = os.path.join(cache_dir, "train_qu_qids.pickle")
test_qu_pairs_dataframe_path = os.path.join(cache_dir, "test_qu_pairs_dataframe.pickle")
test_qu_targets_path = os.path.join(cache_dir, "test_qu_targets.pickle")
test_qu_qids_path = os.path.join(cache_dir, "test_qu_qids.pickle")

use_all_users_in_train = True

training_questions_start_time = make_datetime("01.01.2015 00:00")
training_questions_end_time = make_datetime("01.06.2016 00:01")
testing_questions_start_time = make_datetime("01.06.2016 00:02")
testing_questions_end_time = make_datetime("31.12.2016 23:59")

db_access = Data(verbose=3)

#TTM FEATURES
if load_question_features:
    all_questions_features = pd.read_pickle(raw_question_features_path)
else:
    #SET UP TTM
    db_access.set_time_range(start=None, end=training_questions_end_time)
    posts_for_fitting_ttm = db_access.query("SELECT Id as Question_Id, Body, Tags, CreationDate FROM Posts WHERE PostTypeId = {questionPostType}", use_macros=True)

    words_vectorizer = CountVectorizer(stop_words='english')
    words_pipeline = Pipeline([ ## start text pipline
        ("remove_html", features.RemoveHtmlTags()),
        ("replace_numbers", features.ReplaceNumbers()),
        ("unpack", FunctionTransformer(np.squeeze, validate=False)), # this is necessary cause the vectorizer does not expect 2d data
        ("words_vectorize", words_vectorizer)],
        verbose=True)

    tags_vectorizer = CountVectorizer(token_pattern=r'<.*?>')
    n_tags = tags_vectorizer.fit_transform(posts_for_fitting_ttm['tags']).shape[1]

    ttm_pipeline = Pipeline([
        ('tagword_transf', ColumnTransformer([
                                ('tags_pipeline', CountVectorizer(token_pattern=r'<.*?>'), 'tags'),
                                ('words_pipeline', words_pipeline, 'body')
                                ],
                            verbose=True)),
        ('ttm', custom_lda.TTM(n_tags=n_tags, n_topics=10, n_iter=500))],
        memory=cache_dir+"ttm", verbose=True)

    question_feature_pipeline = NamedColumnTransformer([
        ('question_id', None,  "question_id"),
        ('topic[10]', ttm_pipeline, ["tags", "body"]),
        ('creationdate', None, 'creationdate')
    ])

    #FIT TTM
    question_feature_pipeline.fit(posts_for_fitting_ttm)

    #Compute question features
    db_access.set_time_range(start=None, end=testing_questions_end_time)
    all_questions = db_access.query("SELECT Id as Question_Id, Body, Tags, CreationDate FROM Posts WHERE PostTypeID = {questionPostType}", use_macros=True)
    # I could prefilter here to only take answered questions -> save computation
    # subs = question_feature_pipeline.transform_df(all_questions[:100])

    all_questions_features = question_feature_pipeline.transform_df(all_questions)

    all_questions_features.to_pickle(raw_question_features_path)
print("finished question features")

#For debugging the topics
if print_ttm_topics:
    w_vect = question_feature_pipeline.named_transformers_["topic[10]"].named_steps['tagword_transf'].named_transformers_['words_pipeline'].named_steps['words_vectorize']
    t_vect = question_feature_pipeline.named_transformers_["topic[10]"].named_steps['tagword_transf'].named_transformers_['tags_pipeline']
    ttm_obj = question_feature_pipeline.named_transformers_["topic[10]"].named_steps['ttm']

    top_n_words_by_t = utils.top_n_words_by_topic(w_vect, ttm_obj, 10, words_or_tags='words')
    print('Top 10 words by topic')
    for i in range(len(top_n_words_by_t)):
        print('Topic {}: {}'.format(i, ' '.join(top_n_words_by_t[i])))
    top_n_tags_by_t = utils.top_n_words_by_topic(t_vect, ttm_obj, 10, words_or_tags='tags')
    print('Top 10 tags by topic')
    for i in range(len(top_n_tags_by_t)):
        print('Topic {}: {}'.format(i, ' '.join(top_n_tags_by_t[i])))

###############################################################################
#Build user features and question-user feature pairs by going through questions
###############################################################################
if load_feat_pairs:
    train_qu_pairs_dataframe = pd.read_pickle(train_qu_pairs_dataframe_path)
    train_qu_targets = pickle.load(open(train_qu_targets_path, mode='rb'))
    train_qu_qids = pickle.load(open(train_qu_qids_path, mode='rb'))
    test_qu_pairs_dataframe = pd.read_pickle(test_qu_pairs_dataframe_path)
    test_qu_targets = pickle.load(open(test_qu_targets_path, mode='rb'))
    test_qu_qids = pickle.load(open(test_qu_qids_path, mode='rb'))
else:
    #Get active users
    db_access.set_time_range(start=None, end=testing_questions_end_time)
    active_users = set(db_access.query("""
                    SELECT OwnerUserId AS Id
                    FROM Posts
                    GROUP BY OwnerUserId
                    HAVING COUNT(Posts.Id) >= 80
                    ORDER BY OwnerUserId
                    """, use_macros=True)['id'][1:-1])

    users_coupe_feats = dict()
    train_quest_user_pairs = []
    test_quest_user_pairs = []
    train_quest_user_targets = []
    test_quest_user_targets = []
    train_qids = []
    test_qids = []

    question_answer_pairs = db_access.query("""
                            SELECT Q.Id as question_id, A.Id as answer_id, A.OwnerUserId as answerer_user_id, A.score as answer_score
                            FROM Posts A INNER JOIN Posts Q on A.ParentId = Q.Id
                            """)

    sorted_ids = all_questions_features.creationdate.argsort()
    assert(np.all(sorted_ids == np.arange(len(all_questions_features))))

    id_of_first_question = all_questions_features.creationdate.searchsorted(training_questions_start_time)[0]
    assert(all_questions_features.creationdate[id_of_first_question] >= training_questions_start_time)

    id_of_begin_test = all_questions_features.creationdate.searchsorted(testing_questions_start_time)[0]
    id_of_end_test = all_questions_features.creationdate.searchsorted(testing_questions_end_time)[0]

    #for id in range(sorted_ids[0], id_of_end_test):
    #For debug or quicker computation, start from id_of_first_question, for full user COUPE training start from sorted_ids[0]
    #For default end, id_of_end_test
    start_id = sorted_ids[0]
    end_id = id_of_end_test
    tot_len_ans = 0
    n_len_ans = 0
    for id in range(start_id, end_id):
        if id%50==0:
            frac_done = (id-start_id)/(end_id-start_id)
            print("Make Pairs at {:.1f} %".format(frac_done*100))

        current_target_question = all_questions_features.iloc[id]
        curr_question_topics = list(current_target_question[['topic_{}'.format(i) for i in range(10)]])
        answers = question_answer_pairs[question_answer_pairs.question_id == current_target_question.question_id]
        #Filter by active users
        answers = answers[answers['answerer_user_id'].isin(active_users)]
        #DEBUG: try only with more than 1 ans
        if len(answers) < 2:
            continue
        tot_len_ans += len(answers)
        n_len_ans += 1
        #If in training period, build COUPE features and question-user pairs
        if id < id_of_begin_test:
            ans_targets = []
            if not use_all_users_in_train:
                for ans_id, ans in answers.iterrows():
                    usr_id = ans.answerer_user_id

                    #Build question-user pair features using existing non-aggregated COUPE features (only from actual training onwards)
                    if id >= id_of_first_question:
                        if usr_id in users_coupe_feats:
                            q_u_pre_agg = np.array([np.array(v[:4]) * (1 - scipy.spatial.distance.jensenshannon(curr_question_topics, v[4:])) for v in users_coupe_feats[usr_id]])
                            q_u = {'wins_mean': np.mean(q_u_pre_agg[:,0]), 'wins_sd': np.std(q_u_pre_agg[:,0]), 'wins_sum': np.sum(q_u_pre_agg[:,0]), 'wins_max': np.max(q_u_pre_agg[:,0]), 'wins_min': np.min(q_u_pre_agg[:,0]),
                                   'ties_mean': np.mean(q_u_pre_agg[:,1]), 'ties_sd': np.std(q_u_pre_agg[:,1]), 'ties_sum': np.sum(q_u_pre_agg[:,1]), 'ties_max': np.max(q_u_pre_agg[:,1]), 'ties_min': np.min(q_u_pre_agg[:,1]),
                                   'losses_mean': np.mean(q_u_pre_agg[:,2]), 'losses_sd': np.std(q_u_pre_agg[:,2]), 'losses_sum': np.sum(q_u_pre_agg[:,2]), 'losses_max': np.max(q_u_pre_agg[:,2]), 'losses_min': np.min(q_u_pre_agg[:,2]),
                                   'votes_mean': np.mean(q_u_pre_agg[:,3]), 'new': 0
                                  }
                        else:
                            q_u = {'wins_mean': 0, 'wins_sd': 0, 'wins_sum': 0, 'wins_max': 0, 'wins_min': 0,
                                   'ties_mean': 0, 'ties_sd': 0, 'ties_sum': 0, 'ties_max': 0, 'ties_min': 0,
                                   'losses_mean': 0, 'losses_sd': 0, 'losses_sum': 0, 'losses_max': 0, 'losses_min': 0,
                                   'votes_mean': 0, 'new': 1
                                  }
                        train_quest_user_pairs.append(pd.Series(q_u))
                        #train_quest_user_targets.append(ans.answer_score)
                        ans_targets.append(ans.answer_score)
                        train_qids.append(id)

                    #Build user COUPE features (only while training)
                    wins, ties, losses = 0, 0, 0
                    other_answers = answers[answers.answerer_user_id != usr_id]
                    for oth_ans_id, oth_ans in other_answers.iterrows():
                        if ans.answer_score > oth_ans.answer_score:
                            wins += 1
                        elif ans.answer_score == oth_ans.answer_score:
                            ties += 1
                        else:
                            losses += 1
                    if len(other_answers) > 0:
                        sing_feat_vector = [wins/len(other_answers), ties/len(other_answers), losses/len(other_answers), ans.answer_score] + curr_question_topics
                    else:
                        sing_feat_vector = [0,0,0,ans.answer_score] + curr_question_topics
                    if usr_id not in users_coupe_feats:
                        users_coupe_feats[usr_id] = [sing_feat_vector]
                    else:
                        users_coupe_feats[usr_id].append(sing_feat_vector)
            else:
                for usr_id in active_users:
                    if usr_id in list(answers.answerer_user_id):
                        ans_score = float(answers[answers['answerer_user_id'] == usr_id].answer_score.iloc[0])
                    else:
                        ans_score = 0.0
                    #Build question-user pair features using existing non-aggregated COUPE features (only from actual training onwards)
                    if id >= id_of_first_question:
                        if usr_id in users_coupe_feats:
                            q_u_pre_agg = np.array([np.array(v[:4]) * (1 - scipy.spatial.distance.jensenshannon(curr_question_topics, v[4:])) for v in users_coupe_feats[usr_id]])
                            q_u = {'wins_mean': np.mean(q_u_pre_agg[:,0]), 'wins_sd': np.std(q_u_pre_agg[:,0]), 'wins_sum': np.sum(q_u_pre_agg[:,0]), 'wins_max': np.max(q_u_pre_agg[:,0]), 'wins_min': np.min(q_u_pre_agg[:,0]),
                                   'ties_mean': np.mean(q_u_pre_agg[:,1]), 'ties_sd': np.std(q_u_pre_agg[:,1]), 'ties_sum': np.sum(q_u_pre_agg[:,1]), 'ties_max': np.max(q_u_pre_agg[:,1]), 'ties_min': np.min(q_u_pre_agg[:,1]),
                                   'losses_mean': np.mean(q_u_pre_agg[:,2]), 'losses_sd': np.std(q_u_pre_agg[:,2]), 'losses_sum': np.sum(q_u_pre_agg[:,2]), 'losses_max': np.max(q_u_pre_agg[:,2]), 'losses_min': np.min(q_u_pre_agg[:,2]),
                                   'votes_mean': np.mean(q_u_pre_agg[:,3]), 'new': 0
                                  }
                        else:
                            q_u = {'wins_mean': 0, 'wins_sd': 0, 'wins_sum': 0, 'wins_max': 0, 'wins_min': 0,
                                   'ties_mean': 0, 'ties_sd': 0, 'ties_sum': 0, 'ties_max': 0, 'ties_min': 0,
                                   'losses_mean': 0, 'losses_sd': 0, 'losses_sum': 0, 'losses_max': 0, 'losses_min': 0,
                                   'votes_mean': 0, 'new': 1
                                  }
                        train_quest_user_pairs.append(pd.Series(q_u))
                        ans_targets.append(ans_score)
                        train_qids.append(id)

                    #Build user COUPE features (only while training)
                    if usr_id in list(answers.answerer_user_id):
                        wins, ties, losses = 0, 0, 0
                        other_answers = answers[answers.answerer_user_id != usr_id]
                        for oth_ans_id, oth_ans in other_answers.iterrows():
                            if ans_score > oth_ans.answer_score:
                                wins += 1
                            elif ans_score == oth_ans.answer_score:
                                ties += 1
                            else:
                                losses += 1
                        if len(other_answers) > 0:
                            sing_feat_vector = [wins/len(other_answers), ties/len(other_answers), losses/len(other_answers), ans_score] + curr_question_topics
                        else:
                            sing_feat_vector = [0,0,0,ans_score] + curr_question_topics
                        if usr_id not in users_coupe_feats:
                            users_coupe_feats[usr_id] = [sing_feat_vector]
                        else:
                            users_coupe_feats[usr_id].append(sing_feat_vector)

            if id >= id_of_first_question:
                min_ans_score = min(ans_targets)
                train_quest_user_targets += [v - min_ans_score for v in ans_targets]

        #If in testing period, use all users for which we have COUPE feats (TODO: change to include all users active in timeframe, also new)
        else:
            ans_targets = []
            for usr_id in active_users:
                if usr_id in users_coupe_feats:
                    q_u_pre_agg = np.array([np.array(v[:4]) * (1 - scipy.spatial.distance.jensenshannon(curr_question_topics, v[4:])) for v in users_coupe_feats[usr_id]])
                    q_u = {'wins_mean': np.mean(q_u_pre_agg[:,0]), 'wins_sd': np.std(q_u_pre_agg[:,0]), 'wins_sum': np.sum(q_u_pre_agg[:,0]), 'wins_max': np.max(q_u_pre_agg[:,0]), 'wins_min': np.min(q_u_pre_agg[:,0]),
                           'ties_mean': np.mean(q_u_pre_agg[:,1]), 'ties_sd': np.std(q_u_pre_agg[:,1]), 'ties_sum': np.sum(q_u_pre_agg[:,1]), 'ties_max': np.max(q_u_pre_agg[:,1]), 'ties_min': np.min(q_u_pre_agg[:,1]),
                           'losses_mean': np.mean(q_u_pre_agg[:,2]), 'losses_sd': np.std(q_u_pre_agg[:,2]), 'losses_sum': np.sum(q_u_pre_agg[:,2]), 'losses_max': np.max(q_u_pre_agg[:,2]), 'losses_min': np.min(q_u_pre_agg[:,2]),
                           'votes_mean': np.mean(q_u_pre_agg[:,3]), 'new': 0
                          }
                else:
                    q_u = {'wins_mean': 0, 'wins_sd': 0, 'wins_sum': 0, 'wins_max': 0, 'wins_min': 0,
                           'ties_mean': 0, 'ties_sd': 0, 'ties_sum': 0, 'ties_max': 0, 'ties_min': 0,
                           'losses_mean': 0, 'losses_sd': 0, 'losses_sum': 0, 'losses_max': 0, 'losses_min': 0,
                           'votes_mean': 0, 'new': 1
                          }
                test_quest_user_pairs.append(pd.Series(q_u))
                if usr_id in list(answers.answerer_user_id):
                    #test_quest_user_targets.append(float(answers[answers['answerer_user_id'] == usr_id].answer_score.iloc[0]))
                    ans_targets.append(float(answers[answers['answerer_user_id'] == usr_id].answer_score.iloc[0]))
                else:
                    #test_quest_user_targets.append(0.0)
                    ans_targets.append(0.0)
                test_qids.append(id)
            min_ans_score = min(ans_targets)
            test_quest_user_targets += [v - min_ans_score for v in ans_targets]

    print("Average number of answers for question: {}".format(tot_len_ans/n_len_ans))

    train_qu_pairs_dataframe = pd.concat(train_quest_user_pairs, axis=1).T
    train_qu_targets = np.array(train_quest_user_targets)
    train_qu_qids = np.array(train_qids)
    test_qu_pairs_dataframe = pd.concat(test_quest_user_pairs, axis=1).T
    test_qu_targets = np.array(test_quest_user_targets)
    test_qu_qids = np.array(test_qids)

    train_qu_pairs_dataframe.to_pickle(train_qu_pairs_dataframe_path)
    pickle.dump(train_qu_targets, open(train_qu_targets_path, mode='wb'))
    pickle.dump(train_qu_qids, open(train_qu_qids_path, mode='wb'))
    test_qu_pairs_dataframe.to_pickle(test_qu_pairs_dataframe_path)
    pickle.dump(test_qu_targets, open(test_qu_targets_path, mode='wb'))
    pickle.dump(test_qu_qids, open(test_qu_qids_path, mode='wb'))

#Max target for MRR
#max_target = np.max(np.concatenate((train_qu_targets,test_qu_targets)))

print("Train shape: {}".format(train_qu_pairs_dataframe.shape))
print("Test shape: {}".format(test_qu_pairs_dataframe.shape))

metric = pyltr.metrics.dcg.NDCG(k=10)
#metric = pyltr.metrics.err.ERR(max_target, k=10)
model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=100,
    learning_rate=0.02,
#    max_features=0.5,
#    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)
model.fit(train_qu_pairs_dataframe, train_qu_targets, train_qu_qids)
pred_targets = model.predict(test_qu_pairs_dataframe)
print(metric.calc_mean_random(test_qu_qids, test_qu_targets))
print(metric.calc_mean(test_qu_qids, test_qu_targets, pred_targets))

chance_mrr = utils.mrr3(out_probs=np.random.permutation(pred_targets), grouped_queries=test_qu_qids, ground_truth=test_qu_targets)
mrr_score = utils.mrr3(out_probs=pred_targets, grouped_queries=test_qu_qids, ground_truth=test_qu_targets)
print("Chance MRR: {}".format(chance_mrr))
print("MRR: {}".format(mrr_score))
