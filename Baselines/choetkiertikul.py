from sklearn.pipeline import Pipeline, make_pipeline
from pipeline_utils import NamedColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import scipy.spatial
import sklearn.metrics
from functools import reduce


# from lda import LDA
from features import LDAWrapper as LDA

import numpy as np
from datetime import date
from datetime import timedelta
import re

from data import Data
from features import AppendArgmax
import features
import utils
import pandas as pd
import time

load_from_saved = False
grid_search = False
use_topic_affinity = False


data_training = Data(verbose=1, tables_with_time_res=['Posts'])
begin_training_date =date(year=2000, month=1, day=1)
# begin_training_date = date(year=2015, month=6, day=1)
end_training_date = date(year=2015, month=12, day=31)
data_training.set_time_range(start=begin_training_date, end=end_training_date)

data_testing = Data(verbose=0, tables_with_time_res=['Posts'])
begin_testing_date = end_training_date + timedelta(days=1)
end_testing_date = date(year=2016, month=1, day=31)
data_testing.set_time_range(start=begin_training_date, end=end_testing_date)

answer_threshold = None #10
n_context_questions = 30
chance_mrr = np.mean(1/(1+np.arange(n_context_questions)))
print("Chance Mrr for {} candidates is {}".format(n_context_questions, chance_mrr))

n_questions_to_train_on = None #10000

def get_training_questions():
    question_data = data_training.query("SELECT Id as Question_Id, Title, Body, Tags FROM Posts WHERE PostTypeID = {questionPostType}", use_macros=True)
    return question_data

def get_test_questions():
    question_data = data_testing.query("SELECT Id as Question_Id, Title, Body, Tags FROM Posts WHERE PostTypeID = {{questionPostType}} AND CreationDate >= date '{}' ".format(begin_testing_date), use_macros=True)
    return question_data

tq = get_test_questions()

def overview_score(y_true, y_hat, group):
    assert(y_hat.dtype==np.float)
    y_hat_bin = y_hat >=0.5

    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_hat_bin)
    prec = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_hat_bin)
    rec = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_hat_bin)
    fscore = 2* prec*rec / (prec+rec)


    print("We have accuracy={:.2f} , precission={:.2f} , recall={:.2f} , fscore={:.2f}".format(acc, prec, rec, fscore))
    hist, edges = np.histogram(y_hat, bins=[-0.1, 0.25, 0.75, 1])
    print("The numbers of predictions are {}".format(hist))

    t0 = time.time()
    mrr_score, _ranks = utils.mrr(out_probs=y_hat, grouped_queries=group, ground_truth=y_true)
    print("Mrr={:.2f} took {:2.2f} seconds".format(mrr_score, time.time()-t0))



def get_user_data(date_now):
    date_string = str(date_now)

    basic_user_data = data_training.query("SELECT Id as User_Id, CreationDate, Reputation, UpVotes, DownVotes, date '{}' - CreationDate AS PlattformAge from Users ORDER BY Id".format(date_string))

    n_question_for_user = data_training.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as NumberQuestions from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {questionPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True)
    n_answers_for_user = data_training.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as  NumberAnswers from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {answerPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True)

    n_accepted_answers_query = """SELECT A.OwnerUserId as User_Id, count(A.Id) as NumberAcceptedAnswers from Posts Q LEFT JOIN Posts A on Q.AcceptedAnswerId = A.Id WHERE Q.AcceptedAnswerId IS NOT NULL AND A.OwnerUserId IS NOT NULL GROUP BY A.OwnerUserId ORDER BY A.OwnerUserId"""

    n_accepted_answers = data_training.query(n_accepted_answers_query)

    user_tags = data_training.get_user_tags()[["user_id", "user_tags"]]

    all_data_sources = [basic_user_data, n_question_for_user, n_answers_for_user, n_accepted_answers, user_tags]
    final = reduce(lambda a,b: pd.merge(a, b, on="user_id", how="outer"), all_data_sources)
    #assert(np.all(np.isfinite(final.upvotes)))

    return final

def get_similar_questions(source_questions, context_questions, column_names_to_use_as_features, n_context_questions):
    """
    for each query question get a list of context_questions that are similar


    return pd dataframe with source_question_ids and context_question_ids. there are gonna be multiple lines with same source_question_id
    """
    #TODO change this to only consider questions until that point
    # iterate through questions

    topics = np.unique(source_questions.prevalent_topic)

    all_source_question_ids = list()
    all_similar_question_ids = list()

    for topic in topics:
        source_in_topic = source_questions[source_questions.prevalent_topic == topic]
        context_in_topic = context_questions[context_questions.prevalent_topic == topic]

        features_source_in_topic = source_in_topic[column_names_to_use_as_features]
        features_context_in_topic = context_in_topic[column_names_to_use_as_features]

        closest_context_question_ids = utils.get_closest_n(source_features=features_source_in_topic, context_features=features_context_in_topic, source_ids=source_in_topic.question_id.values, context_ids=context_in_topic.question_id.values, n=n_context_questions, metric='cosine', allow_less=True)

        source_question_ids_repeated = np.repeat(source_in_topic.question_id.values[:, None], repeats=closest_context_question_ids.shape[1], axis=1)

        all_source_question_ids.append(source_question_ids_repeated.flatten())
        all_similar_question_ids.append(closest_context_question_ids.flatten())

    result = pd.DataFrame(data=dict(question_id=np.concatenate(all_source_question_ids), similar_question_id=np.concatenate(all_similar_question_ids)))
    assert(np.all(result.question_id != result.similar_question_id))
    return result


def make_sample(target_question_features, users_who_answered_target_questions, context_question_features, users_who_answered_context_questions, user_features):
    """
    we pair each question in question_features with users from users_who_answered_context_questions

    :param target_question_features: training or testing questions
    :param users_who_answered_target_questions: answeres for training or testing questions
    :param context_question_features: features only for training questions (so we cant forsee future during testing)
    :param users_who_answered_context_questions: answerers only for training questions. (could be that somebody later answers a training question outside of our timeperiod)
    :param user_features: users that are candidates for prediction. only users present already during training (for others we wouldn't have a context question that that user answered anyways)
    :return:  each row of target_question_features turns into n_context_questions many rows with different user candidates
    """
    features_cols_for_similarity = ["titleLength", "questionLength", "nCodeBlocks", "nEquationBlocks", "nExternalLinks", "nTags", "readability"]

    assert(len(context_question_features)>=len(users_who_answered_context_questions)) # for each question we have the answer

    question_ids_and_similar_questions = get_similar_questions(target_question_features, context_question_features, column_names_to_use_as_features=features_cols_for_similarity, n_context_questions=n_context_questions)
    question_with_sim_questions = question_ids_and_similar_questions.merge(users_who_answered_context_questions, left_on="similar_question_id", right_on="question_id", how="left").sort_values("question_id")

    question_with_user_candidates = question_with_sim_questions.filter(items=["question_id", "answerer_id"])
    question_with_user_candidates["label"] = False

    if not (np.all(n_context_questions == np.unique(question_with_user_candidates.question_id.values, return_counts=True)[1])):
        print("WARN>> Questions have different numbers of context questions")
    assert(np.any(n_context_questions != np.unique(question_with_sim_questions.similar_question_id.values, return_counts=True)[1]))

    question_with_actuall_awnserer = target_question_features.merge(users_who_answered_target_questions, how="inner", on="question_id", validate="1:1")[["question_id", "answerer_id"]]
    assert(len(question_with_actuall_awnserer) == len(target_question_features)) # otherwise there are some target questions where we don't know the answerer
    question_with_actuall_awnserer.loc[:, "label"] = True
    n_answerers_that_did_not_awnser_any_context_question = np.count_nonzero(~question_with_actuall_awnserer.answerer_id.isin(users_who_answered_context_questions.answerer_id))
    print("For {:1.2f} % of the questions the algorithm can not find the actuall users because he did not answer any of the context questions".format(n_answerers_that_did_not_awnser_any_context_question/len(question_with_actuall_awnserer)))

    fake_answerer_candidates = utils.rows_left_not_in_right(question_with_user_candidates, question_with_actuall_awnserer, on=["question_id", "answerer_id"])
    true_candidates_selected_through_similarity = len(question_with_user_candidates) - len(fake_answerer_candidates)
    print("({}/{}) {:1.2f} % of the correct answerers were selected through similarity ".format(true_candidates_selected_through_similarity, len(question_with_actuall_awnserer), true_candidates_selected_through_similarity *100 / len(question_with_actuall_awnserer)))

    _overlap = fake_answerer_candidates.merge(question_with_actuall_awnserer, on=["question_id", "answerer_id"], how="inner")
    assert(len(_overlap)==0)

    total_sample = pd.concat([question_with_actuall_awnserer, fake_answerer_candidates]).sort_values("question_id")
    total_sample_with_features = total_sample.merge(target_question_features, on="question_id", how="left").merge(user_features, left_on="answerer_id", right_on="user_id", how="left")
    return total_sample_with_features.sort_values("question_id")


# # some valdiation stuff
# questions_with_answers_ids = set(answers_for_questions_train.index.values)
# assert(len(questions_with_answers_ids) == len(answers_for_questions_train))
# question_data_ids = set(question_data_train.question_id.values)
# assert(len(question_data_train) == len(question_data_ids))
# ids_with_answer_not_in_traindata = questions_with_answers_ids - question_data_ids
# ^ there are some questions asked before the timeperiod which got an answer with enough upvotes only in the timeperiod. those will be in this set.
cache_dir = "../cache/"

lda_pipeline = Pipeline([ ## start text pipline
    ("remove_html", features.RemoveHtmlTags()),
    ("replace_numbers", features.ReplaceNumbers()),
    ("unpack", FunctionTransformer(np.squeeze, validate=False)), # this is necessary cause the vectorizer does not expect 2d data
    ("vectorize", CountVectorizer(stop_words='english')),
    ("lda",  LDA(n_topics=10, n_iter=10000, random_state=2342)),
    ("prevalent_topic", AppendArgmax())],
    memory=cache_dir+"lda", verbose=True)

readability_pipeline = Pipeline(
        [('removeHTML', features.RemoveHtmlTags()),
         ('fog', features.ReadabilityIndexes(['GunningFogIndex'], memory=cache_dir+"readability"))]
         , verbose=True) # caching obviously doesn't help as training is not expensive

question_feature_pipeline = NamedColumnTransformer([
    ('question_id', FunctionTransformer(lambda x: x[:, None], validate=False), "question_id"),
    ('topic[10],prevalent_topic', lda_pipeline, "body"), #end text pipeline
    ('titleLength', features.LengthOfText(), 'title'),
    ('questionLength', Pipeline([('remove_html', features.RemoveHtmlTags()), ('BodyLength', features.LengthOfText())]), 'body'),
    ('nCodeBlocks', features.NumberOfCodeBlocks(), 'body'),
    ('nEquationBlocks', features.NumberOfEquationBlocks(), 'body'),
    ('nExternalLinks', features.NumberOfLinks(), 'body'),
    ('nTags', features.CountStringOccurences('<'), 'tags'),
    ('question_tags', FunctionTransformer(lambda x: x[:, None], validate=False), "tags"),
    ('readability', readability_pipeline,  'body')
]) # end Column transformer

feature_cols = ['titleLength', 'questionLength', 'nCodeBlocks', 'nEquationBlocks', 'nExternalLinks', 'nTags', 'readability', 'reputation', 'upvotes', 'downvotes', 'plattformage_days', 'numberquestions', 'numberanswers', 'numberacceptedanswers'] #+ ['label']
if use_topic_affinity:
    feature_cols += ["topic_affinity"]
    pair_feature_pipeline = NamedColumnTransformer([
        ('topic_affinity', features.TopicAffinity(), ["question_tags", "user_tags"])
    ])
else:
    pair_feature_pipeline = None


classification_pipeline = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                                    ('rf', RandomForestClassifier(n_estimators=150, min_samples_leaf=0.0003, n_jobs=1,
                                                                  class_weight="balanced", max_depth=175))])

def append_pair_features(df, pair_features_pipeline=pair_feature_pipeline):
    if pair_features_pipeline:
        extra_cols = pair_features_pipeline.fit_transform_df(df)
        df = pd.concat([df, extra_cols], axis=1)
        return df
    else:
        return df


def dataframe_to_xy(df):
    global feature_cols

    df["plattformage_days"] = df.plattformage.dt.days

    y = df["label"].values
    _feature_cols = feature_cols

    # df["noisy_label"] = y.astype(float) + 0.1 * np.random.rand(len(y))
    # _feature_cols = feature_cols + ["noisy_label"]
    # print("WARN >> added noisy label")

    actuall_cols = set(df.columns)
    assert(len(set(feature_cols) - actuall_cols)==0)
    cols_that_didnt_get_picked = actuall_cols - set(feature_cols)

    print("Used features: {}".format(_feature_cols))
    print("Columns that didn't get picked {}".format(cols_that_didnt_get_picked))

    X = df[_feature_cols].values
    X = X.astype(float)
    return X, y

def make_training_and_testing_sample_wrapper():
    t0 = time.time()
    user_data_train = get_user_data(date(year=2020, month=1, day=1))
    answers_for_questions_train = data_training.get_answer_for_question(answer_threshold)

    question_data_train_all = get_training_questions()
    question_data_train = question_data_train_all[question_data_train_all.question_id.isin(answers_for_questions_train.index)] # only questions which have an answer in the timewindow
    print("Getting training data took {}".format(time.time() - t0))

    question_features_train = question_feature_pipeline.fit_transform_df(question_data_train)

    print("Making and fitting feature pipeline took {}".format(time.time() - t0))

    if n_questions_to_train_on is not None:
        questions_to_train_on = question_features_train.sample(n=n_questions_to_train_on)
        # I try to predict the user for some questions, but all questions from the beginning are used to compute user candidates
        # TODO maybe see if it's worse if context questions come from time before
    else:
        questions_to_train_on = question_features_train

    vec, lda_obj = utils.find_lda_and_vectorizer(question_feature_pipeline)
    words_for_topic = utils.top_n_words_by_topic(vec, lda_obj, 10)
    for topic in words_for_topic:
        print(topic)

    t0 = time.time()
    training_sample = make_sample(target_question_features=questions_to_train_on, users_who_answered_target_questions=answers_for_questions_train,
                                  context_question_features=question_features_train, users_who_answered_context_questions=answers_for_questions_train, user_features=user_data_train)
    print("Making training sample took {:2f} s".format(time.time() - t0))
    training_sample = append_pair_features(training_sample)

    t0 = time.time()
    question_data_test = get_test_questions()
    assert(np.count_nonzero(question_data_test.question_id.isin(question_data_train.question_id))==0)

    users_who_answered_questions_all = data_testing.get_answer_for_question(answer_threshold)

    # but we only want questions wich have an answer and we only care about answers to our test questions
    answers_for_test_questions = users_who_answered_questions_all[users_who_answered_questions_all.index.isin(question_data_test.question_id)]
    test_questions_with_answers = question_data_test[question_data_test.question_id.isin(answers_for_test_questions.index)]
    print("Getting and filtering the test data took {}".format(time.time() - t0))

    question_features_test = question_feature_pipeline.transform_df(test_questions_with_answers)
    testing_sample = make_sample(target_question_features  = question_features_test, users_who_answered_target_questions  = answers_for_test_questions,
                                 # context_question_features = question_features_test, users_who_answered_context_questions = answers_for_test_questions,
                                 context_question_features = question_features_train, users_who_answered_context_questions = answers_for_questions_train,
                                 user_features=user_data_train) #TODO wrong context
    # TODO wrong context
    # TODO wrong context

    testing_sample = append_pair_features(testing_sample)

    return training_sample, testing_sample

if load_from_saved:
    training_sample = pd.read_pickle(cache_dir + "training_sample.pickle")
    testing_sample = pd.read_pickle(cache_dir + "testing_sample.pickle")

    print("WARN >>> The data was loaded from pickle files none of parameters take effect")
else:
    training_sample, testing_sample = make_training_and_testing_sample_wrapper()

train_X, train_y = dataframe_to_xy(training_sample)

assert(np.all(train_y == training_sample["label"]))
assert(np.all(train_X[:, 0] == training_sample["titleLength"]))
assert(np.all(np.isclose(train_X[:, 7] , training_sample["reputation"])))
# train_X, train_y = train_X[-9999:], train_y[-9999:]
# print("Making the sample took {} s, {} samples".format(time.time()-t0, len(train_y)))

# train_cv_accuracy = cross_validate(classification_pipeline, train_X, train_y, cv=10)
# print("Training CV accuracy: {}".format(train_cv_accuracy))

# Grid Search
if grid_search:
    param_grid = {"rf__n_estimators": (50, 100, 150, 200), "rf__min_samples_leaf": np.linspace(0.000000001, 0.05, 8), "rf__max_depth": [5, 25, 50, 75, 100],  "rf__class_weight": (None, "balanced")}
    grid_clf = GridSearchCV(classification_pipeline, param_grid=param_grid, scoring='f1', n_jobs=-1, verbose=1)
    grid_clf.fit(train_X, train_y)
    print("Best Parmeters: {}".format(grid_clf.best_params_)) # Best Parmeters: {'rf__class_weight': 'balanced', 'rf__min_samples_leaf': 0.1, 'rf__n_estimators': 100}
    print("Best Score: {}".format(grid_clf.best_score_))
    raise KeyboardInterrupt("")
    #

classification_pipeline.fit(train_X, train_y)
train_y_hat = classification_pipeline.predict_proba(train_X)[:, 1]
print("Training >>>", end="")
overview_score(y_hat=train_y_hat, y_true=train_y, group=training_sample.question_id.values)

training_sample["y_hat"] = train_y_hat
training_sample.to_pickle(cache_dir+"training_sample.pickle")

test_X, test_y = dataframe_to_xy(testing_sample)

assert(np.all(test_y == testing_sample["label"]))
assert(np.all(test_X[:, 0] == testing_sample["titleLength"]))
assert(np.all(np.isclose(test_X[:, 7], testing_sample["reputation"])))
test_y_hat = classification_pipeline.predict_proba(test_X)[:, 1]

testing_sample["y_hat"] = test_y_hat
testing_sample.to_pickle(cache_dir + "testing_sample.pickle")
np.save(cache_dir+"test_feat_mat.npy", test_X)

overview_score(y_true=test_y, y_hat=test_y_hat, group=testing_sample.question_id.values)

utils.print_feature_importance(classification_pipeline.named_steps['rf'].feature_importances_, feature_cols)


print("done")
