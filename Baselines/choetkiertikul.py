from sklearn.pipeline import Pipeline, make_pipeline
from pipeline_utils import NamedColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
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


data_training = Data(verbose=1)
begin_training_date =date(year=2001, month=1, day=1)
end_training_date = date(year=2015, month=12, day=31)
data_training.set_time_range(start=begin_training_date, end=end_training_date)

data_testing = Data(verbose=1)
begin_testing_date = end_training_date + timedelta(days=1)
end_testing_date = date(year=2016, month=12, day=31)
data_testing.set_time_range(start=begin_training_date, end=end_testing_date)

answer_threshold = 10
n_context_questions = 50

def get_training_questions():
    question_data = data_training.query("SELECT Id as Question_Id, Title, Body, Tags FROM Posts WHERE PostTypeID = {questionPostType}", use_macros=True)
    return question_data

def get_test_questions():
    question_data = data_testing.query("SELECT Id as Question_Id, Title, Body, Tags FROM Posts WHERE PostTypeID = {{questionPostType}} AND CreationDate >= date '{}' ".format(begin_testing_date), use_macros=True)
    return question_data

tq = get_test_questions()

def overview_score(y_true, y_hat, group):
    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_hat)
    prec = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_hat)
    rec = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_hat)
    fscore = 2* prec*rec / (prec+rec)

    print("We have accuracy={:.2f} , precission={:.2f} , recall={:.2f} , fscore={:.2f}".format(acc, prec, rec, fscore))

    t0 = time.time()



def get_user_data(date_now):
    date_string = str(date_now)

    basic_user_data = data_training.query("SELECT Id as User_Id, CreationDate, Reputation, UpVotes, DownVotes, date '{}' - CreationDate AS PlattformAge from Users ORDER BY Id".format(date_string))

    n_question_for_user = data_training.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as NumberQuestions from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {questionPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True)
    n_answers_for_user = data_training.query("SELECT OwnerUserId as User_Id, count(Posts.Id) as  NumberAnswers from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {answerPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True)

    n_accepted_answers_query = """SELECT A.OwnerUserId as User_Id, count(A.Id) as NumberAcceptedAnswers from Posts Q LEFT JOIN Posts A on Q.AcceptedAnswerId = A.Id WHERE Q.AcceptedAnswerId IS NOT NULL AND A.OwnerUserId IS NOT NULL GROUP BY A.OwnerUserId ORDER BY A.OwnerUserId"""

    n_accepted_answers = data_training.query(n_accepted_answers_query)

    all_data_sources = [basic_user_data, n_question_for_user, n_answers_for_user, n_accepted_answers]
    final = reduce(lambda a,b: pd.merge(a, b, on="user_id", how="outer"), all_data_sources)
    #assert(np.all(np.isfinite(final.upvotes)))

    return final

def get_similar_questions(source_questions, context_questions, column_names_to_use_as_features, n_context_questions):
    """
    for each query question get a list of context_questions that are similar


    return pd dataframe with source_question_ids and context_question_ids. there are gonna be multiple lines with same source_question_id
    """

    topics = np.unique(source_questions.prevalent_topic)

    all_source_question_ids = list()
    all_similar_question_ids = list()

    for topic in topics:
        source_in_topic = source_questions[source_questions.prevalent_topic == topic]
        context_in_topic = context_questions[context_questions.prevalent_topic == topic]

        features_source_in_topic = source_in_topic[column_names_to_use_as_features]
        features_context_in_topic = context_in_topic[column_names_to_use_as_features]


        closest_context_question_ids = utils.get_closest_n(source_features=features_source_in_topic, context_features=features_context_in_topic, source_ids=source_in_topic.question_id.values, context_ids=context_in_topic.question_id.values, n=n_context_questions, metric='cosine')

        source_question_ids_repeated = np.repeat(source_in_topic.question_id.values[:, None], repeats=n_context_questions, axis=1)

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

    assert(len(context_question_features)==len(users_who_answered_context_questions)) # there can be answers posted

    question_ids_and_similar_questions = get_similar_questions(target_question_features, context_question_features, column_names_to_use_as_features=features_cols_for_similarity, n_context_questions=n_context_questions)
    question_with_sim_questions = question_ids_and_similar_questions.merge(users_who_answered_context_questions, left_on="similar_question_id", right_on="question_id", how="left").sort_values("question_id")

    question_with_user_candidates = question_with_sim_questions.filter(items=["question_id", "answerer_id"])
    question_with_user_candidates["label"] = False

    assert(np.all(n_context_questions == np.unique(question_with_user_candidates.question_id.values, return_counts=True)[1]))
    assert(np.any(n_context_questions != np.unique(question_with_sim_questions.similar_question_id.values, return_counts=True)[1]))

    question_with_actuall_awnserer = target_question_features.merge(users_who_answered_target_questions, how="inner", on="question_id", validate="1:1")[["question_id", "answerer_id"]]
    assert(len(question_with_actuall_awnserer) == len(target_question_features)) # otherwise there are some target questions where we don't know the answerer
    question_with_actuall_awnserer.loc[:, "label"] = True
    n_answerers_that_did_not_awnser_any_context_question = np.count_nonzero(~question_with_actuall_awnserer.answerer_id.isin(users_who_answered_context_questions.answerer_id))
    print("For {:1.2f} % of the questions the algorithm can not find the actuall users because he did not answer any of the context questions".format(n_answerers_that_did_not_awnser_any_context_question/len(question_with_actuall_awnserer)))

    fake_answerer_candidates = utils.rows_left_not_in_right(question_with_user_candidates, question_with_actuall_awnserer, on=["question_id", "answerer_id"])
    true_candidates_selected_through_similarity = len(question_with_user_candidates) - len(fake_answerer_candidates)
    print("({}) {:1.2f} % of the correct answerers were selected through similarity ".format(true_candidates_selected_through_similarity, true_candidates_selected_through_similarity *100 / len(question_with_actuall_awnserer)))

    _overlap = fake_answerer_candidates.merge(question_with_actuall_awnserer, on=["question_id", "answerer_id"], how="inner")
    assert(len(_overlap)==0)

    total_sample = pd.concat([question_with_actuall_awnserer, fake_answerer_candidates]).sort_values("question_id")
    total_sample_with_features = total_sample.merge(target_question_features, on="question_id", how="left").merge(user_features, left_on="answerer_id", right_on="user_id", how="left")
    return total_sample_with_features.sort_values("question_id")

t0 = time.time()
user_data_train = get_user_data(date(year=2020, month=1, day=1))
answers_for_questions_train = data_training.get_answer_for_question(answer_threshold)

question_data_train_all = get_training_questions()
question_data_train = question_data_train_all[question_data_train_all.question_id.isin(answers_for_questions_train.index)] # only questions which have an answer in the timewindow

# some valdiation stuff
questions_with_answers_ids = set(answers_for_questions_train.index.values)
assert(len(questions_with_answers_ids) == len(answers_for_questions_train))
question_data_ids = set(question_data_train.question_id.values)
assert(len(question_data_train) == len(question_data_ids))
ids_with_answer_not_in_traindata = questions_with_answers_ids - question_data_ids
print("Getting training data took {}".format(time.time() - t0))
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
    ('readability', readability_pipeline,  'body')
]) # end Column transformer

feature_cols = ['prevalent_topic', 'titleLength', 'questionLength', 'nCodeBlocks', 'nEquationBlocks', 'nExternalLinks', 'nTags', 'readability', 'reputation', 'upvotes', 'downvotes', 'plattformage', 'numberquestions', 'numberanswers', 'numberacceptedanswers']
def dataframe_to_xy(df):
    actuall_cols = set(df.columns)

    assert(len(set(feature_cols) - actuall_cols)==0)

    cols_that_didnt_get_picked = actuall_cols - set(feature_cols)

    print("Columns that didn't get picked {}".format(cols_that_didnt_get_picked))

    df.plattformage = df.plattformage.dt.days
    X = df[feature_cols].values
    X = X.astype(float)

    y = df["label"].values
    return X, y

t0 = time.time()
classification_pipeline = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)), ('rf', RandomForestClassifier(n_estimators=100))], memory=cache_dir+"rf")

question_features_train = question_feature_pipeline.fit_transform_df(question_data_train)
print("Making and fitting feature pipeline took {}".format(time.time() - t0))

vec, lda_obj = utils.find_lda_and_vectorizer(question_feature_pipeline)
words_for_topic = utils.top_n_words_by_topic(vec, lda_obj, 10)
for topic in words_for_topic:
    print(topic)

t0 = time.time()
training_sample = make_sample(target_question_features=question_features_train, users_who_answered_target_questions=answers_for_questions_train, context_question_features=question_features_train, users_who_answered_context_questions=answers_for_questions_train, user_features=user_data_train)
train_X, train_y = dataframe_to_xy(training_sample)
print("Making the sample took {} s".format(time.time()-t0))

# train_cv_accuracy = cross_validate(classification_pipeline, train_X, train_y, cv=10)
# print("Training CV accuracy: {}".format(train_cv_accuracy))

classification_pipeline.fit(train_X, train_y)
train_y_hat = classification_pipeline.predict(train_X)
print("Train accuracy: {}".format(np.mean(train_y_hat == train_y)))

t0 = time.time()
mrr_train = utils.mrr(out_probs=train_y_hat, grouped_queries=training_sample.question_id.values, ground_truth=train_y)
print("Training MRR is {} | took {} s".format(mrr_train, time.time()-t0))

### now we make the test data
t0 = time.time()
question_data_test = get_test_questions()
assert(np.count_nonzero(question_data_test.question_id.isin(question_data_train.question_id))==0)

users_who_answered_questions_all = data_testing.get_answer_for_question(answer_threshold)

# but we only want questions wich have an answer and we only care about answers to our test questions
answers_for_test_questions = users_who_answered_questions_all[users_who_answered_questions_all.index.isin(question_data_test.question_id)]
test_questions_with_answers = question_data_test[question_data_test.question_id.isin(answers_for_test_questions.index)]
print("Getting and filtering the test data took {}".format(time.time() - t0))

question_features_test = question_feature_pipeline.transform_df(test_questions_with_answers)
testing_sample = make_sample(target_question_features=question_features_test, users_who_answered_target_questions=answers_for_test_questions, context_question_features=question_features_train, users_who_answered_context_questions=answers_for_questions_train, user_features=user_data_train)
testing_sample.to_csv("testing_sample.csv")
test_X, test_y = dataframe_to_xy(testing_sample)

test_y_hat = classification_pipeline.predict(test_X)

print("Testing accuracy is {}".format(np.mean(test_y_hat == test_y)))

mrr = utils.mrr(out_probs=test_y_hat, grouped_queries=testing_sample.question_id, ground_truth=test_y)
print("The Test MRR is {}".format(mrr))

utils.print_feature_importance(classification_pipeline.named_steps['randomforestclassifier'].feature_importances_, feature_cols)


print("done")
