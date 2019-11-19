from data import Data
from datetime import datetime
import pickle

import pandas as pd
from functools import reduce

from data_utils import Time_Binned_Features, make_datetime
from choetkiertikul_helpers import *

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
import os


# from lda import LDA
from features import LDAWrapper as LDA

import numpy as np
from datetime import date
from datetime import timedelta
import re


from data import Data, GetAnswerersStrategy
from features import AppendArgmax
import features
import utils
import pandas as pd
import time




training_questions_start_time = make_datetime("01.01.2015 00:00")
training_questions_end_time = make_datetime("01.06.2016 00:01")
testing_questions_start_time = make_datetime("01.06.2016 00:02")
testing_questions_end_time = make_datetime("31.12.2016 23:59")

n_feature_time_bins = 50

n_candidate_questions = 30
votes_threshold_for_answerers = 3

cache_dir = "../cache/"

load_question_features = True
load_user_features = True
load_final_pairs = True

raw_question_features_path = os.path.join(cache_dir, "raw_question_features.pickle")
binned_user_features_path = os.path.join(cache_dir, "binned_user_features.pickle")
all_pairs_path = os.path.join(cache_dir, "final_candidates_pairs.pickle")


feature_cols_for_training = ['titleLength', 'questionLength', 'nCodeBlocks', 'nEquationBlocks', 'nExternalLinks', 'nTags', 'readability', 'reputation', 'upvotes', 'downvotes', 'plattformage', 'numberquestions', 'numberanswers', 'numberacceptedanswers']

# define times
# fit LDA

################################
# Pipelines
################################

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
         , verbose=True) # the ReadabilityFeature itself caches it's transform method now

question_feature_pipeline = NamedColumnTransformer([
    ('question_id', None,  "question_id"),
    ('topic[10],prevalent_topic', lda_pipeline, "body"), #end text pipeline
    ('titleLength', features.LengthOfText(), 'title'),
    ('questionLength', Pipeline([('remove_html', features.RemoveHtmlTags()), ('BodyLength', features.LengthOfText())]), 'body'),
    ('nCodeBlocks', features.NumberOfCodeBlocks(), 'body'),
    ('nEquationBlocks', features.NumberOfEquationBlocks(), 'body'),
    ('nExternalLinks', features.NumberOfLinks(), 'body'),
    ('nTags', features.CountStringOccurences('<'), 'question_tags'),
    ('question_tags', None,  "question_tags"),
    ('readability', readability_pipeline,  'body'),
    ('creationdate', None, 'creationdate')
]) # end Column transformer

############
# Now do data stuff
#############

db_access = Data(verbose=3)

if load_user_features:
    with open(binned_user_features_path, "rb") as f:
        binned_user_features= pickle.load(f)
else:
    binned_user_features = Time_Binned_Features(gen_features_func=get_user_data, start_time=training_questions_start_time, end_time=testing_questions_end_time, n_bins=n_feature_time_bins, verbose=1)

    with open(binned_user_features_path, "wb") as f:
        pickle.dump(binned_user_features, f)

print('finished user features')
###################################
# Fit the LDA
###################################

if load_question_features:
    all_questions_features = pd.read_pickle(raw_question_features_path)
else:
    db_access.set_time_range(start=None, end=training_questions_start_time)
    posts_for_fitting_lda = db_access.query("SELECT Id as Question_Id, Title, Body, Tags as question_tags, CreationDate FROM Posts WHERE PostTypeId = {questionPostType}", use_macros=True) # we use both questions and answers to fit the lda

    question_feature_pipeline.fit(posts_for_fitting_lda)



    ################################
    # Compute Question Features Training Data
    ################################
    db_access.set_time_range(start=None, end=testing_questions_end_time)
    all_questions = db_access.query("SELECT Id as Question_Id, Title, Body, Tags as question_tags, CreationDate FROM Posts WHERE PostTypeID = {questionPostType}", use_macros=True)
    # I could prefilter here to only take answered questions -> save computation

    # subs = question_feature_pipeline.transform_df(all_questions[:100])

    all_questions_features = question_feature_pipeline.transform_df(all_questions)

    all_questions_features.to_pickle(raw_question_features_path)
print("finished question features")


if load_final_pairs:
    all_pairs = pd.read_pickle(all_pairs_path)
else:
    features_cols_for_similarity = ["titleLength", "questionLength", "nCodeBlocks", "nEquationBlocks", "nExternalLinks", "nTags", "readability"]
    all_pairs = make_pairs(all_questions_features, question_features_to_use_for_similarity=features_cols_for_similarity,
                           question_start_time=training_questions_start_time,
                           group_column_name='prevalent_topic', answerers_strategy=GetAnswerersStrategy(votes_threshold=votes_threshold_for_answerers, verbose=0),
                           n_candidate_questions=n_candidate_questions, user_features=binned_user_features)

    pd.to_pickle(all_pairs, all_pairs_path)

print('finished making pairs')

# go through questions and get users. (all or just best answer)
# get and make the pairs, annotate where they come from


training_pairs = all_pairs[all_pairs.creationdate_question <= training_questions_end_time]
testing_pairs = all_pairs[all_pairs.creationdate_question >= testing_questions_start_time]

##########
# classification
#########
classification_pipeline = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                                    ('rf', RandomForestClassifier(n_estimators=150, min_samples_leaf=0.0003, n_jobs=1,
                                                                  class_weight="balanced", max_depth=75))])

# Training
train_X, train_y = dataframe_to_xy(training_pairs, feature_cols=feature_cols_for_training)
test_X, test_y = dataframe_to_xy(testing_pairs, feature_cols=feature_cols_for_training)

classification_pipeline.fit(train_X, train_y)
train_y_hat = classification_pipeline.predict(train_X)
print("Done with Training")
pass

test_y_hat = classification_pipeline.predict_proba(test_X)[:, 1]
print(test_y)
print(test_y_hat)

overview_score(y_true=test_y, y_hat=test_y_hat, group=testing_pairs.question_id)


