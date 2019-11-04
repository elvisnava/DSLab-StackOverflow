from data import Data
from datetime import datetime

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



training_questions_start_time = make_datetime("01.01.2015 00:00")
training_questions_end_time = make_datetime("01.06.2016 00:01")
testing_questions_start_time = make_datetime("01.06.2016 00:02")
testing_questions_end_time = make_datetime("31.12.2016 23:59")

n_feature_time_bins = 5
cache_dir = "../cache/"




db_access = Data(verbose=3)

user_features = Time_Binned_Features(db_access=db_access, gen_features_func=get_user_data, start_time=training_questions_start_time, end_time=testing_questions_end_time, n_bins=n_feature_time_bins, verbose=1)

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

###################################
# Fit the LDA
###################################

db_access.set_time_range(start=None, end=training_questions_start_time)
posts_for_fitting_lda = db_access.query("SELECT Id as Question_Id, Title, Body, Tags FROM Posts WHERE PostTypeId = {questionPostType} OR PostTypeId = {answerPostType}", use_macros=True) # we use both questions and answers to fit the lda

question_feature_pipeline.fit(posts_for_fitting_lda)

################################
# Compute Question Features Training Data
################################
db_access.set_time_range(start=None, end=testing_questions_end_time)
all_questions = db_access.query("SELECT Id as Question_Id, Title, Body, Tags, CreationDate as question_date FROM Posts WHERE PostTypeID = {questionPostType}", use_macros=True)





# compute question features global (double check)
# fit on LDA fit questions



# go through questions and get users. (all or just best answer)
# get and make the pairs, annotate where they come from