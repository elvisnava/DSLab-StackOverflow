import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import re

from sklearn.pipeline import Pipeline, make_pipeline
from pipeline_utils import NamedColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from data import Data
from data_utils import Time_Binned_Features, make_datetime
import features
import utils

import custom_lda

cache_dir = "../cache/"

training_questions_start_time = make_datetime("01.01.2018 00:00")
training_questions_end_time = make_datetime("01.06.2019 00:01")

db_access = Data(verbose=3)

db_access.set_time_range(start=None, end=training_questions_start_time)
posts_for_fitting_ttm = db_access.query("SELECT Id as Question_Id, Body, Tags FROM Posts WHERE PostTypeId = {questionPostType}", use_macros=True)

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
    ('ttm', custom_lda.TTM(n_tags=n_tags, n_topics=10, n_iter=5))],
    memory=cache_dir+"ttm", verbose=True)

#FIT TTM

print("start fitting ttm")
questions = ttm_pipeline.fit_transform(posts_for_fitting_ttm)

w_vect = ttm_pipeline.named_steps['tagword_transf'].named_transformers_['words_pipeline'].named_steps['words_vectorize']
t_vect = ttm_pipeline.named_steps['tagword_transf'].named_transformers_['tags_pipeline']
ttm_obj = ttm_pipeline.named_steps['ttm']

top_n_words_by_t = utils.top_n_words_by_topic(w_vect, ttm_obj, 10, words_or_tags='words')
print('Top 10 words by topic')
for i in range(len(top_n_words_by_t)):
    print('Topic {}: {}'.format(i, ' '.join(top_n_words_by_t[i])))
top_n_tags_by_t = utils.top_n_words_by_topic(t_vect, ttm_obj, 10, words_or_tags='tags')
print('Top 10 tags by topic')
for i in range(len(top_n_tags_by_t)):
    print('Topic {}: {}'.format(i, ' '.join(top_n_tags_by_t[i])))
