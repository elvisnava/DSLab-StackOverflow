from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer

from lda import LDA

from datetime import date
import re

from data import Data
from features import AppendArgmax
import features
import utils

data = Data()

data.set_time_range(start=date(year=2012, month=5, day=3), end=date(year=2014, month=1, day=1))

questions = data.query("SELECT * FROM Posts WHERE PostTypeID = {}".format(data.questionPostType))

lda_pipeline = ColumnTransformer([
    ('body2lda',
     Pipeline([ ## start text pipline
         ("remove_html", features.RemoveHtmlTags()),
        ("vectorize", CountVectorizer(stop_words='english', preprocessor=lambda x: re.sub(r'(\d[\.]?)+', '#N', x.lower()))),
        ("lda",  LDA(n_topics=10, n_iter=1000)),
         ("append_argmax", AppendArgmax())
    ]), "body") #end text pipeline
]) # end Column transformer


out = lda_pipeline.fit_transform(questions)

vec, lda_obj = utils.find_lda_and_vectorizer(lda_pipeline)

words_for_topic = utils.top_n_words_by_topic(vec, lda_obj, 10)
for topic in words_for_topic:
    print(topic)

print("done")
