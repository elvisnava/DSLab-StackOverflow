from sklearn.pipeline import Pipeline
from pipeline_utils import NamedColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from functools import reduce

from lda import LDA
import numpy as np
from datetime import date
import re

from data import Data
from features import AppendArgmax
import features
import utils
import pandas as pd
from datetime import date

data = Data()
data.set_time_range(start=date(year=2012, month=5, day=3), end=date(year=2013, month=1, day=1))

def get_question_data():
    question_data = data.query("SELECT Id as QuestionId, Title, Body, Tags FROM Posts WHERE PostTypeID = {questionPostType}", use_macros=True)
    return question_data

def get_user_data(date_now):
    date_string = str(date_now)

    basic_user_data = data.query("SELECT Id as UserId, CreationDate, Reputation, UpVotes, DownVotes, date '{}' - CreationDate AS PlattformAge from Users ORDER BY Id".format(date_string))

    n_question_for_user = data.query("SELECT OwnerUserId as UserId, count(Posts.Id) as NumberQuestions from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {questionPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True)
    n_answers_for_user = data.query("SELECT OwnerUserId as UserId, count(Posts.Id) as  NumberAnswers from Posts where OwnerUserId IS NOT NULL AND Posts.PostTypeId = {answerPostType} GROUP BY OwnerUserId ORDER BY OwnerUserId ", use_macros=True)

    n_accepted_answers_query = """SELECT A.OwnerUserId as UserId, count(A.Id) as NumberAcceptedAnswers from Posts Q LEFT JOIN Posts A on Q.AcceptedAnswerId = A.Id WHERE Q.AcceptedAnswerId IS NOT NULL AND A.OwnerUserId IS NOT NULL GROUP BY A.OwnerUserId ORDER BY A.OwnerUserId"""

    n_accepted_answers = data.query(n_accepted_answers_query)

    all_data_sources = [basic_user_data, n_question_for_user, n_answers_for_user, n_accepted_answers]
    final = reduce(lambda a,b: pd.merge(a, b, on="userid", how="outer"), all_data_sources)
    #assert(np.all(np.isfinite(final.upvotes)))

    return final

def get_similar_questions(source_questions, context_questions):
    """
    for each query question get a list of context_questions that are similar


    return pd dataframe with source_question_ids and context_question_ids. there are gonna be multiple lines with same source_question_id
    """
    pass

def get_users_who_answered():
    """

    :return: dataframe question_id, answerer_id all pairs where user_id answered question question_id to an acceptable standard
    """
    pass




get_user_data(date(year=2020, month=1, day=1))


question_data = get_question_data()

lda_pipeline = Pipeline([ ## start text pipline
    ("remove_html", features.RemoveHtmlTags()),
    ("replace_numbers", features.ReplaceNumbers()),
    ("unpack", FunctionTransformer(np.squeeze, validate=False)), # this is necessary cause the vectorizer does not expect 2d data
    ("vectorize", CountVectorizer(stop_words='english')),
    ("lda",  LDA(n_topics=10, n_iter=10)),
    ("append_argmax", AppendArgmax())])


question_feature_pipeline = NamedColumnTransformer([
    ('QuestionId', FunctionTransformer(lambda x: x[:, None], validate=False), "questionid"),
    ('topic[10],best_topic_id', lda_pipeline, "body"), #end text pipeline
    ('titleLength', features.LengthOfText(), 'title'),
    ('questionLength', Pipeline([('remove_html', features.RemoveHtmlTags()), ('BodyLength', features.LengthOfText())]), 'body'),
    ('nCodeBlocks', features.NumberOfCodeBlocks(), 'body'),
    ('nEquationBlocks', features.NumberOfEquationBlocks(), 'body'),
    ('nExternalLinks', features.NumberOfLinks(), 'body'),
    ('nTags', features.CountStringOccurences('<'), 'tags'),
    ('readability', Pipeline([('removeHTML', features.RemoveHtmlTags()), ('fog', features.ReadabilityIndexes(['GunningFogIndex']))]), 'body')
]) # end Column transformer

out = question_feature_pipeline.fit_transform_df(question_data)

vec, lda_obj = utils.find_lda_and_vectorizer(question_feature_pipeline)

words_for_topic = utils.top_n_words_by_topic(vec, lda_obj, 10)
for topic in words_for_topic:
    print(topic)

print("done")
