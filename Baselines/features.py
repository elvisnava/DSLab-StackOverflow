import lda
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AppendArgmax(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        max_ids = np.argmax(X, axis=1)
        return np.concatenate([X, max_ids[:, np.newaxis]], axis=1)


# class Feature:
#     pass
#
# class LDAFeatures:
#     def __init__(self, column_name, n_topics):
#         self.column_name = column_name
#         self.n_topics = n_topics
#
#
#     def fit(self, data, n_iter=1500, random_state=1):
#         self.co
#
#         self.lda = lda.LDA(n_topics=self.n_topics, n_iter=n_iter, random_state=random_state)
#         pass
#
#     def to_features(self, data):
#         pass
#
#
#     def term_doc_matrix(self, document_list):
#         pass
#
#
#

