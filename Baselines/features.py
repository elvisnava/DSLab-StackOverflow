import lda
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
import readability
from lda import LDA
import joblib

class AppendArgmax(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        max_ids = np.argmax(X, axis=1)
        return np.concatenate([X, max_ids[:, np.newaxis]], axis=1)


class ReadabilityIndexes(BaseEstimator, TransformerMixin):
    @staticmethod
    def all_readbility_measures(text):
        list_of_all_measures = [readability.getmeasures(t, lang="en")['readability grades'] for t in text]

        df = pd.DataFrame.from_records(list_of_all_measures)

        return df

    def get_readability_measures(self, text, readability_measures):
        """

        :param text:
        :param readability_measures: a string array with the names of the measures
        :return:
        """

        if self.memory is not None:
            func_cached = joblib.Memory(self.memory, verbose=0).cache(ReadabilityIndexes.all_readbility_measures)
        else:
            func_cached = ReadabilityIndexes.all_readbility_measures

        all_measures = func_cached(text)

        out = all_measures[readability_measures]

        return out

    def __init__(self, measures_to_compute, memory=None):
        self.measures_to_compute = measures_to_compute

        self.memory = memory


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #assert(len(X.shape)==1 or X.shape[1]==1)

        result = self.get_readability_measures(X, self.measures_to_compute)
        return result

class LDAWrapper(BaseEstimator, TransformerMixin, LDA):
    pass

class _StatelessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, require_pandas=False):
        self.require_pandas = require_pandas


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.require_pandas and type(X) != pd.Series:
            X = pd.Series(np.squeeze(X))

        out = self._transform(X)

        out = np.array(out)

        if len(out.shape) < 2:
            return out[:, None]
        else:
            return out

class RemoveHtmlTags(_StatelessTransformer):
    @staticmethod
    def clean_html(raw_html):
        cleantext = re.sub(r'<.*?>', '', raw_html)
        return cleantext

    def __init__(self):
        super().__init__(require_pandas=True)
        self.pattern = re.compile(r'<.*?>')


    def _transform(self, X, y=None):
        res =  X.str.replace(self.pattern, '')
        return res

class ReplaceNumbers(_StatelessTransformer):
    def __init__(self):
        super().__init__(require_pandas=True)
        self.pattern = re.compile(r'(\d[\.]?)+')

    def _transform(self, X):
        res = X.str.replace(self.pattern, '#N')
        return res


class LengthOfText(_StatelessTransformer):
    def __init__(self):
        super().__init__(require_pandas=True)

    def _transform(self, X, y=None):
        res = np.array(X.str.len())
        return res[:, None]

class RegexHits(_StatelessTransformer):

    def __init__(self, regex, flags=None):
        super().__init__(require_pandas=True)

        if flags:
            self.regex = re.compile(regex, *flags)
        else:
            self.regex = re.compile(regex)

    def _transform(self, X, y=None):
        return X.str.count(self.regex)

    def _get_matches(self, X):
        return X.str.findall(self.regex)



class NumberOfCodeBlocks(_StatelessTransformer):
    def __init__(self):
        super().__init__(require_pandas=True)

    def _transform(self, X, y=None):
        n_code = X.str.count("</code>")
        return n_code

class NumberOfEquationBlocks(_StatelessTransformer):
    def __init__(self):
        super().__init__(require_pandas=True)

    def _transform(self, X, y=None):
        n_delim = X.str.count("\$\$")

        assert(np.all(n_delim%1 ==0))

        n_eq = n_delim //2

        return n_eq

class NumberOfLinks(_StatelessTransformer):
    def __init__(self):
        super().__init__(require_pandas=True)

    def _transform(self, X, y=None):
        return X.str.count('<a href=')

class CountStringOccurences(_StatelessTransformer):

    def __init__(self, pattern):
        super().__init__(require_pandas=True)
        self.pattern = pattern

    def _transform(self, X, y=None):
        return X.str.count(self.pattern)




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
