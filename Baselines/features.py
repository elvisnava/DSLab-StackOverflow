import lda
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
import readability
from lda import LDA
import joblib
from sklearn.preprocessing import FunctionTransformer


class AppendArgmax(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        max_ids = np.argmax(X, axis=1)
        return np.concatenate([X, max_ids[:, np.newaxis]], axis=1)

class TopicAffinity(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.question_tags_col = "question_tags"
        self.user_tags_col = "user_tags"

    def fit(self, X, y=None):
        return self

    def to_taglist(self, string):
        if (not isinstance(string, str) and np.isnan(string)) or len(string)==0:
            return []

        assert(string[0]=="<")
        assert(string[-1]==">")

        return string[1:-1].split("><")


    def transform(self, X, y=None):
        n_samples, n_cols = X.shape
        assert(n_cols == 2)

        out = np.zeros((n_samples, 1))

        for i in range(n_samples):
            question_tags_str = X[self.question_tags_col][i]
            user_tags_str = X[self.user_tags_col][i]

            user_tags = self.to_taglist(user_tags_str)

            unique_user_tags, counts = np.unique(user_tags, return_counts=True)
            user_pdf = counts/np.sum(counts)

            question_tags = self.to_taglist(question_tags_str)

            activated_tags = np.isin(unique_user_tags, question_tags)


            probs_of_activated = user_pdf[activated_tags]
            if len(probs_of_activated) > 0:
                prod = np.prod(probs_of_activated)
            else:
                prod = 0

            out[i, 0] = prod

        return out

def make_identity_transformer():
    return FunctionTransformer(lambda x: x[:, None], validate=False)



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

class IdentityTransformer(_StatelessTransformer):
    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X

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

        #assert(np.all(n_delim%2 ==0))
        # the number of equation blocks there some people doing stuff like n$_1$$^2$ like wtf

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

class Reputation(_StatelessTransformer):

    def __init__(self, max_date, data):
        super().__init__(require_pandas=True)
        self.max_date = max_date
        self.data = data

    def compute_factor(self, row):
        if row["posttypeid"]==1 and row["votetypeid"]==2: # upvoted question
            factor = 5
        elif row["posttypeid"]==1 and row["votetypeid"]==3: # downvoted question
            factor = -5
        elif row["posttypeid"]==2 and row["votetypeid"]==2: # upvoted answer
            factor = 10
        elif row["posttypeid"]==2 and row["votetypeid"]==3: # downvoted anser
            factor = -10
        elif row["posttypeid"]==2 and row["votetypeid"]==1: # accepted answer
            factor = 15
        elif row["votetypeid"]==8:
            factor = row["bountyamount"]
        else:
            factor = 0
        res = factor*row["postid"]
        return res

    def transform(self, X, y=None):

        reputation = self.data.query("SELECT * FROM Votes LEFT JOIN (SELECT Id, PostTypeId, OwnerUserId FROM Posts) b ON Votes.Id=b.Id WHERE CreationDate<'{}'".format(str(self.max_date)))
        grouped = reputation.groupby(["owneruserid", "posttypeid", "votetypeid"]).agg({"postid": "count", "bountyamount":"sum"})
        grouped = grouped.reset_index()
        grouped["score"] = grouped.apply(self.compute_factor, axis=1)
        out = grouped.groupby(["owneruserid"]).agg({"score":"sum"}).reset_index()
        merged = pd.merge(X, out, on="owneruserid", how="left")
        return merged


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
