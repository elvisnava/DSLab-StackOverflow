from data import Data
import numpy as np
from datetime import date
from functools import reduce
import pandas as pd

from collections import Counter
import readability

def cumulative_term_entropy(text):
    """
    Computes cumulative term entropy of a question body as specified in
    section 3.3.2 in the paper
    """
    for bad_words in [".", "<", ">", "/", "\n", "?p", "'", "(", ")"]:
        text = text.replace(bad_words, "")
    text_list = text.lower().split(" ")
    word_list, word_count = np.unique(text_list, return_counts=True)
    num_words = len(word_list)
    cte = word_count * (np.log(num_words) - np.log(word_count))/num_words
    return sum(cte)

def get_question_features(questions):
    """
    Takes a dataframe with PostIds and question texts, and returns this dataframe with some more columns:
    Each additional column corresponds to a feature that the authors use as Question Features:
    See comments below for information on the features
    """
    # Number of words
    questions["num_words"] = questions['body'].str.count(' ') + 1
    # Referral count
    questions["num_hyperlinks"] = questions['body'].str.count('href')
    # GunningFogIndex and LIX
    readability_measures = questions["body"].apply(lambda x: readability.getmeasures(x, lang='en')['readability grades'])
    questions["GunningFogIndex"] = readability_measures.apply(lambda x: x['GunningFogIndex'])
    questions["LIX"] = readability_measures.apply(lambda x: x['LIX'])
    # Cumulative cross entropy
    questions["cumulative_term_entropy"] = questions["body"].apply(lambda x: cumulative_term_entropy(x))
    return questions

if __name__=="__main__":
    data = Data()

    data.set_time_range(start=date(year=2014, month=1, day=3), end=date(year=2014, month=5, day=1))

    ##  Todo:
    # * The authors have one feature "Question Age" which is time passed since creation date
    # * In the get_question_features function, one feature is missing: The one computed with SentiWordNet

    # Simplification of query: only take first few posts
    # questions = data.query("SELECT Id, body FROM Posts WHERE PostTypeId=1 AND Id<100")
    questions = data.query("SELECT Id, body FROM Posts WHERE PostTypeId=1")
    question_features = get_question_features(questions)

    print(question_features.head(20))
