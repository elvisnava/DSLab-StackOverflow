from lda import LDA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def pipeline2nested_list(pipeline):
    if hasattr(pipeline, "transformers_"):
        children = [c[1] for c in pipeline.transformers_]
    elif hasattr(pipeline, "steps"):
        children = pipeline.steps
    elif type(pipeline) == tuple:
        # now its a (name, transformer) pair
        return pipeline[1]
    else:
        return pipeline


    next_level =  [pipeline2nested_list(cc) for cc in children]
    return next_level

def _rek_find_lda_and_vectorizer(nested_list_of_steps):
    for s_id, stage in enumerate(nested_list_of_steps):
        if type(stage) == CountVectorizer:
            for follow_stage in nested_list_of_steps[s_id+1:]:
                if type(follow_stage) == LDA:
                    return stage, follow_stage
        elif type(stage) == list:
            return _rek_find_lda_and_vectorizer(stage)

    raise ValueError("not found")


def find_lda_and_vectorizer(pipeline):
    nested_list = pipeline2nested_list(pipeline)
    return _rek_find_lda_and_vectorizer(nested_list)

def top_n_words_by_topic(vectorizer, lda_obj, n_words):
    topic_words = lda_obj.topic_word_
    top_n_ids = np.argsort(topic_words, axis=-1)[:, :-n_words-1:-1]

    vocab = np.array(vectorizer.get_feature_names())


    result = list()

    for i in range(n_words):
        result.append( vocab[top_n_ids[i, :]].tolist())

    return result