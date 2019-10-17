from lda import LDA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def mrr(out_probs, grouped_queries, ground_truth):
    """
    :param out_probs: List/Array of probabilities indicating likelihood of user to select a question
    :param grouped_queries: list indicating the group it belongs to - can contain any identifier,
    but in each group there must be exactly one ground truth
    :param ground_truth: List/array of same length, containing the actual probabilities (0 for negative, 1 for positive samples)
    """
    assert(len(out_probs)==len(ground_truth))
    assert(len(out_probs)==len(grouped_queries))
    summed_score = 0
    for q in np.unique(grouped_queries):
        # select the current block (one user-(answer+openquestions) pair)
        gt_group = ground_truth[grouped_queries==q]
        gt_label = np.nonzero(gt_group)[0]
        assert(len(gt_label)==1)
        gt_label = gt_label[0]
        out_group = out_probs[grouped_queries==q]
        ranks = np.argsort(out_group).tolist()
        rank = len(ranks)-ranks.index(gt_label)
        summed_score += 1/rank
    return summed_score/len(np.unique(grouped_queries))

# Make dataset
def split_inds(nr_groups, split=0.8):
    inds = np.random.permutation(nr_groups)
    split_point = int(split*nr_groups)
    train_inds = inds[:split_point]
    test_inds = inds[split_point:]
    return train_inds, test_inds

def split_groups(df_grouped):
    df_grouped = df.groupby("decision_time")
    print(df_grouped.groups.keys())
    key_list = np.asarray(df_grouped.groups.keys)
    train_inds, test_inds = split_inds(len(key_list))

    df_train = pd.concat([df_grouped[k] for k in key_list[train_inds]])
    # df_test = pd.concat([ df_grouped.get_group(group) for i,group in enumerate(df_grouped.groups) if i in test_inds])
    return df_train, df_test


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
