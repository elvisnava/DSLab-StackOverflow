from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.spatial
from features import LDAWrapper
from scipy.stats import rankdata

def mrr(out_probs, grouped_queries, ground_truth):
    """
    :param out_probs: List/Array of probabilities indicating likelihood of user to select a question
    :param grouped_queries: list indicating the group it belongs to - can contain any identifier,
    but in each group there must be exactly one ground truth
    :param ground_truth: List/array of same length, containing the actual probabilities (0 for negative, 1 for positive samples)

    returns the mrr score and a list of the same size as the parameter lists, which contains the predicted rank for each example
    """
    assert(len(out_probs)==len(ground_truth))
    assert(len(out_probs)==len(grouped_queries))
    summed_score = 0
    rank_list = np.zeros(len(out_probs))
    for q in np.unique(grouped_queries):
        # select the current block (one user-(answer+openquestions) pair)
        gt_group = ground_truth[grouped_queries==q]
        # find index of the gt answer in group g (the one with label 1)
        gt_label = np.nonzero(gt_group)[0]
        assert(len(gt_label)==1)
        gt_label = gt_label[0]
        # select predictions for current group
        out_group = out_probs[grouped_queries==q]
        # compute the ranks for the group (len(out_group) necessary for ascending)
        ranks = len(out_group)+1 - rankdata(out_group).astype(int)
        rank = ranks[gt_label] # get predicted rank of ground truth
        rank_list[grouped_queries==q] = ranks
        summed_score += 1/rank
    assert(not np.any(rank_list==0)) # now rank_list should be filled completely 
    return summed_score/len(np.unique(grouped_queries)), rank_list

def multi_mrr(out_probs, grouped_queries, ground_truth):
    """
    same as mrr but accepts multiple True ground_truth entries per group. For each group the reciprocal rank of the
     highest scoring (in out_probs) element with ground_truth = True is taken.

    :param out_probs:
    :param grouped_queries:
    :param ground_truth:
    :return:
    """
    assert(len(out_probs)==len(ground_truth))
    assert(len(out_probs)==len(grouped_queries))
    summed_score = 0
    rank_list = np.zeros(len(out_probs))
    group_ids = np.unique(grouped_queries)
    for q in group_ids:
        gt_group = ground_truth[grouped_queries==q]
        out_group = out_probs[grouped_queries==q]

        ranks = rankdata(-out_group)
        rank_list[grouped_queries==q] = ranks

        ranks_of_actually_true = ranks[gt_group]
        best_rank_of_true = np.min(ranks_of_actually_true)

        summed_score += 1/best_rank_of_true

    assert(not np.any(rank_list == 0 ))
    mrr_final = summed_score/len(group_ids)
    return mrr_final, rank_list

def success_at_n(out_probs, grouped_query, ground_truth):
    pass



def shuffle_3(X,Y,G):
    assert(len(X)==len(Y))
    assert(len(X)==len(Y))
    randinds = np.random.permutation(len(Y))
    return X[randinds], Y[randinds], G[randinds]

def mrr2(out_probs, grouped_queries, ground_truth):
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
    if hasattr(pipeline, "named_transformers_"):
        children = pipeline.named_transformers_.values()
    elif hasattr(pipeline, "named_steps"):
        children = pipeline.named_steps.values()
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
                if type(follow_stage) == LDAWrapper:
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

def rows_left_not_in_right(left, right, on):
    """

    :param left: dataframe
    :param right: dataframe
    :param on: list of strings with all the columns that should be matched
    :return: all the rows in left that are not also in right (as specified by on)
    """
    # https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe
    reduced_right = (right[on]).drop_duplicates()

    # reduced_left = left[on]

    df_all = left.merge(reduced_right, on=on, how='left', indicator=True)

    indicator = df_all._merge == 'left_only'

    values_left_not_in_right = df_all[indicator].drop(columns="_merge")

    _overlap = values_left_not_in_right.merge(right, on=on, how="inner")
    assert(len(_overlap)==0)
    return values_left_not_in_right


def set_partitions(left, right):
    left_only = left-right
    right_only = right-left
    intersection = left & right

    assert(sorted(list(left|right))== sorted(list(left_only)+list(right_only)+list(intersection)))
    return (left_only, intersection, right_only)

def df_set_partitions(left, right, on):
    """
    Given two dataframes with id columns (as specified by on keyword) as well as data columns this returns elements where the on
    columns are in the left dataframe only, right dataframe only or in both dataframes. rows that are in both dataframes are returned
    once with their data column values taken from the left and once with them taken from the right column

    :param left: dataframe
    :param right: dataframe
    :param on: list of strings with all the columns that should be matched
    :return: left_only, (both_with_left_values, both_with_right_values), right_only
    """
    raise NotImplementedError("not done")
    all_column_names = list(left.columns)
    assert(np.all(left.columns == right.columns))

    df_all = left.merge(right, on=on, how="outer", indicator=True, suffixes=['_left', '_right'])
    pass


def get_closest_n(source_features, context_features, n, source_ids=None, context_ids=None, metric='cosine', allow_less=False):
    """
    For each source point (with source_features and source_id) find the ids of the n closest context points.
    However if the context points contain the same point as the source point (as identified by matching source_id and context_id)
    This point is not returned. I.e. a point is not the closest point to itself.


    :param source_features: n_source x n_features numpy array
    :param context_features: n_context x n_features numpy array
    :param source_ids : n_source numpy array with indices, if None assume source_ids are all different then context_ids
    :param context_ids: n_context numpy array with indices, if None assume context_ids = np.arange(0, len(context_features)
    :return: a numpy array of shape len(source_ids) x n || for each source sample the ids (in context_ids) of the samples with the minimal distance to the source example
    """
    if context_ids is None:
        context_ids = np.arange(0, len(context_features))


    distances = scipy.spatial.distance.cdist(source_features, context_features, metric=metric) # output is n_source x n_context

    if source_ids is not None:
        # else we assume the source set is different from the context set of points
        is_same_id = source_ids[:, None] == context_ids[None, :]

        distances[is_same_id] = np.inf

        n_available = len(context_features) - np.any(is_same_id) # if the context contains the source we can't take them
    else:
        n_available = len(context_features)


    if n_available < n:
        if allow_less:
            n = n_available
        else:
            raise RuntimeError("You wanted {} closest points, but there are only {} context points. Consider allow_less".format(n, n_available))

    inplace_indices_of_smallest_n = np.argpartition(distances, n, axis=1)[:, :n]

    context_indices_of_smallest_n = context_ids[inplace_indices_of_smallest_n]

    return context_indices_of_smallest_n

def check_overlap(df1, df2, on):
    merged = df1.merge(df2, on=on, how="inner")

    overlap = len(merged) > 0
    return overlap

def print_feature_importance(importanceval, names):
    assert(len(importanceval) == len(names))

    sorted = np.argsort(-importanceval)

    for i in sorted:
        print("Importance {:.3f} of Feature {} ".format(importanceval[i], names[i]))