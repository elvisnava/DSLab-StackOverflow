import numpy as np
import itertools as it

def tw_matrices_to_lists(doc_word, doc_tag):
    """
    Modification of matrix_to_lists from lda.utils into its tag-word analogue
    Convert (sparse) matrices of counts into arrays of tagwords and doc indices

    Parameters
    ----------
    doc_word : array or sparse matrix (D, W)
    doc_tag : array or sparse matrix (D, T)

    Returns
    -------
    (TS, WS, DS) : Tuple of three arrays
        TS[k] contains the tag in the kth tag-word in the corpus
        WS[k] contains the word in the kth tag-word in the corpus
        DS[k] contains the document index for the kth tag-word

    """
    if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
        logger.warning("all zero row in document-word matrix found")
    if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
        logger.warning("all zero column in document-word matrix found")
    if np.count_nonzero(doc_tag.sum(axis=1)) != doc_tag.shape[0]:
        logger.warning("all zero row in document-tag matrix found")
    if np.count_nonzero(doc_tag.sum(axis=0)) != doc_tag.shape[1]:
        logger.warning("all zero column in document-tag matrix found")

    dw_sparse = True
    try:
        # if doc_word is a scipy sparse matrix
        doc_word = doc_word.copy().tolil()
    except AttributeError:
        dw_sparse = False

    dt_sparse = True
    try:
        # if doc_tag is a scipy sparse matrix
        doc_tag = doc_tag.copy().tolil()
    except AttributeError:
        dt_sparse = False

    if (dw_sparse and not np.issubdtype(doc_word.dtype, int)) or (dt_sparse and not np.issubdtype(doc_tag.dtype, int)):
        raise ValueError("expected sparse matrix with integer values, found float values")

    #Obtain doc id + word/tag id lists for nonzero entries in doc_word and doc_tag
    dw_doc_i, dw_word_i = np.nonzero(doc_word)
    if dw_sparse:
        dw_counts_i = tuple(doc_word[i, j] for i, j in zip(dw_doc_i, dw_word_i))
    else:
        dw_counts_i = doc_word[dw_doc_i, dw_word_i]
    dt_doc_i, dt_tag_i = np.nonzero(doc_tag)

    #group the words and tags by doc id, each iterator returns at each step (doc_id, iter_of_words/tags)
    dw_word_gb_doc = it.groupby(zip(dw_doc_i, dw_word_i, dw_counts_i), lambda x: x[0])
    dt_tag_gb_doc = it.groupby(zip(dt_doc_i,dt_tag_i), lambda x: x[0])
    #get an iterator that returns at each step for different doc_ids, (iter_of_words X iter_of_tags) * word repetition
    #in the form (doc_id, word_id, tag_id) * repetition
    doc_tagword_iter = it.chain.from_iterable(
                            it.repeat((tup[0][0], tup[0][1], tup[1][1]), tup[0][2]) for tup in (it.chain.from_iterable(
                                it.product(doc_words[1], doc_tags[1]) for doc_words, doc_tags in zip(dw_word_gb_doc, dt_tag_gb_doc)
                            ))
                        )

    #doc-tagword array
    DTWS = np.array(list(doc_tagword_iter))
    #return TS, WS, DS
    return DTWS[:,2], DTWS[:,1], DTWS[:,0]
