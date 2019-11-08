import numpy as np
import lda

import custom_lda._custom_lda
import custom_lda.utils
import lda.utils

class TTM(lda.LDA):
    def __init__(self, n_tags, gamma=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tags = n_tags
        self.gamma = gamma

    def _fit(self, X):
        """
        Fit the model with X, a concatenation of matrices (D x T) and (D x W)
        """
        doc_tag = X[:, :self.n_tags]
        doc_word = X[:, self.n_tags:]

        random_state = lda.utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(doc_word, doc_tag)
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            self._sample_topics(rands)
        ll = self.loglikelihood()
        logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))
        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        #TTM ADDITION: TAG COMPONENTS
        self.tag_components_ = (self.nzt_ + self.gamma).astype(float)
        self.tag_components_ /= np.sum(self.tag_components_, axis=1)[:, np.newaxis]
        self.topic_tag_ = self.tag_components_

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.TS
        del self.ZS
        return self

    def _initialize(self, doc_word, doc_tag):
        D, W = doc_word.shape
        T = self.n_tags
        n_topics = self.n_topics
        n_iter = self.n_iter
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("tags_vocab_size: {}".format(T))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        #n_w|z counting tagwords with word w for topic z
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        #n_t|z counting tagwords with tag t for topic z
        self.nzt_ = nzt_ = np.zeros((n_topics, T), dtype=np.intc)
        #n_d|z counting tagwords in doc d for topic z
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        #n_z counting total tagwords for topic z
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        self.TS, self.WS, self.DS = TS, WS, DS = custom_lda.utils.tw_matrices_to_lists(doc_word, doc_tag)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        N = len(WS)
        logger.info("n_tagwords: {}".format(N))

        for i in range(N):
            t, w, d = TS[i], WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nzt_[z_new, t] += 1
            nz_[z_new] += 1
        self.loglikelihoods_ = []

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(t,w,z)
        Formula used is log p(t,w,z) = log p(t|z) + log p(w|z) + log p(z)
        """
        nzw, nzt, ndz, nz = self.nzw_, self.nzt_, self.ndz_, self.nz_
        alpha = self.alpha
        eta = self.eta
        gamma = self.gamma
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return custom_lda._custom_lda._loglikelihood(nzw, nzt, ndz, nz, nd, alpha, eta, gamma)

    def _sample_topics(self, rands):
        """Samples all topic assignments. Called once per iteration."""
        n_topics, vocab_size = self.nzw_.shape
        tags_vocab_size = self.n_tags
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        gamma = np.repeat(self.gamma, tags_vocab_size).astype(np.float64)
        custom_lda._custom_lda._sample_topics(self.WS, self.TS, self.DS, self.ZS, self.nzw_, self.nzt_, self.ndz_, self.nz_,
                                alpha, eta, gamma, rands)
