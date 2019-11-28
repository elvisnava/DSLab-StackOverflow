import numpy as np
import pandas as pd

from datetime import timedelta

import custom_lda

cache_dir = "../cache/"
raw_question_features_path = os.path.join(cache_dir, "raw_question_features.pickle")

class GP_Feature_Collection:

    def __init__(self, *args):
        self.features = args

    def update_event(self, event):
        for f in self.features:
            f.update(event)

    def compute_features(self, user_id, questions, event_time=None):
        sub_features = [f.compute_features(user_id, questions, event_time) for f in self.features]

        return np.concatenate(sub_features, axis = 1)

class GP_Features:

    def update_event(self, event) -> None:
        """ update interna"""
        pass

    def compute_features(self, user_id, questions, event_time=None):
        # questions is a dataframe with one row per questions, and question features as columns
        #both same length for each a feature vector
        #return matrix of same length with a feature vector for the pair in each row
        pass


# TODO make subclasses

class GP_Features_TTM(GP_Features):

    def __init__(self, n_topics=10, timedelta_wait=timedelta(days=2), load_question_features=True):
        self.n_topics = n_topics
        self.timedelta_wait = timedelta_wait
        self.users_coupe_feats = dict()
        if load_question_features:
            self.ttm_questions_features = pd.read_pickle(raw_question_features_path)
        else:
            raise NotImplementedError("Need to copy-paste the code to compute TTM feats")
        self.ttm_questions_features.set_index('question_id', inplace=True)

    def update_event(self, event):
        ttm_question = self.ttm_questions_features.loc[event.question_id]
        question_topics = list(ttm_question[['topic_{}'.format(i) for i in range(self.n_topics)]])
        sing_feat_vector = [event.answer_score, event.answer_date] + curr_question_topics

        if event.answerer_user_id not in users_coupe_feats:
            users_coupe_feats[event.answerer_user_id] = [sing_feat_vector]
        else:
            users_coupe_feats[event.answerer_user_id].append(sing_feat_vector)

    def compute_features(self, user_id, questions, event_time=None):
        """
        Compute features for user-question pairs with a single user and a list of questions.

        :param user_id: A user id to find the "non-condensed" user (COUPE) feature list for the current user
        :param questions: Df of features for the selected questions
        :param event_time: If we specify an event time, we filter the user feat list based on this
        """
        feat_pairs = []
        for q_id, question in questions.iterrows():
            ttm_question = self.ttm_questions_features.loc[question.question_id]
            curr_question_topics = list(ttm_question[['topic_{}'.format(i) for i in range(self.n_topics)]])

            if user_id in self.users_coupe_feats:
                user_feats = self.users_coupe_feats[user_id]
                #If I have event_time, I filter first to check if I'm within "timedelta_wait" days
                if event_time is not None:
                    user_feats_filter = [v for v in user_feats if (event_time - v[1]) >= self.timedelta_wait]
                else:
                    user_feats_filter = user_feats

                #I use v[0] as I assume that right now I only save votes and not the whole COUPE
                #Otherwise use np.array(v[:4]) (the 4 depends on the no of coupe feats)
                #Assume that the TTM vector in the user_feats starts from pos 2 (pos 1 is the date)
                q_u_pre_agg = np.array([v[0] * (1 - scipy.spatial.distance.jensenshannon(curr_question_topics, v[2:])) for v in user_feats_filter])
                q_u = {
                       'votes_mean': np.mean(q_u_pre_agg), 'votes_sd': np.std(q_u_pre_agg), 'votes_sum': np.sum(q_u_pre_agg), 'votes_max': np.max(q_u_pre_agg), 'votes_min': np.min(q_u_pre_agg), 'new': 0
                      }
            else:
                q_u = {
                       'votes_mean': 0, 'votes_sd': 0, 'votes_sum': 0, 'votes_max': 0, 'votes_min': 0, 'new': 1
                      }
            feat_pairs.append(pd.Series(q_u))
        pairs_dataframe = pd.concat(feat_pairs, axis=1).T
        return pairs_dataframe
