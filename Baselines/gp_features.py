import numpy as np

class GP_Feature_Collection:

    def __init__(self, *args):
        self.features = args

    def update_event(self, event):
        for f in self.features:
            f.update(event)

    def compute_features(self, list_of_user_id, questions, event_time=None):
        sub_features = [f.compute_features(list_of_user_id, questions, event_time) for f in self.features]

        return np.concatenate(sub_features, axis = 1)

class GP_Features:

    def update_event(self, event) -> None:
        """ update interna"""
        pass

    def compute_features(self, list_of_user_id, questions, event_time=None):
        # questions is a dataframe with one row per questions, and question featrues as columnbs
        #both same length for each a feature vector
        #return matrix of same length with a feature vector for the pair in each row
        pass


# TODO make subclasses

