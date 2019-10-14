from sklearn.compose import ColumnTransformer
import pandas as pd

class NamedColumnTransformer(ColumnTransformer):
    def __init__(self, transformers, remainder="drop", sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False):
        """

        :param transformers: a list with each element (output_columns_identifier, Pipeline, name_of_column_to_acct_on)
            output_columns_identifier should be a comma seperated list of names for each column that is outputed by the Pipeline, but a name can be folloed by name[3] then 3 columns are named like that.
            e.g. 'lda[4],best_topic,entropy' => would generate 6 column names: lda0, lda1, lda2, lda3, best_topic, entropy

        :param remainder:
        :param sparse_threshold:
        :param n_jobs:
        :param transformer_weights:
        :param verbose:
        """
        super().__init__(transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs, transformer_weights=transformer_weights, verbose=verbose)

        self.raw_transformer_names = [t[0] for t in transformers]
        self.final_column_names = self.get_final_column_names(self.raw_transformer_names)


    def label2column_names(self, description_string):
        parts = description_string.split(",")
        result = list()

        for part in parts:
            if part[-1] == "]":
                start = part.find("[")
                assert(start != -1)
                n_features_in_this = int(part[start+1:-1])
                name = part[:start]

                for i in range(n_features_in_this):
                    result.append("{}_{}".format(name, i))
            else:
                result.append(part)
        return result

    def transform_df(self, X):
        out = self.transform(X)
        return self.np_array2dataframe(out)

    def fit_transform_df(self, X, y=None):
        assert(y is None)
        out = self.fit_transform(X)
        return self.np_array2dataframe(out)



    def get_final_column_names(self, raw_transformer_names):
        result = list()
        for t in raw_transformer_names:
            parsed = self.label2column_names(t)
            result.extend(parsed)
        return result

    def np_array2dataframe(self, x):
        return pd.DataFrame(data=x, columns=self.final_column_names)