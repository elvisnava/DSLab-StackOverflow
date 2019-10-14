import unittest
import pipeline_utils


class Test_PipelineUtils(unittest.TestCase):


    def test_name_parsing(self):
        namedColTrans = pipeline_utils.NamedColumnTransformer([])

        out = namedColTrans.label2column_names("lda[10],best_topic[2]")

        pass


if __name__ == '__main__':
    unittest.main()
