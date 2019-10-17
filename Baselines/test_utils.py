import unittest
import utils
import pandas as pd
import numpy as np

class TestUtils(unittest.TestCase):
    def test_left_not_in_right(self):
        candidates = pd.DataFrame(data=dict(question_id=[1,1,2,2,3,3], answer_id=[10,11,12,13,14,15]))
        ground_truth = pd.DataFrame(data=dict(question_id=[1, 2, 3], answer_id=[11, 10, 12]))

        actual = np.array([True, False, True, True, True, True])

        output = utils.indicator_left_not_in_right(left=candidates, right=ground_truth, on=["question_id", "answer_id"])

        self.assertTrue(np.all(output==actual))


if __name__ == '__main__':
    unittest.main()
