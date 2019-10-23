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

    def test_closest_n(self):

        center_0 = 3*np.array([1,1,1])
        center_1 = 3*np.array([-1,-1,-1])

        centers = np.zeros((2,3))
        centers[0, :] = center_0
        centers[1, :] = center_1

        center_ids = np.array([0, 1])

        # context points that are different
        context_points = np.zeros((12, 3))
        context_points[0:5, :] = center_0
        context_points[5:10, :] = center_1

        context_points += np.random.rand(*context_points.shape)

        context_point_ids = np.arange(12) + 100 # all the context points have different ids
        # except for the last 2 which are again our center

        context_points[10] = center_0
        context_points[11] = center_1
        context_point_ids[10] = 0
        context_point_ids[11] = 1

        output = utils.get_closest_n(source_features=centers, context_features=context_points, source_ids=center_ids, context_ids=context_point_ids, n=5, metric="euclidean")

        # the last two context points should not be picked cause their id is the same as the corresponding source id ( point0 would be the closest point to point0)
        self.assertTrue(np.all(np.unique(output[0, :])==context_point_ids[0:5]))
        self.assertTrue(np.all(np.unique(output[1, :])==context_point_ids[5:10]))

        # now we change the ids of the last two points and they should now be picked as they are now

        context_point_ids[10]=1
        context_point_ids[11]=0
        output2 = utils.get_closest_n(source_features=centers, context_features=context_points, source_ids=center_ids, context_ids=context_point_ids, n=5, metric="euclidean")

        # the 10th context point with id 1 is the closest to center 0 (it's actually the same point)
        self.assertTrue(1 in output2[0])
        self.assertTrue(0 in output2[1])


    def test_left_not_in_right(self):

        # make left array

        a = np.random.randint(0, 5, 100).astype(float)
        b = np.random.randint(0, 5, 100).astype(float)
        c = + np.arange(100).astype(float) + 1

        left = pd.DataFrame(data=dict(a=a, b=b, c=c))

        # make right array
        a2 = np.random.randint(0,10, 30).astype(float)
        b2 = np.random.randint(0,10, 30).astype(float)
        c2 = -np.arange(30).astype(float) -1


        right = pd.DataFrame(data=dict(a=a2, b=b2, c=c2))

        # make sure there is some overlap

        overlap = left.merge(right, on=["a", "b"], how="inner")

        assert(len(overlap)>0) # otherwise the test was unlucky
        # assert(len(overlap)< len(right))

        left_only = utils.rows_left_not_in_right(left, right, on=["a", "b"])

        overlap_after = left_only.merge(right, on=['a', 'b'], how="inner")
        # there should not be anny overlap any more
        self.assertTrue(len(overlap_after) ==0)

        both = pd.concat([left_only, right])

        # all elements in right should be contained
        both_right_only = both.merge(right, on=['a','b','c'], how="inner")
        self.assertTrue(np.all(both_right_only == right))

        # not all left elements should be contained. I use the c column of left as indicator
        self.assertFalse(np.all(np.isin(c, both.c.values)))
        pass




if __name__ == '__main__':
    unittest.main()
