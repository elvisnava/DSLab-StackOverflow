import unittest
import data
from datetime import date
import numpy as np

class Test_Data(unittest.TestCase):
    def setUp(self):
        self.data = data.Data()
        self.a = date(year=2012, month=5, day=23)
        self.b = date(year=2015, month=1, day=12)

        self.data.set_time_range(start=self.a, end=self.b)
        pass
        self.maxDiff = None

    def test_connection(self):
        out = self.data.query("Select * from Posts limit 5")
        self.assertTrue(len(out)==5)

    def test_query_replacement(self):
        def check(orig, target):
            replaced = self.data.replace_all_tables_with_views(orig)
            self.assertEqual(replaced, target)


        check("SELECT * FROM Posts", "SELECT * FROM InTimewindow_Posts")
        check("SELECT * FROM Posts posts", "SELECT * FROM InTimewindow_Posts InTimewindow_Posts")

        check("SELECT OwnerUserId, count(OwnerUserId) as number_answers FROM Posts WHERE PostTypeId=2 GROUP BY OwnerUserId",
              "SELECT OwnerUserId, count(OwnerUserId) as number_answers FROM InTimewindow_Posts WHERE PostTypeId=2 GROUP BY OwnerUserId")

        check("SELECT OwnerUserId, count(OwnerUserId) as num_solved_questions FROM Posts INNER JOIN Users ON Posts.OwnerUserId=Users.Id WHERE PostTypeId=1 AND AcceptedAnswerId IS NOT NULL GROUP BY OwnerUserId",
              "SELECT OwnerUserId, count(OwnerUserId) as num_solved_questions FROM InTimewindow_Posts INNER JOIN InTimewindow_Users ON InTimewindow_Posts.OwnerUserId=InTimewindow_Users.Id WHERE PostTypeId=1 AND AcceptedAnswerId IS NOT NULL GROUP BY OwnerUserId")



    def test_time_range_condition_string(self):
        a = date(year=2012, month=5, day=23)
        b = date(year=2015, month=1, day=12)
        out_string = self.data._time_range_condition_string(start=a, end=b)
        self.assertEqual("CreationDate >= date '2012-05-23' AND CreationDate <= date '2015-01-12'", out_string)

        out_string2 = self.data._time_range_condition_string(end=b)
        self.assertEqual("CreationDate <= date '2015-01-12'", out_string2)

    def test_get_question(self):
        a = date(year=2012, month=5, day=23)
        b = date(year=2015, month=1, day=12)


        questions = self.data._get_questions_in_timerange(start=a, end=b)
        pass

    def test_get_questions(self):
        qs = self.data.query("select * from Posts")
        self._check_date_in_range(qs.creationdate)
        pass

    def _check_date_in_range(self, dates):
        self.assertTrue(np.all(dates >= self.a))
        self.assertTrue(np.all(dates <= self.b))
