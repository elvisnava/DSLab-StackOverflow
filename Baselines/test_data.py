import unittest
import data
from datetime import date

class Test_Data(unittest.TestCase):
    def setUp(self):
        self.data = data.Data()

    def test_connection(self):
        out = self.data.query("Select * from Posts limit 5")
        self.assertTrue(len(out)==5)

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