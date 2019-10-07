import unittest
import features
import data
from datetime import date

class Test_Features(unittest.TestCase):
    def setUp(self):
        self.data = data.Data()
        self.data.set_time_range(start = date(year=2012, month=5, day=23), end=date(year=2013, month=1, day=2))

    def test_LDA(self):
        questions = self.data.get_questions()

        ldaFeatures = features.LDAFeatures(column_name="Body")

        ldaFeatures.fit(questions)



if __name__ == '__main__':
    unittest.main()
