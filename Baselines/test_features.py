import unittest
import features
import data
from datetime import date
import pandas as pd
import numpy as np

class Test_Features(unittest.TestCase):
    def setUp(self):
        self.data = data.Data()
        self.data.set_time_range(start = date(year=2012, month=5, day=23), end=date(year=2013, month=1, day=2))

    def test_LDA(self):
        questions = self.data.get_questions()

        ldaFeatures = features.LDAFeatures(column_name="Body")

        ldaFeatures.fit(questions)

    def test_clean_html(self):
        html_clean = features.RemoveHtmlTags()

        questions = self.data.query("Select Body from Posts WHERE PostTypeId=2")

        cleaned = html_clean.fit_transform(questions.body)
        pass

    def test_readability(self):
        test_text = "Fires in the workings hampered rescue efforts, and it took several days before they were under control. It took several weeks for most of the bodies to be recovered. The subsequent enquiry pointed to errors made by the company and its management leading to charges of negligence against Edward Shaw, the colliery manager, and the owners. Shaw was fined £24 while the company was fined £10; newspapers calculated the cost of each miner lost was just 1 shilling ​1 1⁄4d (about ​5 1⁄2 pence at the time)."
        test_sentences = [s for s in test_text.split(".") if len(s) > 0]

        rF = features.ReadabilityIndexes(["GunningFogIndex"])

        readability_scors = rF.fit_transform(test_sentences)
        pass

    def print_example_questions_with_value(self, feature, n_q, column ="body"):
        questions = self.data.get_questions()[column][:n_q]
        values = feature.fit_transform(questions)

        for i in range(len(questions)):
            print("||||{} in question |||||>> {}".format(values[i], questions[i]))
            print("---------------------------\n")

    def test_code_hits(self):
        code_hits_feature = features.NumberOfCodeBlocks()

        some_questions = self.data.get_questions().body[:10]

        out = code_hits_feature.fit_transform(some_questions)

        for i in range(len(some_questions)):
            print("||||{} in question |||||>> {}".format(out[i], some_questions[i]))
            print("---------------------------\n")
            pass

    def test_number_equations(self):
        eq_hits_features = features.NumberOfEquationBlocks()

        test_questions = pd.Series(["$$ \bla eq $$ bla text $$ 29394$$ test", " $\sigma$ inline"])
        test_out = eq_hits_features.fit_transform(test_questions)
        self.assertTrue(np.all(np.equal(test_out, np.array([2, 0]))))

        self.print_example_questions_with_value(eq_hits_features, 30)

    def test_number_links(self):
        link_f = features.NumberOfLinks()

        self.print_example_questions_with_value(link_f, 20)

    def test_tag_count(self):
        tag_count_f = features.CountStringOccurences("<")
        self.print_example_questions_with_value(tag_count_f, 30, column="tags")

    def test_reputation(self):
        questions = self.data.query("Select Id as owneruserid, Reputation from Users")
        repu = features.Reputation('2019-03-14 15:15:10.813000', self.data)
        print(questions.head(20))
        cleaned = repu.fit_transform(questions)
        print(cleaned.head(20))


if __name__ == '__main__':
    unittest.main()
