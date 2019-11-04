import unittest
import data
from datetime import date
from data_utils import make_datetime
import numpy as np

class Test_Data(unittest.TestCase):
    def setUp(self):
        self.data = data.Data(verbose=3)
        self.a = date(year=2012, month=5, day=23)
        self.b = date(year=2015, month=1, day=12)

        self.data.set_time_range(start=self.a, end=self.b)
        pass
        self.maxDiff = None

    def test_connection(self):
        out = self.data.query("Select * from Posts limit 5")
        self.assertTrue(len(out)==5)

    def test_get_user_tags(self):
        uT = self.data.get_user_tags()
        pass

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

        check("SELECT Id as UserId, CreationDate, Reputation, UpVotes, DownVotes from Users",
              "SELECT Id as UserId, CreationDate, Reputation, UpVotes, DownVotes from InTimewindow_Users")

        check("SELECT OwnerUserId as UserId, count(Posts.Id) from Posts where Posts.PostTypeId = 2 GROUP BY OwnerUserId",
              "SELECT OwnerUserId as UserId, count(InTimewindow_Posts.Id) from InTimewindow_Posts where InTimewindow_Posts.PostTypeId = 2 GROUP BY OwnerUserId")

    def test_get_answerers_strategy(self):
        question_list = [101274]
        only_accepted = data.GetAnswerersStrategy(votes_threshold=None, verbose=3)

        all = data.GetAnswerersStrategy(votes_threshold=5, verbose=3)

        accepted_answer = only_accepted.get_answers_list(question_list)
        self.assertTrue(len(accepted_answer)==1)

        all_answers = all.get_answers_list(question_list)
        self.assertEqual(len(all_answers), 3)
        pass
        print("start 2")
        all_answers_before = all.get_answers_list(question_list, before_timepoint=make_datetime("01.01.2018 12:00"))
        self.assertEqual(len(all_answers_before), 2)


        all_answers_two_questions = all.get_answers_list(question_list + [173621], make_datetime("01.01.2018 12:00"))
        self.assertEqual(len(all_answers_two_questions), 4)
        pass


    def test_temp_table_string(self):
        s = self.data.get_temp_tables_string()
        pass

    def test_time_range_condition_string(self):
        a = date(year=2012, month=5, day=23)
        b = date(year=2015, month=1, day=12)
        out_string = self.data._time_range_condition_string(start=a, end=b)
        self.assertEqual("WHERE CreationDate >= date '2012-05-23' AND CreationDate <= date '2015-01-12'", out_string)

        out_string2 = self.data._time_range_condition_string(end=b)
        self.assertEqual("WHERE CreationDate <= date '2015-01-12'", out_string2)


    def test_get_questions(self):
        qs = self.data.query("select * from Posts")
        self._check_date_in_range(qs.creationdate)
        pass

    def _check_date_in_range(self, dates):
        self.assertTrue(np.all(dates >= self.a))
        self.assertTrue(np.all(dates <= self.b))

    def test_get_accepted_answer(self):
        acept_Answers = self.data.get_accepted_answer()
        pass

    # @unittest.skip("doesnt work")
    def test_best_answer_above_threshold(self):
        ans = self.data.get_best_answer_above_threshold(upvotes_threshold=100)

        pass

    def test_get_answers(self):
        thres = 10

        acept_Answers = self.data.get_accepted_answer()
        best_answers = self.data.get_best_answer_above_threshold(upvotes_threshold=thres)

        indices_of_questions_for_which_both_esists = set(acept_Answers.index).intersection( set(best_answers.index))

        idx = np.array(list(indices_of_questions_for_which_both_esists))

        indices_whith_different_answers = idx[acept_Answers.answer_post_id[idx] != best_answers.answer_post_id[idx]]

        combined_answers = self.data.get_answer_for_question(threshold=thres)

        all_indices = np.unique(np.concatenate([acept_Answers.index , best_answers.index]))

        # try if all indicces are in the combined_ansers
        _t = combined_answers.loc[all_indices]
        self.assertEqual(len(_t), len(combined_answers))


        # check if the questions iwth different best and accepted answers have the accepted answers

        have_accepted_answer = combined_answers.loc[indices_whith_different_answers, "answer_post_id"] == acept_Answers.loc[indices_whith_different_answers, "answer_post_id"]
        self.assertTrue(np.all(have_accepted_answer))

        have_best_answer = combined_answers.loc[indices_whith_different_answers, "answer_post_id"] == best_answers.loc[indices_whith_different_answers, "answer_post_id"]
        self.assertTrue(np.count_nonzero(have_best_answer)==0)
        pass
