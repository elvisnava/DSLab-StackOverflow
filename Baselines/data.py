from sqlalchemy import create_engine
import pandas as pd
import re

class Data:
    """note that all operations of this class should only return data in the time range set with 'set_time_range' """

    def __init__(self, db_address = None, verbose = 0):

        if db_address is None:
            db_address = 'postgresql://localhost/crossvalidated'

        self.cnx = create_engine(db_address)

        self.macro_dict = dict(questionPostType=1, answerPostType=2)

        self.time_window_view_template = "InTimewindow_{}"

        self.table_names_that_need_timerestrictions = ["Posts", "Users", "Votes"]

        self.start, self.end = None, None
        self.drop_timewindow_views()
        self.verbose = verbose

    def set_time_range(self, start=None, end=None):
        self.start = start
        self.end = end

    def drop_timewindow_views(self):
        for raw_tablename in self.table_names_that_need_timerestrictions:
            viewname = self.time_window_view_template.format(raw_tablename)
            drop_command = "DROP VIEW IF EXISTS {};".format(viewname)
            self.execute_command(drop_command)


    def create_time_range_views(self, start, end):
        raise DeprecationWarning("Dont use this")
        self.drop_timewindow_views()

        time_range_condition = self._time_range_condition_string(start=start, end=end)

        for raw_tablename in self.table_names_that_need_timerestrictions:
            viewname = self.time_window_view_template.format(raw_tablename)
            command = "CREATE VIEW {} AS SELECT * FROM {} WHERE {}".format(viewname, raw_tablename, time_range_condition)
            self.execute_command(command)

    def get_temp_tables_string(self, raw_tablenames = None):
        if raw_tablenames is None:
            raw_tablenames = self.table_names_that_need_timerestrictions

        time_range_condition = self._time_range_condition_string(start=self.start, end=self.end)

        collector = list()
        for raw_tablename in raw_tablenames:
            viewname = self.time_window_view_template.format(raw_tablename)
            temp_table = " {} AS (SELECT * FROM {} WHERE {})".format(viewname, raw_tablename, time_range_condition)
            collector.append(temp_table)

        all_views = "WITH " + " ,".join(collector) + " "

        return all_views



    def get_tables(self):
        query = "SELECT * FROM pg_catalog.pg_tables WHERE schemaname='public';"
        return self.raw_query(query)

    def get_views(self):
        query = """select table_name from INFORMATION_SCHEMA.views WHERE table_schema = ANY (current_schemas(false))"""
        return self.raw_query(query)

    def execute_command(self, command):
        print("Execute command >>" + command)
        res = self.cnx.execute(command)
        return res

    def query(self, query_str, use_macros=False):
        """
        Run a custom query but Posts,Users,Votes tables will be restricted to contain rows created within the time intervall set in set_time_range

        :param query_str:
        :param use_macros: if True occurences of e.g. {questionPostType} will be replaced by the corresponding value in self.macro_dict
        :return:
        """
        if self.start is None and self.end is None:
            raise ValueError("you first have to set a timerange with set_time_range")

        if use_macros:
            query_str = query_str.format(**self.macro_dict)

        replaced_query = self.replace_all_tables_with_views(query_str)

        raw_tablenames_in_query = [raw_name for raw_name in self.table_names_that_need_timerestrictions
                                   if replaced_query.find(self.time_window_view_template.format(raw_name)) != -1]

        query_with_temp_tables = self.get_temp_tables_string(raw_tablenames_in_query) + replaced_query


        if self.verbose >= 1:
            print("Replaced Query>>>>"+ query_with_temp_tables)

        return self._raw_query(query_with_temp_tables)

    def raw_query(self, query_str):
        if self.verbose >= 1:
            print("WARING >> calling raw_query({}) means that data outside of the set timeperiod is also used".format(query_str))
        return self._raw_query(query_str)

    def _raw_query(self, query_str):
        return pd.read_sql_query(query_str, self.cnx)


    def replace_tablename_by_viewname(self, tablename, query):
        regex_str = r"([ \(,.=])" + tablename + r"(?=([. \n]|$))"
        replacement_str = r"\1" + self.time_window_view_template.format(tablename)

        new_query = re.sub(regex_str, replacement_str,  query, flags=re.IGNORECASE)
        return new_query

    def replace_all_tables_with_views(self, query):
        for tablename in self.table_names_that_need_timerestrictions:
            query = self.replace_tablename_by_viewname(tablename, query)
        return query

    def _time_range_condition_string(self, start=None, end=None, time_variable_name="CreationDate"):
        if start is None and end is None:
            raise ValueError("in _time_range_condition_string at least one of start or end date has to be provided")

        conds = list()
        if start:
            conds.append("{} >= date '{}'".format(time_variable_name, start))
        if end:
            conds.append("{} <= date '{}'".format(time_variable_name, end))

        return " AND ".join(conds)

    def get_questions(self):
        return self.query("SELECT Id, Title, Body, Tags FROM Posts WHERE PostTypeID = {}".format(self.questionPostType))


    def _clean_html(self, raw_html):
        cleantext = re.sub(r'<.*?>', '', raw_html)
        return cleantext


    def get_answer_for_question(self, threshold):
        """
        :return: dataframe question_id (index), answerer_id, answer_post_id, score || all pairs where user anwerer_id answered question question_id to an acceptable standard.
        there is at most one answerer per question it is the accepted answer if it exists or the answer with the hightest score above the given threshold.
        Note that the index of the returned dataframe corresponds to the question_id
        """
        accepted_answers = self.get_accepted_answer()
        best_answers = self.get_best_answer_above_threshold(threshold)

        filtered_best_answers = best_answers.drop(index=accepted_answers.index, errors="ignore") # forget all best answers for questions where an accepted answer exists

        result = pd.concat([accepted_answers, filtered_best_answers], verify_integrity=True)
        return result

    def get_best_answer_above_threshold(self, upvotes_threshold):
        """
        :return: dataframe question_id, answerer_id, answer_post_id with the answer with the most number of upvotes as long as it is above the given threshold. Doesn't care whether answer is accepted or not
        """
        # this query results in an endless loop I don't know why:
        #
        # q = """SELECT Q.Id as question_id, A.OwnerUserId as answerer_id, A.Id AS answer_post_id
        #     FROM (SELECT Id FROM Posts WHERE PostTypeId = {{questionPostType}} AND AcceptedAnswerId IS NULL) AS Q
        #         INNER JOIN
        #     (SELECT Id, OwnerUserId, Score, ParentId FROM Posts WHERE PostTypeId = {{answerPostType}} AND Score >= {threshold} AND OwnerUserId IS NOT NULL) AS A
        #     ON A.ParentId = Q.Id
        #     ORDER BY Q.Id, A.Score DESC;""".format(threshold=upvotes_threshold)

        q2 = """SELECT Distinct ON (ParentId) ParentId as question_id, OwnerUserId as answerer_id,  Id as answer_post_id, Score FROM Posts WHERE PostTypeId = {{answerPostType}} AND Score >= {threshold} AND OwnerUserId IS NOT NULL ORDER BY ParentId, Score Desc""".format(threshold=upvotes_threshold)

        out = self.query(q2, use_macros=True)

        out.set_index("question_id", inplace=True, verify_integrity=True)
        return out

    def get_accepted_answer(self):
        """

        :return: dataframe question_id, answerer_id, answer_post_id with the accepted answer for each question
        """

        q = """SELECT Q.Id as question_id, A.OwnerUserId as answerer_id, A.Id AS answer_post_id, A.Score FROM Posts Q INNER JOIN Posts A on Q.AcceptedAnswerId = A.Id WHERE Q.PostTypeId = {questionPostType} AND A.PostTypeId = {answerPostType} AND A.OwnerUserId IS NOT NULL"""

        out = self.query(q, use_macros=True)
        out.set_index("question_id", inplace=True, verify_integrity=True)
        return out