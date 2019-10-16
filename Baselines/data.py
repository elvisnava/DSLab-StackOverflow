from sqlalchemy import create_engine
import pandas as pd
import re

class Data:
    """note that all operations of this class should only return data in the time range set with 'set_time_range' """

    def __init__(self, db_adress = None, verbose=0):

        if db_adress is None:
            db_adress = 'postgresql://localhost/crossvalidated'

        self.cnx = create_engine(db_adress)

        self.macro_dict = dict(questionPostType=1, answerPostType=2)

        self.time_window_view_template = "InTimewindow_{}"

        self.table_names_that_need_timerestrictions = ["Posts", "Users", "Votes"]

        self.start, self.end = None, None
        self.drop_timewindow_views()
        self.verbose = verbose

    def set_time_range(self, start=None, end=None):
        self.start = start
        self.end = end
        self.create_time_range_views(start, end)

    def drop_timewindow_views(self):
        for raw_tablename in self.table_names_that_need_timerestrictions:
            viewname = self.time_window_view_template.format(raw_tablename)
            drop_command = "DROP VIEW IF EXISTS {};".format(viewname)
            self.execute_command(drop_command)


    def create_time_range_views(self, start, end):
        self.drop_timewindow_views()

        time_range_condition = self._time_range_condition_string(start=start, end=end)

        for raw_tablename in self.table_names_that_need_timerestrictions:
            viewname = self.time_window_view_template.format(raw_tablename)
            command = "CREATE VIEW {} AS SELECT * FROM {} WHERE {}".format(viewname, raw_tablename, time_range_condition)
            self.execute_command(command)

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
        if self.verbose >= 1:
            print("Replaced Query>>>>"+replaced_query)

        return self._raw_query(replaced_query)

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
