from sqlalchemy import create_engine
import pandas as pd
import re

class Data:
    """note that all operations of this class should only return data in the time range set with 'set_time_range' """

    def __init__(self, db_adress = None):

        if db_adress is None:
            db_adress = 'postgresql://localhost/crossvalidated'

        self.cnx = create_engine(db_adress)

        self.questionPostType = 1

        self.set_time_range()

    def set_time_range(self, start=None, end=None):
        self.start = start
        self.end = end

    def query(self, query_str):
        return pd.read_sql_query(query_str, self.cnx)

    def _time_range_condition_string(self, start=None, end=None, time_variable_name="CreationDate"):
        if start is None and end is None:
            raise ValueError("in _time_range_condition_string at least one of start or end date has to be provided")

        conds = list()
        if start:
            conds.append("{} >= date '{}'".format(time_variable_name, start))
        if end:
            conds.append("{} <= date '{}'".format(time_variable_name, end))

        return " AND ".join(conds)

    def get_questions(self, clean_html = True):
        questions = self._get_questions_in_timerange(start=self.start, end=self.end)
        if clean_html:
            questions['body'] = questions['body'].apply(lambda x: self._clean_html(x))
        return questions


    def _get_questions_in_timerange(self, start=None, end=None):
        """
        :param start: first allowed creation date
        :param end: last allowed creation date
        :return:
        """
        query_string = "SELECT * FROM Posts WHERE PostTypeId={postTypeId}".format(postTypeId=self.questionPostType)
        if start or end:
            query_string += " AND "
            query_string += self._time_range_condition_string(start=start, end=end)

        return self.query(query_string)

    def _clean_html(self, raw_html):
        cleantext = re.sub(r'<.*?>', '', raw_html)
        return cleantext
