from datetime import datetime, timedelta
import numpy as np
import data


def make_datetime(s):
    return datetime.strptime(s, '%d.%m.%Y %H:%M')

class Time_Binned_Features:
    # a class for storing and acessing time binned features

    def log(self, string, priority=1):
        if priority <= self.verbose:
            print("TimeBinnedFeatures: {}".format(string))

    def __init__(self, gen_features_func, start_time, end_time, n_bins, time_intervals_open_left=True, verbose=0):
        """

        :param gen_features_func: a function db_access -> pandas dataframe (computes the desired features using the data object. i.e. this function is gonna be called once for every bin
        :param start_time: datetime object
        :param end_time: datetime object
        :param n_bins: int
        :param time_intervalls_open_left: if True then in each bin the time starts at the beginning, if datetime object then the timeintervalls all start at this timeintervall
        """

        # each bin is beginning of time util edge

        db_access = data.Data(verbose=verbose)

        assert(time_intervals_open_left)


        intervall_length = (end_time - start_time) / n_bins

        self.start_time = start_time
        self.intervall_length = intervall_length
        self.end_time = end_time
        self.n_bins = n_bins
        self.bin_edges = self._compute_bin_edges()

        self.time_intervalls_open_left = time_intervals_open_left

        self.verbose = verbose

        self.log("The intervall length is {}".format(intervall_length))


        self._binned_features = dict()

        for bin_id in range(n_bins):
            s = self.bin_edges[bin_id]
            e = self.bin_edges[bin_id]

            # if self.time_intervalls_open_left == False:
            #     db_access.set_time_range(start=s, end=e)
            # elif self.time_intervalls_open_left == True:
            #     db_access.set_time_range(start=None, end=e)
            # elif isinstance( self.time_intervalls_open_left, datetime):
            #     db_access.set_time_range(start=self.time_intervalls_open_left, end=e)
            # else:
            #     ValueError("time_intervalls_open_left has to be True, False or datetime object")

            # imporantly we only use data before the start of the current time bin
            db_access.set_time_range(start=None, end=s)
            features_this_time = gen_features_func(db_access)


            self._binned_features[bin_id] = features_this_time


    def __getitem__(self, key):
        """implements timeBinnedFeatures[datetime(year=2012, month=3, day=12, hour=2, minute=29)] -> returns features from the last bin that ended before that"""
        timepoint = self._ensure_datetime(key)


        bin_id = self._last_id_before(timepoint)

        self.log("Using timebin {} - {}".format(self.bin_edges[bin_id], self.bin_edges[bin_id +1]), priority=2)

        return self._binned_features[bin_id]

    def _ensure_datetime(self, key):
        if isinstance(key, datetime):
            timepoint = key
        elif isinstance(key, str):
            timepoint = make_datetime(key)
        else:
            raise ValueError(
                "The index should be a datetime object or a string that can be formated to a datetime with .data_utils.make_datetime")
        return timepoint

    def age_of_data(self, key):
        timepoint = self._ensure_datetime(key)
        bin_id = self._last_id_before(timepoint)
        _age = timepoint - self.bin_edges[bin_id]
        return _age





    def _compute_bin_edges(self):
        bin_edges  = self.start_time + np.arange(0, self.n_bins+1)*self.intervall_length
        assert(len(bin_edges)==self.n_bins+1)
        assert(bin_edges[-1] == self.end_time)
        return bin_edges


    def _last_id_before(self, timepoint):
        assert(timepoint < self.end_time)
        assert(timepoint >= self.start_time)

        after_start = timepoint - self.start_time

        idd = after_start // self.intervall_length

        return idd



def user_answers_young_question_event_iterator_with_candidates(data_cache: data.DataHandleCached, hour_threshold, start_time=None):
    """
    Iterate over events where a user answers a young question (as by hour threshold)
    each event is a tuple of 4:
        (event_date(time when user answered),
        user_id,
        actually_answered_id (the question that user_id picked),
        young_open_questions_at_that_time ( a pandas dataframe with all questions that are open at that time and young enough)

    :param data_cache:
    :param hour_threshold:
    :param start_time: ignore all events before this
    :return:
    """

    for (event_date, user_id, actually_answered_id) in user_answers_young_question_event_iterator(data.Data(), hour_threshold=hour_threshold, start_time=start_time):
# TODO get young candidates
        open_questions = data_cache.open_questions_at_time(event_date, actually_answered_id)

        young_open_questions_at_the_time = open_questions[open_questions.question_date >= (event_date - timedelta(hours=hour_threshold))]

        _t = actually_answered_id in open_questions.question_id
        yield (event_date, user_id, actually_answered_id, young_open_questions_at_the_time)

# TODO modify this to go through all events
def user_answers_young_question_event_iterator(data_handle: data.Data, hour_threshold, start_time = None):
    """
    Iterator over events where a user answers a question that is younger then hour_threshold, we assume that this indicates that she answered a question that was suggested (and not one she stumbled upon)
    :param data_handle:
    :param hour_threshold:
    :return: yields (event_date, user_id, actually_answered) the user user_id was looking for someting to do at event_date and then answered question with id actually_answered
    """
    data_handle.set_time_range(start=None, end=None)

    all_answers = data_handle.query("""
    SELECT Q.Id as question_id, A.Id as answer_id, A.OwnerUserId as answerer_user_id, Q.body as question_body, Q.tags as question_tags, (A.CreationDate - Q.CreationDate) as question_age_at_answer, Q.CreationDate as question_date, A.CreationDate as answer_date
    FROM Posts AS A JOIN Posts as Q on A.ParentId = Q.Id
    WHERE A.PostTypeId = {{answerPostType}} AND Q.PostTypeId = {{questionPostType}} AND (A.CreationDate - Q.CreationDate) < interval '{} hours'
    ORDER BY A.CreationDate
    """.format(hour_threshold), use_macros=True)

    if start_time is not None:
        all_answers = all_answers[all_answers.answer_date >= start_time]


    last_date = make_datetime('01.01.1900 00:00')
    for i in range(len(all_answers)):
        current = all_answers.iloc[i]
        event_date = current.answer_date

        assert(last_date < event_date)
        last_date = event_date

        user_id = current.answerer_user_id
        actually_answered = current.question_id

        sample = (event_date, user_id, actually_answered)
        yield sample


def all_answer_events_iterator(data_handle: data.Data=None, start_time = None, end_time=None):
    """
    Iterator over events where a user answers a question
    :param data_handle:
    :return: a pandas series with fields question_id, answer_id, answerer_user_id, question_body, question_tags, question_age_at_answer, question_date, answer_date
    """

    if data_handle is None:
        data_handle = data.Data()

    data_handle.set_time_range(start=None, end=None)

    start_time_cond = ""
    if start_time:
        start_time_cond = "AND A.CreationDate >= date '{}' ".format(start_time)

    end_time_cond = ""
    if end_time:
        end_time_cond = "AND A.CreationDate < date '{}'".format(end_time)

    all_answers = data_handle.query("""
    SELECT Q.Id as question_id, A.Id as answer_id, Q.OwnerUserId as asker_user_id, A.OwnerUserId as answerer_user_id, Q.body as question_body, Q.tags as question_tags, (A.CreationDate - Q.CreationDate) as question_age_at_answer, Q.CreationDate as question_date, A.CreationDate as answer_date, A.score as answer_score, Q.score as question_score
    FROM Posts AS A JOIN Posts as Q on A.ParentId = Q.Id
    WHERE A.PostTypeId = {{answerPostType}} AND Q.PostTypeId = {{questionPostType}} {} {}
    ORDER BY A.CreationDate
    """.format(start_time_cond, end_time_cond), use_macros=True)

    print("all_answer_events_iterator will go through {} events until {}".format(len(all_answers), all_answers.answer_date.max()))

    last_date = make_datetime('01.01.1900 00:00')
    for i in range(len(all_answers)):
        current = all_answers.iloc[i]
        event_date = current.answer_date

        assert(last_date < event_date)
        last_date = event_date

        # user_id = current.answerer_user_id
        # actually_answered = current.question_id
        #
        # sample = (event_date, user_id, actually_answered, current)
        yield current




def shuffle_columns_within_group(dataframe, mask, column_names):
    """for all rows where mask is true, shuffle the values of the columns given by column names."""
    df = dataframe.copy()

    n_elem = np.count_nonzero(mask)
    new_order = np.random.permutation(n_elem)

    for col_name in column_names:
        current_values = df.loc[mask, col_name]
        shuffled_values = current_values[new_order]
        df.loc[mask, col_name] = shuffled_values

    return df
