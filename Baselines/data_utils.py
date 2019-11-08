from datetime import datetime
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

    def age(self, key):
        timepoint = self._ensure_datetime(key)
        bin_id = self._last_id_before(timepoint)
        _age = timepoint - self.bin_edges(bin_id)
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








