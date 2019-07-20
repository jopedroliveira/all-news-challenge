#!/usr/bin/python

import unittest
import analyzer
import pandas as pd
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal

class TestsAnalyzer(unittest.TestCase):

    def test_load_set(self):
        analyz = analyzer.Analyzer()
        compare_data = pd.read_pickle("../assets/prepared_data_tests.pkl")
        assert_frame_equal(analyz.load_set(
            "../assets/prepared_data_tests.pkl"), compare_data)

    def test_get_score(self):
        analyz = analyzer.Analyzer()
        compare_data = pd.read_pickle("../assets/get_score_test.pkl")
        assert_series_equal(analyz.get_score("title","Rift"), compare_data)


    def test_compute_score(self):
        analyz = analyzer.Analyzer()
        compare_data = pd.read_pickle("../assets/compute_score_test.pkl")
        assert_frame_equal(analyz.compute_score({"title":"Rift"}), compare_data)

    def test_perform_search(self):
        analyz = analyzer.Analyzer()
        total = pd.read_pickle("../assets/perform_search_tot_test.pkl")
        df = pd.read_pickle("../assets/perform_search_df_test.pkl")
        out1,out2 = analyz.perform_search({"title":"Rift"})
        self.assertEqual(out1,total)
        assert_frame_equal(out2, df)

