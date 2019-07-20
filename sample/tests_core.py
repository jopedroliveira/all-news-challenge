#!/usr/bin/python

import unittest
from core import *


class TestsCore(unittest.TestCase):

    def test_text_clean(self):
        self.assertAlmostEqual(core.text_clean("test"), "test")
        self.assertAlmostEqual(core.text_clean("123"), "")
        self.assertAlmostEqual(core.text_clean("%$fas"), "fas")

    def test_text_stem(self):
        self.assertAlmostEqual(core.text_stem("testing"), "test ")
        self.assertAlmostEqual(core.text_stem("123"),"123 ")
        self.assertAlmostEqual(core.text_stem("well played darling"),"well play "
                                                                "darl ")

    def compute_dataset_score(self):
        # def test_compute_database_score(self):
        # # TODO
        return True

