#!/usr/bin/python

import unittest
from main import get_query_params

class TestsMain(unittest.TestCase):
    def test_get_query_params(self):
        dic = {"title": "", "publication": "", "author": "","content": "test"}
        self.assertDictEqual(get_query_params('test'), dic)
