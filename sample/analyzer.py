#!/usr/bin/python


from sklearn.feature_extraction.text import CountVectorizer
from core import *
import pandas as pd
import os
import numpy as np


class Analyzer:

    def load_set(self, set_path):
        if not os.path.exists(set_path):
            raise FileNotFoundError("The provided path does not contain a valid file. "
                            "\nPlease execute the program in training mode")
        return pd.read_pickle(set_path)

    def get_score(self, key, value):
        inters = ""
        text = core.text_clean(value)
        text = core.text_stem(text)
        cv = CountVectorizer(stop_words='english')
        cv.fit_transform([text])
        words = cv.get_feature_names()

        if key.lower() == "title":
            inters = self.title_set[self.title_set.columns.intersection(words)]
            if len(inters.columns) == 0:
                inters = pd.DataFrame(np.zeros((len(inters.index), 1)))
            inters = inters.prod(axis=1)
            inters.columns = ['score']
        elif key.lower() == "author":
            inters = self.author_set[self.author_set.columns.intersection(
                words)]
            if len(inters.columns) == 0:
                inters = pd.DataFrame(np.zeros((len(inters.index), 1)))
            inters = inters.prod(axis=1)
            inters.columns = ['score']
        elif key.lower() == "publication":
            inters = self.pub_set[self.pub_set.columns.intersection(words)]
            if len(inters.columns) == 0:
                inters = pd.DataFrame(np.zeros((len(inters.index), 1)))
            inters = inters.prod(axis=1)
            inters.columns = ['score']
        elif key.lower() == "content":
            inters = self.content_set[
                self.content_set.columns.intersection(words)]
            if len(inters.columns) == 0:
                inters = pd.DataFrame(np.zeros((len(inters.index), 1)))
            inters = inters.prod(axis=1)
            inters.columns = ['score']
        else:
            inters = pd.DataFrame(np.zeros((len(self.title_set.index), 1)))
        return inters

    def compute_score(self, dic_params):
        pdres = None
        for k in dic_params:
            if bool(dic_params[k]):
                sc = self.get_score(k, dic_params[k])
                if pdres is None:
                    pdres = sc
                else:
                    pdres = pd.concat([pdres, sc], axis=1)
                    pdres = pdres.prod(axis=1)
        return pd.DataFrame(pdres, columns=["score"])

    def perform_search(self, dic_params):
        selection = pd.DataFrame(np.zeros((1, 1)))
        score_values = self.compute_score(dic_params)
        score_values = score_values.sort_values(by='score', ascending=False)

        match_count = score_values.apply(
            lambda x: True if x['score'] > 0 else False, axis=1)
        total_articles = len(match_count[match_count == True].index)
        if total_articles > 0:
            articles_index = score_values.index.values[0:20]
            selection_aux_var = articles_index.tolist()
            selected_articles = self.data_set.iloc[selection_aux_var]
            selection = pd.concat([score_values[0: 20], selected_articles],
                                  axis=1)

        return total_articles, selection

    def __init__(self, data_set=None, title_set=None, pub_set=None,
                 author_set=None, content_set=None):
        if data_set and title_set and pub_set and author_set and content_set:
            self.data_set = self.load_set(data_set)
            self.title_set = self.load_set(title_set)
            self.pub_set = self.load_set(pub_set)
            self.author_set = self.load_set(author_set)
            self.content_set = self.load_set(content_set)
        else:
            self.data_set, self.title_set, self.pub_set, \
            self.author_set, self.content_set = core.compute_dataset_score()
            # TODO
