#!/usr/bin/python

import pandas as pd
import re
import string
import os
import pickle


from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def text_clean(text):
    '''
    This function allows to prepare the text for pre-processing.
    It makes text lowercase, removes unknown unicode chars, ponctuation and
    remove words containing numbers.
    :param text: string
    :return text: string
    '''
    try:
        # remove non-ascii characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[\xe1\xe9\xed]', '', text)
        # remove numbers
        text = re.sub('\w*\d\w*', '', text)
        # remove ponctucation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        # lowercase
        text = text.lower()
    except:
        return "na"
    return text

def text_stem(text):
    '''
    This function allows to pre-processing a given test, stemming and
    tokenizing the text.
    :param text: string
    :return text: string
    '''
    token_words = word_tokenize(text)
    text_stem = []
    stemmer = PorterStemmer()
    for word in token_words:
        text_stem.append(stemmer.stem(word))
        text_stem.append(" ")
    return "".join(text_stem)


def compute_dataset_score(path_raw = "../assets/articles1.csv", encoding="utf-8"):

    # if not os.path.exists(path_raw):
    #     raise FileNotFoundError("The provided path does not contain a valid "
    #                             "file. \nPlease execute the program in "
    #                             "training mode")
    # rawdata = pd.read_csv(path_raw, encoding=encoding)
    # data = rawdata
    # # save for later
    # #rawdata.to_pickle('../assets/raw_data.pkl')
    #
    #
    # # pre-process dataset
    # clean_fnc = lambda x: text_clean(x)
    # data.title = pd.DataFrame(data.title.apply(clean_fnc))
    # data.publication = pd.DataFrame(data.publication.apply(clean_fnc))
    # data.author = pd.DataFrame(data.author.apply(clean_fnc))
    # data.content = pd.DataFrame(data.content.apply(clean_fnc))
    #
    # # stemming & tokenizing
    # stemming_fnc = lambda x: text_stem(x)
    # data.title = pd.DataFrame(data.title.apply(stemming_fnc))
    # data.publication = pd.DataFrame(data.publication.apply(stemming_fnc))
    # data.author = pd.DataFrame(data.author.apply(stemming_fnc))
    # data.content = pd.DataFrame(data.content.apply(stemming_fnc))

    # save for later
    # data.to_pickle('../assets/prepared_data.pkl')

    rawdata = pd.read_pickle('../assets/raw_data.pkl')
    data = pd.read_pickle('../assets/prepared_data.pkl')

    # COUNTING - TITLE
    cv = CountVectorizer(stop_words='english')
    title_data_cv = cv.fit_transform(data.title)
    # title_data_dtm_df = pd.DataFrame(title_data_cv.toarray(),
    #                                  columns=cv.get_feature_names())
    # title_data_dtm_df.to_pickle('../assets/title_dtm_df.pkl')

    # TF-IDF - TITLE
    tfidf_transformer_title = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer_title.fit(title_data_cv)
    tf_idf_vec_title = tfidf_transformer_title.transform(cv.transform(
        data.title))
    tfidf_title_data = pd.DataFrame(tf_idf_vec_title.toarray(),
                                    columns=cv.get_feature_names())
    # tfidf_title_data.to_pickle('../assets/tfidf_title_data.pkl')

    # COUNTING - PUBLICATION
    publication_data_cv = cv.fit_transform(data.publication)
    # publication_data_dtm_df = pd.DataFrame(publication_data_cv.toarray(),
    #                                        columns=cv.get_feature_names())
    # publication_data_dtm_df.to_pickle('../assets/publication_data_dtm_df.pkl')

    # TF-IDF - PUBLICATION
    tfidf_transformer_publication = TfidfTransformer(smooth_idf=True,
                                                     use_idf=True)
    tfidf_transformer_publication.fit(publication_data_cv)
    tf_idf_vec_publication = tfidf_transformer_publication.transform(
        cv.transform(data.publication))
    tfidf_publication_data = pd.DataFrame(tf_idf_vec_publication.toarray(),
                                          columns=cv.get_feature_names())
    # tfidf_publication_data.to_pickle('../assets/tfidf_publication_data.pkl')

    # COUNTING - AUTHOR
    author_data_cv = cv.fit_transform(data.author)
    # author_data_dtm_df = pd.DataFrame(author_data_cv.toarray(),
    #                                   columns=cv.get_feature_names())
    # author_data_dtm_df.to_pickle('../assets/author_data_dtm_df.pkl')

    # TF-IDF AUTHOR
    tfidf_transformer_author = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer_author.fit(author_data_cv)
    tf_idf_vec_author = tfidf_transformer_author.transform(cv.transform(
        data.author))
    tfidf_author_data = pd.DataFrame(tf_idf_vec_author.toarray(),
                                     columns=cv.get_feature_names())
    # tfidf_author_data.to_pickle('../assets/tfidf_author_data.pkl')

    # COUNTING - CONTENT
    content_data_cv = cv.fit_transform(data.content)
    # content_data_dtm_df = pd.DataFrame(content_data_cv.toarray(),
    #                                   columns=cv.get_feature_names())
    # content_data_dtm_df.to_pickle('../assets/content_data_dtm_df.pkl')

    # TF-IDF - CONTENT
    tfidf_transformer_content = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer_content.fit(content_data_cv)
    tf_idf_vec_content = tfidf_transformer_content.transform(cv.transform(
        data.content))
    tfidf_content_data = pd.DataFrame(tf_idf_vec_content.toarray(),
                                      columns=cv.get_feature_names())
    # tfidf_content_data.to_pickle('../assets/tfidf_content_data.pkl')
    return rawdata, tfidf_title_data, tfidf_publication_data, \
           tfidf_author_data, tfidf_content_data
