#!/usr/bin/python

#1 - read the file line by line
#2 - for each line pre-process it and store into a pandas structure
#3 - pre-process it

import pandas as pd
import re
import string
import pickle


from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def text_clean(text):
    '''
    This functions allows to pre-format the text before data pre-processing
    :param text:
    :return text:
    '''

    # lowercase
    text = text.lower()
    # remove non-ascii characters
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    # remove numbers
    text = re.sub('\w*\d\w*', '', text)
    # remove ponctucation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)

    return text



def stem_sentence(text):
    token_words=word_tokenize(text)
    stem_sentence=[]
    stemmer = PorterStemmer()
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)




if __name__ == "__main__":

    data = pd.read_csv('../assets/articles1.csv', encoding='utf-8')
    labels = data.columns.values
    #data.to_pickle('raw_data.pkl')

    clean_call = lambda x: text_clean(x)
    data.title = pd.DataFrame(data.title.apply(clean_call))
    data.content = pd.DataFrame(data.content.apply(clean_call))

    #save for later
    #data.to_pickle('cleaned_data.pkl')

    #stemming
    stemming = lambda x: stem_sentence(x)
    data.title = pd.DataFrame(data.title.apply(stemming))
    data.content = pd.DataFrame(data.content.apply(stemming))

    #tokenizing
    cv = CountVectorizer(stop_words='english')

    title_data_cv = cv.fit_transform(data.title)
    title_data_dtm = pd.DataFrame(title_data_cv.toarray(), columns=cv.get_feature_names())

    content_data_cv = cv.fit_transform(data.content)
    content_data_dtm = pd.DataFrame(content_data_cv.toarray(), columns=cv.get_feature_names())

    #tf-idf
    tfidf_transformer_title = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer_title.fit(title_data_cv)
    tf_idf_vec_title = tfidf_transformer_title.transform(cv.transform(data.title));
    tfidf_title_data = pd.DataFrame(tf_idf_vec_title.toarray(), columns=cv.get_feature_names())
    # aaa = title_data_tfidf[['trump']]
    #aaa.sort_values(by='trump',ascending=False)



    tfidf_transformer_content = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer_content.fit(content_data_cv)
    tf_idf_vec_content = tfidf_transformer_content.transform(cv.transform(data.content));
    tfidf_content_data = pd.DataFrame(tf_idf_vec_content.toarray(), columns=cv.get_feature_names())