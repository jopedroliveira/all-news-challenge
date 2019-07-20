#!/usr/bin/python


import re
import analyzer
import sys
import core


def get_query_params(query):
    '''
    :param query:
    :return search_attr_results:
    find tags in query
    assuming query only contains context_
    # text_clean(query)
    # stem_sentence(query)
    # get score for each word on query in the dataset
    # compute the intercpetion
    # return title and content
    '''
    search_attrs = ["title", "publication", "author", "content"]
    search_attr_result = {"title": "", "publication": "", "author": "",
                          "content": ""}
    matches = [x for x in search_attrs if x in query]
    if len(matches) > 0:
        for match in matches:
            search_aux = re.search("(%s\:)(.*)(\s\S+:)*" % match,
                                   query).group(2)
            search_attr_result[match] = re.sub("(\s\S+:).*", '', search_aux)
    else:
        search_attr_result["content"] = query
    return search_attr_result


def main(argv):
    # if len(argv) == 0:
    #     tfidf_title_data, tfidf_author_data, tfidf_publication_data, \
    #     tfidf_content_data = core.compute_dataset_score()
    # elif len(argv) == 1:
    #     tfidf_title_data, tfidf_author_data, tfidf_publication_data, \
    #     tfidf_content_data = core.compute_dataset_score(argv[1])
    # elif len(argv) == 2:
    #     tfidf_title_data, tfidf_author_data, tfidf_publication_data, \
    #     tfidf_content_data = core.compute_dataset_score(argv[1], argv[2])

    print("Please wait while the engine is being prepared...\n")
    analyz = analyzer.Analyzer()

    query_input = ""
    repeat = True
    while repeat:
        while len(query_input) < 1:
            query_input = input("Type your search (click return to "
                                "search) or type quit! to exit:\n")

        if query_input == "quit!":
            repeat = False
            break
        query_params = get_query_params(query_input)
        total, articles = analyz.perform_search(query_params)
        print("Total of articals: %f\n" % total)
        if total > 0:
            print("Articles: \n%s\n" % articles.to_string())
        else:
            print("No results found for your search: %s\n", query_input)

        query_input = ""


if __name__ == "__main__":
    main(sys.argv[1:])
