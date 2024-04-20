from typing import List, Dict
import argparse

from collections import OrderedDict
from elasticsearch import Elasticsearch


class ElasticsearchRetriever:
    """
    Some useful resources for constructing ES queries:
    # https://stackoverflow.com/questions/28768277/elasticsearch-difference-between-must-and-should-bool-query
    # https://stackoverflow.com/questions/49826587/elasticsearch-query-to-match-two-different-fields-with-exact-values

    # bool/must acts as AND
    # bool/should acts as OR
    # bool/filter acts as binary filter w/o score (unlike must and should).
    """

    def __init__(self, host: str = "localhost", port: int = 9200):
        self._es = Elasticsearch([host], scheme="http", port=port, timeout=30)

    def retrieve_paragraphs(
        self,
        corpus_name: str,
        query_text: str = None,
        is_abstract: bool = None,
        allowed_titles: List[str] = None,
        allowed_paragraph_types: List[str] = None,
        query_title_field_too: bool = False,
        paragraph_index: int = None,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query = {
            "size": max_buffer_count,
            # what records are needed in result
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {
                "bool": {
                    "should": [],
                    "must": [],
                }
            },
        }

        if query_text is not None:
            # must is too strict for this:
            query["query"]["bool"]["should"].append({"match": {"paragraph_text": query_text}})

        if query_title_field_too:
            query["query"]["bool"]["should"].append({"match": {"title": query_text}})

        if is_abstract is not None:
            query["query"]["bool"]["filter"] = [{"match": {"is_abstract": is_abstract}}]

        if allowed_titles is not None:
            if len(allowed_titles) == 1:
                query["query"]["bool"]["must"] += [{"match": {"title": _title}} for _title in allowed_titles]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _title}}}} for _title in allowed_titles
                ]

        if allowed_paragraph_types is not None:
            if len(allowed_paragraph_types) == 1:
                query["query"]["bool"]["must"] += [
                    {"match": {"paragraph_type": _paragraph_type}} for _paragraph_type in allowed_paragraph_types
                ]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _paragraph_type}}}}
                    for _paragraph_type in allowed_paragraph_types
                ]

        if paragraph_index is not None:
            query["query"]["bool"]["should"].append({"match": {"paragraph_index": paragraph_index}})

        assert query["query"]["bool"]["should"] or query["query"]["bool"]["must"]

        if not query["query"]["bool"]["must"]:
            query["query"]["bool"].pop("must")

        if not query["query"]["bool"]["should"]:
            query["query"]["bool"].pop("should")

        result = self._es.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["paragraph_text"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())

        retrieval = sorted(retrieval, key=lambda e: e["_score"], reverse=True)
        retrieval = retrieval[:max_hits_count]
        for retrieval_ in retrieval:
            retrieval_["_source"]["score"] = retrieval_["_score"]
        retrieval = [e["_source"] for e in retrieval]

        if allowed_titles is not None:
            lower_allowed_titles = [e.lower().strip() for e in allowed_titles]
            retrieval = [item for item in retrieval if item["title"].lower().strip() in lower_allowed_titles]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval

    def retrieve_titles(
        self,
        query_text: str,
        corpus_name: str,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query = {
            "size": max_buffer_count,
            # what records are needed in the result.
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {
                "bool": {
                    "must": [
                        {"match": {"title": query_text}},
                    ],
                    "filter": [
                        {"match": {"is_abstract": True}},  # so that same title doesn't show up many times.
                    ],
                }
            },
        }

        result = self._es.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["title"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())[:max_hits_count]

        retrieval = [e["_source"] for e in retrieval]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="retrieve paragraphs or titles")
    parser.add_argument(
        "dataset_name", type=str, help="dataset_name", choices={"hotpotqa", "2wikimultihopqa", "iirc", "musique"}
    )
    parser.add_argument("--host", type=str, help="host", default="localhost")
    parser.add_argument("--port", type=int, help="port", default=9200)
    args = parser.parse_args()

    retriever = ElasticsearchRetriever(
        corpus_name=args.dataset_name,
        host=args.host,
        port=args.port,
    )

    print("\n\nRetrieving Titles ...")
    results = retriever.retrieve_titles("injuries")
    for result in results:
        print(result)

    print("\n\nRetrieving Paragraphs ...")
    results = retriever.retrieve_paragraphs("injuries")
    for result in results:
        print(result)
