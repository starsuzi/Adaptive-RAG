from typing import List, Dict

from elasticsearch_retriever import ElasticsearchRetriever


class UnifiedRetriever:
    """
    This class wrapper multiple different retrievers we experimented with.
    Since we settled with Elasticsearch, I've removed code related to other
    retrievers from here. Still keeping the wrapper for reproducibility.
    """

    def __init__(
        self,
        host: str = "http://localhost/",
        port: int = 9200,
    ):
        self._elasticsearch_retriever = ElasticsearchRetriever(host=host, port=port)

    def retrieve_from_elasticsearch(
        self,
        query_text: str,
        max_hits_count: int = 3,
        max_buffer_count: int = 100,
        document_type: str = "paragraph_text",
        allowed_titles: List[str] = None,
        allowed_paragraph_types: List[str] = None,
        paragraph_index: int = None,
        corpus_name: str = None,
    ) -> List[Dict]:

        assert document_type in ("title", "paragraph_text", "title_paragraph_text")

        if paragraph_index is not None:
            assert (
                document_type == "paragraph_text"
            ), "paragraph_index not valid input for the document_type of paragraph_text."

        if self._elasticsearch_retriever is None:
            raise Exception("Elasticsearch retriever not initialized.")

        if document_type in ("paragraph_text", "title_paragraph_text"):
            is_abstract = True if corpus_name == "hotpotqa" else None  # Note "None" and not False
            query_title_field_too = document_type == "title_paragraph_text"
            paragraphs_results = self._elasticsearch_retriever.retrieve_paragraphs(
                query_text=query_text,
                is_abstract=is_abstract,
                max_hits_count=max_hits_count,
                allowed_titles=allowed_titles,
                allowed_paragraph_types=allowed_paragraph_types,
                paragraph_index=paragraph_index,
                corpus_name=corpus_name,
                query_title_field_too=query_title_field_too,
                max_buffer_count=max_buffer_count,
            )

        elif document_type == "title":
            paragraphs_results = self._elasticsearch_retriever.retrieve_titles(
                query_text=query_text, max_hits_count=max_hits_count, corpus_name=corpus_name
            )

        return paragraphs_results
