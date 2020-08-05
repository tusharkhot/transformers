from elasticsearch import Elasticsearch

from modularqa.retrievers.retriever import Retriever


class EsRetriever(Retriever):

    def __init__(self, es_host, es_index, num_es_hits):
        super().__init__()
        self.es = Elasticsearch([es_host], retries=3, timeout=180)
        self.es_index = es_index
        self.num_es_hits = num_es_hits

    def search_es(self, question, num_es_hits):
        body = {
            "from": 0,
            "size": num_es_hits,
            "query": {
                "bool": {
                    "must": [
                        {"match": {
                            "para": question
                        }}
                    ],
                    "filter": [
                        {"type": {"value": "sentence"}}
                    ]
                }
            }
        }
        res = self.es.search(index=self.es_index,
                             body=body,
                             search_type="dfs_query_then_fetch")
        if res["timed_out"]:
            raise ValueError("ElasticSearch timed out!!")
        hits = []
        for idx, es_hit in enumerate(res['hits']['hits']):
            hit_text = es_hit["_source"]["para"].strip()
            hits.append(hit_text)
        return hits

    def retrieve_paragraphs(self, qid, question):
        if qid in self.hard_coded_paras:
            return self.hard_coded_paras[qid]
        self.search_es(question, self.num_es_hits)
