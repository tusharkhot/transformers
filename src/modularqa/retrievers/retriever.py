
class Retriever:

    def __init__(self):
        self.hard_coded_paras = {}

    def set_paras(self, id, paras):
        self.hard_coded_paras[id] = paras

    @staticmethod
    def load_retriever(hotpotqa_file=None, drop_file=None, squad_file=None,
                       es_host=None, es_index="hpqa_para",
                       tfidf_file=None, db_file=None):
        from modularqa.retrievers.es_retriever import EsRetriever
        from modularqa.retrievers.file_retriever import FileRetriever
        from modularqa.retrievers.tfidf_retriever import TfidfDocRanker

        if es_host and tfidf_file:
            raise ValueError("Only one of es_host or tfidf_file should be set!")
        if es_host:
            return EsRetriever(es_host=es_host, es_index=es_index)
        elif tfidf_file:
            return TfidfDocRanker(tfidf_file=tfidf_file, db_file=db_file)
        else:
            return FileRetriever(hotpotqa_file=hotpotqa_file, drop_file=drop_file,
                                 squad_file=squad_file)

    def retrieve_paragraphs(self, qid, question):
        raise NotImplementedError("Class should implement retrieve_paragraphs")
