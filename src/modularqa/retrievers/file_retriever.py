import json

from modularqa.retrievers.retriever import Retriever


class FileRetriever(Retriever):

    def __init__(self, hotpotqa_file=None, drop_file=None, squad_file=None):
        if hotpotqa_file is not None:
            self._qid_doc_map = self.get_qid_doc_map_hotpotqa(hotpotqa_file)
        elif drop_file is not None:
            self._qid_doc_map = self.get_qid_doc_map_drop(drop_file)
        elif squad_file is not None:
            self._qid_doc_map = self.get_qid_doc_map_squad(squad_file)
        else:
            self._qid_doc_map = None

    def retrieve_paragraphs(self, qid, question):
        if qid in self.hard_coded_paras:
            return self.hard_coded_paras[qid]
        if qid not in self._qid_doc_map:
            raise ValueError("QID: {} not found in the qid->doc map loaded.".format(qid))
        else:
            doc_map = self._qid_doc_map[qid]
            # ignore title as it is not present in SQuAD models
            paragraphs = [" ".join(doc) for (t, doc) in doc_map.items()]
            return paragraphs

    def get_qid_doc_map_hotpotqa(self, para_file, only_gold_para=False):
        print("Loading paragraphs from {}".format(para_file))
        with open(para_file, "r") as input_fp:
            input_json = json.load(input_fp)
        qid_doc_map = {}
        for entry in input_json:
            supporting_docs = {doc for (doc, idx) in entry["supporting_facts"]}
            title_doc_map = {}
            qid = entry["_id"]
            for title, document in entry["context"]:
                if not only_gold_para or title in supporting_docs:
                    title_doc_map[title] = [doc.strip() for doc in document]

            qid_doc_map[qid] = title_doc_map

        return qid_doc_map


    def get_qid_doc_map_drop(self, drop_file):
        print("Loading paragraphs from {}".format(drop_file))
        with open(drop_file, "r") as input_fp:
            input_json = json.load(input_fp)
        qid_doc_map = {}
        for paraid, item in input_json.items():
            para = item["passage"]
            title_doc_map = {paraid: [para]}
            for qa_pair in item["qa_pairs"]:
                qid = qa_pair["query_id"]
                qid_doc_map[qid] = title_doc_map

        return qid_doc_map


    def get_qid_doc_map_squad(self, squad_file):
        print("Loading paragraphs from {}".format(squad_file))
        with open(squad_file, "r") as input_fp:
            input_json = json.load(input_fp)

        qid_doc_map = {}
        for data in input_json["data"]:
            title = data.get("title", "")
            if title:
                title_prefix = title.replace("_", " ") + "||"
            else:
                title_prefix = ""

            for paragraph in data["paragraphs"]:
                para = title_prefix + paragraph["context"].replace("\n", " ")
                title_doc_map = {title: [para]}
                for qa in paragraph["qas"]:
                    qid = qa["id"]
                    qid_doc_map[qid] = title_doc_map
        return qid_doc_map
