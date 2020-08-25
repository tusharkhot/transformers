import json

from modularqa.retrievers.retriever import Retriever


class FileRetriever(Retriever):

    def __init__(self, hotpotqa_file=None, drop_file=None, squad_file=None):
        super().__init__()
        # Load all the provided
        self._qid_doc_map = {}
        if hotpotqa_file is not None:
            self._qid_doc_map.update(self.get_qid_doc_map_hotpotqa(hotpotqa_file))
        if drop_file is not None:
            self._qid_doc_map.update(self.get_qid_doc_map_drop(drop_file))
        if squad_file is not None:
            self._qid_doc_map.update(self.get_qid_doc_map_squad(squad_file))

    def output_for_skiff_demo(self):
        # OUTPUT FOR SKIFF DEMO
        dropq_arr = [
            ("5a87ab905542996e4f3088c1",
             "The arena where the Lewiston Maineiacs played their home games can seat how many people?"),
            ("5a80840f554299485f59863b",
             "What WB supernatural drama series was Jawbreaker star Rose Mcgowan best known for being in?"),
            ("a097cb28-cd2a-40f8-a860-28b38d0372a8",
             "How many years after the Battle of Nieuwpoort did French forces first begin occupying the city?"),
            ("17169f44-cb53-4809-9dc7-19872cc70480",
             "How many months after 25 villages belonging to Memmingen rebelled were the Twelve Articles agreed on by the Upper Swabian Peasants Confederation?")
        ]
        output_json = []
        for id, question in dropq_arr:
            output_json.append({
                "question": question,
                "paragraphs": self.retrieve_paragraphs(id, question)
            })
        print(json.dumps(output_json, indent=2))

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

    def get_qid_doc_map_hotpotqa(self, para_files, only_gold_para=False):
        qid_doc_map = {}
        for para_file in para_files.split(","):
            print("Loading paragraphs from {}".format(para_file))
            with open(para_file, "r") as input_fp:
                input_json = json.load(input_fp)
            for entry in input_json:
                supporting_docs = {doc for (doc, idx) in entry["supporting_facts"]}
                title_doc_map = {}
                qid = entry["_id"]
                for title, document in entry["context"]:
                    if not only_gold_para or title in supporting_docs:
                        title_doc_map[title] = [doc.strip() for doc in document]

                qid_doc_map[qid] = title_doc_map

        return qid_doc_map

    def get_qid_doc_map_drop(self, drop_files):
        qid_doc_map = {}
        for drop_file in drop_files.split(","):
            print("Loading paragraphs from {}".format(drop_file))
            with open(drop_file, "r") as input_fp:
                input_json = json.load(input_fp)

            for paraid, item in input_json.items():
                para = item["passage"]
                title_doc_map = {paraid: [para]}
                for qa_pair in item["qa_pairs"]:
                    qid = qa_pair["query_id"]
                    qid_doc_map[qid] = title_doc_map

        return qid_doc_map

    def get_qid_doc_map_squad(self, squad_files):
        qid_doc_map = {}
        for squad_file in squad_files.split(","):
            print("Loading paragraphs from {}".format(squad_file))
            with open(squad_file, "r") as input_fp:
                input_json = json.load(input_fp)

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
