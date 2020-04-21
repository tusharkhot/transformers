import json

from modularqa.drop.drop_utils import format_drop_answer


class DatasetReader:

    def read_examples(self, file):
        return NotImplementedError("read_examples not implemented by " + self.__class__.__name__)


class HotpotQAReader(DatasetReader):

    def read_examples(self, file):
        with open(file, 'r') as input_fp:
            input_json = json.load(input_fp)

        for entry in input_json:
            yield {
                "qid": entry["_id"],
                "query": entry["question"],
                # metadata
                "answer": entry["answer"],
                "question": entry["question"],
                "type": entry["type"],
                "level": entry["level"]
            }


class DropReader(DatasetReader):

    def read_examples(self, file):
        with open(file, 'r') as input_fp:
            input_json = json.load(input_fp)

        for paraid, item in input_json.items():
            para = item["passage"]
            for qa_pair in item["qa_pairs"]:
                question = qa_pair["question"]
                qid = qa_pair["query_id"]
                answer, _ = format_drop_answer(qa_pair["answer"], "drop")
                yield {
                    "qid": qid,
                    "query": question,
                    # metadata
                    "answer": answer,
                    "question": question
                }
