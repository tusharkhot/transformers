import json
import re

from modularqa.inference.executer import OperationExecuter
from modularqa.inference.executer_utils import build_models
from modularqa.inference.model_search import ParticipantModel


class ModelRouter(ParticipantModel):

    def __init__(self, question_pattern=None):
        if question_pattern:
            self.question_pattern = re.compile(question_pattern)
        else:
            self.question_pattern = re.compile("^\(([^\)]+)\)(.*)$")

    def query(self, state, debug=False):
        data = state._data
        question = data["question_seq"][-1]
        qid = data["qid"]
        new_state = state.copy()
        if question == "[EOQ]":
            new_state._next = "EOQ"
            return [new_state]
        m = self.question_pattern.match(question)
        if m:
            send_to = m.group(1)
            new_q = m.group(2).strip()
            if debug: print("<ROUTE>: %s, qid=%s, route=%s" % (new_q, qid, send_to))

            new_state._data["model_seq"].append(send_to)
            new_state._data["question_seq"][-1] = new_q
            new_state._data["command_seq"].append("route")
            new_state._next = send_to
        else:
            print("Question didn't match format!: {}".format(question))
            # new_state = state.copy()
            # new_state._score = float('inf')
            # new_state._data["model_seq"].append("N/A")
            # new_state._data["answer_seq"].append("N/A")
            # new_state._data["command_seq"].append("route")
            return []

        return [new_state]


class ExecutionRouter(ParticipantModel):
    def __init__(self, remodel_file):
        if remodel_file:
            with open(remodel_file, "r") as input_fp:
                input_json = json.load(input_fp)
            self.kb_lang_groups = []
            self.qid_to_kb_lang_idx = {}
            for input_item in input_json:
                kb = input_item["kb"]
                pred_lang = input_item["pred_lang_config"]
                idx = len(self.kb_lang_groups)
                self.kb_lang_groups.append((kb, pred_lang))
                for qa_pair in input_item["qa_pairs"]:
                    qid = qa_pair["id"]
                    self.qid_to_kb_lang_idx[qid] = idx
            self.operation_regex = re.compile("\[(.*)\] <(.*)> (.*)")

    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        """
        ## the data
        data = state._data
        question = data["question_seq"][-1]
        qid = data["qid"]
        (kb, pred_lang) = self.kb_lang_groups[self.qid_to_kb_lang_idx[qid]]
        model_lib = build_models(pred_lang, kb)
        ### run the model (as before)
        if debug: print("<OPERATION>: %s, qid=%s" % (question, qid))
        m = self.operation_regex.match(question)
        assignment = {}
        for ans_idx, ans in enumerate(data["answer_seq"]):
            assignment["#" + str(ans_idx + 1)] = json.loads(ans)
        executer = OperationExecuter(model_library=model_lib)
        answers, facts_used = executer.execute_operation(operation=m.group(1),
                                                         model=m.group(2),
                                                         question=m.group(3),
                                                         assignments=assignment)
        if isinstance(answers, list) and len(answers) == 0:
            return []
        # copy state
        new_state = state.copy()

        ## TODO update score?

        ## add answer
        new_state._data["answer_seq"].append(json.dumps(answers))
        new_state._data["para_seq"].append("")
        new_state._data["command_seq"].append("qa")

        ## change output
        new_state.last_output = answers
        new_state._next = "gen"

        return [new_state]
