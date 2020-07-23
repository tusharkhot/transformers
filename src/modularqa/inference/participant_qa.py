import json
import re
import string

from modularqa.con_gen.constraints import QAConstraint
from modularqa.inference.model_search import ParticipantModel
from modularqa.utils.math_qa import MathQA
from modularqa.utils.qa import LMQuestionAnswerer, BoolQuestionAnswerer


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
            return []

        return [new_state]


class LMQAParticipant(LMQuestionAnswerer, ParticipantModel):
    """A slightly modified BERT QA model that has a new method, called `query`,
    which will be used directly by the controller (in place of `answer_question_only`.

    Hopefully this gives some idea of how to implement a method for a model that works
    in the controller.
    """

    def __init__(self, max_answers=1, exp_ans_file=None, output_qa=None, **kwargs):
        self.max_answers = max_answers
        self.exp_ans_map = self.load_exp_ans(exp_ans_file) if exp_ans_file else None
        self.output_qa = output_qa
        super(LMQAParticipant, self).__init__(**kwargs)

    def load_exp_ans(self, exp_ans_file):
        exp_ans_map = {}
        with open(exp_ans_file, "r") as exp_ans_fp:
            for line in exp_ans_fp:
                input_json = json.loads(line)
                qaconstraints = [QAConstraint.from_json(c) for c in input_json["constraints"]]
                qid = input_json["id"]
                if qid not in exp_ans_map:
                    exp_ans_map[qid] = {}
                for idx, constraint in enumerate(qaconstraints):
                    if constraint.model == "FAILED" or constraint.model == "OUTOFSCOPE":
                        break
                    if not constraint.aconstraint.exp_ans:
                        break
                    if idx not in exp_ans_map[qid]:
                        exp_ans_map[qid][idx] = set()
                    exp_ans_map[qid][idx].add(constraint.aconstraint.exp_ans)
        return exp_ans_map

    def get_exp_ans(self, qid, idx):
        if qid in self.exp_ans_map and idx in self.exp_ans_map[qid]:
            return self.exp_ans_map[qid][idx]
        return []

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
        if question == "[EOQ]":
            new_state = state.copy()
            new_state._next = "EOQ"
            return [new_state]

        ### run the model (as before)
        if debug: print("<BERTQA>: %s, qid=%s" % (question, qid))
        model_output = self.answer_question_only(question, qid)[:self.max_answers]

        ## will deteremine stopping condition
        max_answers = 4
        new_states = []

        for bert_out in model_output:
            if self.output_qa:
                exp_ans = self.get_exp_ans(qid, len(data["answer_seq"]))
                with open(self.output_qa, 'a') as qa_fp:
                    row = [qid, question, "==".join(exp_ans), bert_out.answer, bert_out.para_text]
                    qa_fp.write("\t".join(row) + "\n")
                if len(exp_ans) == 1:
                    # if there is an unique answer
                    exp_ans_val = list(exp_ans)[0]
                    if exp_ans_val in bert_out.para_text:
                        bert_out.answer = exp_ans_val
                    else:
                        bert_out.answer = ""
            if bert_out.answer == "":
                continue
            if debug:
                print("<BERTQA> Ans: {} Score: {} Para: {}".format(bert_out.answer,
                                                                   bert_out.score,
                                                                   bert_out.para_text[:15]))
            # copy state
            new_state = state.copy()

            ## add score
            # new_state._score += bert_out.score
            ## strip unnecessary punctuations
            answer = bert_out.answer.strip(string.punctuation)
            if re.match("^[0-9,.]+$", answer):
                answer = answer.replace(",", "")
            new_state._data["answer_seq"].append(answer)
            new_state._data["para_seq"].append(bert_out.para_text)

            ## add initial question + answer as tuple to `question_seq`
            # new_state._data["question_seq"][-1] = (question, bert_out.answer)
            new_state._data["command_seq"].append("qa")

            ## change output
            new_state.last_output = answer

            ## determine next state based on
            if len(new_state._data["answer_seq"]) >= max_answers:
                new_state._next = "EOQ"
            else:
                new_state._next = "gen"
            new_states.append(new_state)

        return new_states


class MathQAParticipant(MathQA, ParticipantModel):

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

        ### run the model (as before)
        if debug: print("<MATHQA>: %s, qid=%s" % (question, qid))
        answer = self.answer_question(question)
        if debug:
            print("<MathQA> Ans: {}".format(answer))
        if answer == "":
            return []
        # copy state
        new_state = state.copy()

        ## TODO update score?

        ## add answer
        new_state._data["answer_seq"].append(answer)
        new_state._data["para_seq"].append("")
        new_state._data["command_seq"].append("qa")

        ## change output
        new_state.last_output = answer
        new_state._next = "gen"

        return [new_state]


class BoolQAParticipant(BoolQuestionAnswerer, ParticipantModel):

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

        ### run the model (as before)
        if debug: print("<BOOLQ>: %s, qid=%s" % (question, qid))
        model_answer = self.answer_question_only(question=question, qid=qid)
        if debug:
            print("<BOOLQ> Ans: {} Score: {} Para: {}".format(model_answer.answer,
                                                              model_answer.score,
                                                              model_answer.para_text[:15]))
        # copy state
        new_state = state.copy()

        ## TODO update score?

        ## add answer
        new_state._data["answer_seq"].append(model_answer.answer)
        new_state._data["para_seq"].append(model_answer.para_text)
        new_state._data["command_seq"].append("qa")

        ## change output
        new_state.last_output = model_answer.answer
        new_state._next = "gen"

        return [new_state]
