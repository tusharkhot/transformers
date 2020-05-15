import re

from modularqa.con_gen.verifiers import LMSpansQuestionVerifier, LMQuestionVerifier
from modularqa.inference.model_search import ParticipantModel
from modularqa.utils.classifier import LMClassifier
from modularqa.utils.qa import LMQuestionAnswerer, BoolQuestionAnswerer
from modularqa.utils.math_qa import MathQA

class ModelRouter(ParticipantModel):

    def __init__(self, question_pattern=None):
        if question_pattern:
            self.question_pattern = re.compile(question_pattern)
        else:
            self.question_pattern = re.compile("^\(([^\)]+)\) (.*)$")

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
            new_q = m.group(2)
            if debug: print("<ROUTE>: %s, qid=%s, route=%s" % (new_q, qid, send_to))

            new_state._data["model_seq"].append(send_to)
            new_state._data["question_seq"][-1] = new_q
            new_state._data["command_seq"].append("route")
            new_state._next = send_to
        else:
            print("Question didn't match format!: {}".format(question))
            #new_state = state.copy()
            #new_state._score = float('inf')
            return []

        return [new_state]


class LMQAParticipant(LMQuestionAnswerer, ParticipantModel):
    """A slightly modified BERT QA model that has a new method, called `query`,
    which will be used directly by the controller (in place of `answer_question_only`.

    Hopefully this gives some idea of how to implement a method for a model that works
    in the controller.
    """

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
        model_output = self.answer_question_only(question, qid)[:1]

        ## will deteremine stopping condition
        max_answers = 4
        new_states = []

        for bert_out in model_output:
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
            ## add answer
            new_state._data["answer_seq"].append(bert_out.answer)

            ## add initial question + answer as tuple to `question_seq`
            # new_state._data["question_seq"][-1] = (question, bert_out.answer)
            new_state._data["command_seq"].append("qa")

            ## change output
            new_state.last_output = bert_out.answer

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
        answer = self.answer_question_only(question=question, qid=qid)
        if debug:
            print("<BOOLQ> Ans: {}".format(answer))
        if answer == "":
            return []
        # copy state
        new_state = state.copy()

        ## TODO update score?

        ## add answer
        new_state._data["answer_seq"].append(answer)
        new_state._data["command_seq"].append("qa")

        ## change output
        new_state.last_output = answer
        new_state._next = "gen"

        return [new_state]
