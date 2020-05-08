from copy import deepcopy

import torch
from torch.nn.functional import softmax

from modularqa.inference.model_search import ParticipantModel
from modularqa.utils.classifier import LMClassifier
from modularqa.utils.seq_utils import get_sequence_representation
from modularqa.utils.seq_utils import score_question_answer_chain
from transformers import InputExample, \
    glue_convert_examples_to_features, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class QualityCheckerExample(ParticipantModel):

    def query(self, state, debug=False):
        """Checks the quality of a given question. In this case, it does
        nothing. This shows how you might build functions that do something,
        e.g., change the score of an example, but that don't change the state.

        :param state: pass
        """
        ## state data
        data = state.data
        ## copy the state
        new_state = state.copy()
        new_state._next = "qa"
        if debug: print("<QUALITYCHECK>: %s" % data["question_seq"][-1])

        ## checks last question (stored in data["question_seq"]) and sees
        ## if it has any overlap with original question (stored in data["query"])
        initial_question_words = set(data["query"].split())
        if not [i for i in data["question_seq"][-1].split() \
                if i not in initial_question_words]:
            ## if so, manipulates the score to be infinity
            new_state.last_output = "terrible quality!"
            new_state._score = float('inf')

        else:
            new_state.last_output = "good quality!"

        return new_state


class ChainOverlapScorer(ParticipantModel):

    def query(self, state, debug=False):
        ## state data
        data = state.data
        ## copy the state
        new_state = state.copy()
        new_state._next = "qa"
        origq = data["query"]
        qchain = data["question_seq"]
        achain = data["answer_seq"]

        if debug:
            print("<QUALITYCHECK>: Qs: {} As: {} Q: {}".format(
                ", ".join(qchain), ", ".join(achain), origq))
        if qchain[-1] == "[EOQ]":
            new_qchain = deepcopy(qchain)
            new_qchain.pop(-1)
            new_tok_score, missed_tok_score, new_toks, missed_toks, unmatched_answers = \
                score_question_answer_chain(new_qchain, achain, origq, repeat_ok=False,
                                            score_answers=True)
            if unmatched_answers > 0:
                if debug:
                    print("Unmatched answers! Rejecting!"
                          "Qs: {} As: {} Q: {}".format(", ".join(qchain), ", ".join(achain), origq))
                new_state._score = float("inf")
            else:
                new_state._score = new_tok_score + missed_tok_score
        else:
            new_tok_score, missed_tok_score, new_toks, missed_toks = \
                score_question_answer_chain(qchain, achain, origq, repeat_ok=False,
                                            score_answers=False)
            new_state._score = new_tok_score

        new_state.last_output = "Missed: {} New: {}".format(",".join(missed_toks),
                                                            ",".join(new_toks))

        return new_state


class LMQualityChecker(LMClassifier, ParticipantModel):


    def query(self, state, debug=False):
        ## state data
        data = state.data
        ## copy the state
        new_state = state.copy()
        new_state._next = "qa"
        origq = data["query"]
        qchain = data["question_seq"]
        achain = data["answer_seq"]
        mchain = data["model_seq"]
        score_answers = False

        if debug:
            print("<QUALITYCHECK>: Qs: {} As: {} Q: {}".format(
                ", ".join(qchain), ", ".join(achain), origq))

        # check for correct usage of answers
        if qchain[-1] == "[EOQ]":
            new_qchain = deepcopy(qchain)
            new_qchain.pop(-1)
            new_tok_score, missed_tok_score, new_toks, missed_toks, unmatched_answers = \
                score_question_answer_chain(new_qchain, achain, origq, repeat_ok=False,
                                            score_answers=True)
            if unmatched_answers > 0:
                if debug:
                    print("Unmatched answers! Rejecting!"
                          "Qs: {} As: {} Q: {}".format(", ".join(qchain), ", ".join(achain), origq))
                new_state._score = float("inf")
                new_state.last_output = "Rejected for bad answers"
                return new_state

        sequence = get_sequence_representation(origq, qchain, achain, mchain, for_generation=False)
        output_probs = self.score_sequence(sequence1=sequence)

        #print(sequence, output_probs)
        new_state._score += output_probs[0] # higher is worse; so take prob of 0
        new_state.last_output = "Score: {}".format(output_probs)
        return new_state
