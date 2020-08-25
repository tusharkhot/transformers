import json
from math import ceil

from modularqa.con_gen.constants import SQUAD_MODEL
from modularqa.inference.model_search import ParticipantModel
from modularqa.utils.generation import LMGenerator
from modularqa.utils.seq_utils import get_sequence_representation


class LMGenParticipant(LMGenerator, ParticipantModel):

    def __init__(self, scale_by_step=1, **kwargs):
        self.scale_by_step = scale_by_step
        super(LMGenParticipant, self).__init__(**kwargs)

    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param data: should have a dictionary as input containing
          mutable data
        :type data: dict
        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        :raises: ValueError
        """
        ## first checks state of `json_input` to figure out how to format things
        ## the first question
        data = state._data
        question_seq = data["question_seq"]
        answer_seq = data["answer_seq"]
        model_seq = data["model_seq"]
        gen_seq = get_sequence_representation(origq=data["query"], question_seq=question_seq,
                                              answer_seq=answer_seq, model_seq=model_seq,
                                              for_generation=True)
        if debug: print("<GPTGen>: %s" % gen_seq)

        ## eventual output
        new_states = []
        num_samples = ceil(self.num_samples * pow((1 / self.scale_by_step), len(answer_seq)))
        if self.top_samples:
            top_samples = ceil(self.top_samples * pow((1 / self.scale_by_step), len(answer_seq)))
        ## go through generated questions
        output_seqs, output_scores = self.generate_sequences(gen_seq,
                                                             num_samples=num_samples,
                                                             top_samples=top_samples)
        for output in list(set(output_seqs)):
            output = output.strip()
            # copy state
            new_state = state.copy()
            ## add new question to question_seq
            new_state._data["question_seq"].append(output)
            ## specify that in this case, the qa model should come next in this case
            new_state._next = "qal"
            ## maniuplate score (e.g., add)
            # new_state._score += score
            new_state._data["command_seq"].append("gen")
            ## mark the last output
            new_state.last_output = output

            new_states.append(new_state)
        ##
        return new_states


class DecompRCGenParticipant(ParticipantModel):
    """
    Model that produces the decompositions as specified in the input file
    """

    def __init__(self, decomp_file):
        self.decompositions = self.load_decomp_file(decomp_file)

    def load_decomp_file(self, file):
        qid_to_reasoning_map = {}
        with open(file, "r") as input_fp:
            input_json = json.load(input_fp)
            for key, value in input_json.items():
                if key in qid_to_reasoning_map:
                    print("Multiple reasonings for qid: {}".format(key))
                else:
                    qid_to_reasoning_map[key] = []

                qid_to_reasoning_map[key].append({
                    "reasoning": value["reasoning"],
                    "queries": value["queries"]
                })
        return qid_to_reasoning_map

    def query(self, state, debug=False):
        ## first checks state of `json_input` to figure out how to format things
        ## the first question
        data = state._data
        question_seq = data["question_seq"]
        answer_seq = data["answer_seq"]
        model_seq = data["model_seq"]
        qid = data["qid"]
        if debug: print("<DecompRCGen>: %s" % qid)

        ## eventual output
        new_states = []
        if qid not in self.decompositions:
            print("No decompositions for {}".format(qid))
            return []
        ## go through generated questions
        for reasoning in list(self.decompositions[qid]):
            if reasoning["queries"] is None:
                # one hop question
                if len(question_seq) == 0:
                    curr_query = "(" + SQUAD_MODEL + ") " + data["question"]
                else:
                    curr_query = "[EOQ]"
            elif len(question_seq) == len(reasoning["queries"]) - 1:
                # end of reasoning
                curr_query = "[EOQ]"
            else:
                curr_query = reasoning["queries"][len(question_seq)+1]
                if curr_query.isupper():
                    # Comparison. Can't handle right now
                    print("Failed on question: {}".format(curr_query))
                    return []

                if "[answer]" in curr_query:
                    curr_query = curr_query.replace("[answer]", answer_seq[-1])
                curr_query = "(" + SQUAD_MODEL + ") " + curr_query
            # copy state
            new_state = state.copy()
            ## add new question to question_seq
            new_state._data["question_seq"].append(curr_query)
            ## specify that in this case, the qa model should come next in this case
            new_state._next = "qa"
            new_state._data["command_seq"].append("gen")
            ## mark the last output
            new_state.last_output = json.dumps(reasoning)

            new_states.append(new_state)
        ##
        return new_states
