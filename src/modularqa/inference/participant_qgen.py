from math import ceil

from modularqa.inference.model_search import ParticipantModel

from modularqa.con_gen.generators import LMQuestionGenerator
from modularqa.utils.generation import LMGenerator
from modularqa.utils.seq_utils import get_sequence_representation


class LMGenParticipant(LMGenerator, ParticipantModel):

    def __init__(self, scale_by_step=1, **kwargs):
        self.scale_by_step = scale_by_step
        super(LMGenerator, self).__init__(**kwargs)

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
        num_samples = ceil(self.num_samples * pow((1/self.scale_by_step), len(answer_seq)))
        ## go through generated questions
        for output in list(set(self.generate_sequences(gen_seq, num_samples=num_samples))):
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
