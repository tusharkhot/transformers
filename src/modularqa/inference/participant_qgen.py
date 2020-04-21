from modularqa.inference.model_search import ParticipantModel

from modularqa.con_gen.generators import LMQuestionGenerator
from modularqa.utils.generation import LMGenerator


class LMGenParticipant(LMGenerator, ParticipantModel):

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
        init_question = "QC: %s" % data["query"]
        question_seq = data["question_seq"]
        answer_seq = data["answer_seq"]
        model_seq = data["model_seq"]
        ## now intermediate ones
        if len(question_seq) != len(answer_seq):
            raise ValueError("Number of generted questions and answers should match before"
                             "question generation. Qs: {} As: {}".format(question_seq, answer_seq))
        if len(model_seq):
            intermediate = ["QI: (%s) %s A: %s" % (m, q, a) for m, q, a in zip(model_seq, question_seq, answer_seq)]
        else:
            intermediate = ["QI: %s A: %s" % (q, a) for q, a in zip(question_seq, answer_seq)]
        question = "%s %s QS:" % (init_question, ' '.join(intermediate))
        if debug: print("<GPTGen>: %s" % question)

        ## eventual output
        new_states = []

        ## go through generated questions
        for output in self.generate_sequences(question):
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
