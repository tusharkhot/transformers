import json
from math import ceil
import re
from modularqa.con_gen.constants import SQUAD_MODEL, MATH_MODEL
from modularqa.drop.drop_utils import parse_number
from modularqa.inference.model_search import ParticipantModel
from modularqa.utils.generation import LMGenerator
from modularqa.utils.seq_utils import get_sequence_representation


class LMGenParticipant(LMGenerator, ParticipantModel):

    def __init__(self, scale_by_step=1, add_eos=False, add_prefix="", **kwargs):
        self.scale_by_step = scale_by_step
        self.add_eos = add_eos
        self.add_prefix = add_prefix
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
        if self.add_prefix:
            gen_seq = self.add_prefix + gen_seq
        if self.add_eos:
            gen_seq = gen_seq + "</s>"

        if debug: print("<GPTGen>: %s" % gen_seq)

        ## eventual output
        new_states = []
        num_samples = ceil(self.num_samples * pow((1 / self.scale_by_step), len(answer_seq)))
        if self.top_samples:
            top_samples = ceil(self.top_samples * pow((1 / self.scale_by_step), len(answer_seq)))
        else:
            top_samples = None
        ## go through generated questions
        output_seqs, output_scores = self.generate_sequences(gen_seq,
                                                             num_samples=num_samples,
                                                             top_samples=top_samples)
        for output in list(set(output_seqs)):
            output = output.strip()
            if self.model_type == "t5":
                # T5 does not produce "<"!
                m = re.match(".* ([^ <]+>) .*", output)
                if m:
                    output = output.replace(m.group(1), "<" + m.group(1))
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


class BreakLMGenParticipant(LMGenerator, ParticipantModel):

    def __init__(self, scale_by_step=1, **kwargs):
        self.scale_by_step = scale_by_step
        super(BreakLMGenParticipant, self).__init__(**kwargs)

    def format_breakq(self, breakq, answers):
        m = re.match("\(([a-zA-Z]+)\) (.*)", breakq)
        if m:
            operation = m.group(1)
            question = m.group(2)
            if operation == "select" or operation == "filter" or operation == "project":
                question = question.replace("return ", "")
                newq = "({}) {}".format(SQUAD_MODEL, question)
            elif operation == "boolean":
                if " same as " in question:
                    numbers = re.findall("#[0-9]+", question)
                    if len(numbers) == 2:
                        newq = "({}) if_then_str({}!={}, no, yes)".format(MATH_MODEL,
                                                                          numbers[0], numbers[1])
                    else:
                        print("Need exactly two answers for if_then_str question: " + question)
                        return None
                else:
                    print("Can not handle question: " + question)
                    return None
            elif operation == "arithmetic":
                if " difference " in question:
                    numbers = re.findall("#[0-9]+", question)
                    unit = None
                    if " year" in question:
                        unit = "years"
                    elif " day" in question:
                        unit = "days"
                    elif " month" in question:
                        unit = "months"
                    if len(numbers) == 1:
                        tempq = question.replace(numbers[0], "")
                        other_number = parse_number(tempq)
                        if other_number:
                            numbers.append(other_number)
                    if len(numbers) == 2:
                        if unit:
                            newq = "({}) diff({}, {}, {})".format(MATH_MODEL,
                                                                  numbers[0], numbers[1],
                                                                  unit)
                        else:
                            newq = "({}) diff({}, {})".format(MATH_MODEL, numbers[0], numbers[1])
                    else:
                        print("Need exactly two answer for diff q. Found: {} in {}".format(
                            numbers, question))
                        return None
                else:
                    print("Can not handle arithmetic operation in {}".format(breakq))
                    return None
            else:
                print("Can not handle operation: {} in {}".format(operation, breakq))
                return None
        else:
            print("Could not parse question: {}".format(breakq))
            return None
        for idx in range(len(answers)):
            newq = newq.replace("#" + str(idx+1), answers[idx])
        return newq


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

        # Representation for BREAK
        gen_seq = " QC: " + data["query"]
        if "break_seq" in data:
            for q in data["break_seq"]:
                gen_seq += " QI: " + q
        gen_seq += " QS: "
        if debug: print("<GPTGen>: %s" % gen_seq)

        ## eventual output
        new_states = []
        num_samples = ceil(self.num_samples * pow((1 / self.scale_by_step), len(answer_seq)))
        if self.top_samples:
            top_samples = ceil(self.top_samples * pow((1 / self.scale_by_step), len(answer_seq)))
        else:
            top_samples = None
        ## go through generated questions
        output_seqs, output_scores = self.generate_sequences(gen_seq,
                                                             num_samples=num_samples,
                                                             top_samples=top_samples)
        for output in list(set(output_seqs)):
            output = output.strip()
            # copy state
            new_state = state.copy()
            if "break_seq" not in new_state._data:
                new_state._data["break_seq"] = []
            new_state._data["break_seq"].append(output)
            if output == "[EOQ]":
                formatted_q = output
            else:
                formatted_q = self.format_breakq(output, answer_seq)
                if formatted_q is None:
                    continue
            ## add new question to question_seq
            new_state._data["question_seq"].append(formatted_q)
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

