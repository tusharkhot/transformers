import itertools
import math
import random
from collections import Counter
from copy import deepcopy
from dateutil.parser import parse
from typing import Tuple, List, Dict, Any

from modularqa.con_gen.constants import HINT_MARKER, HINTS_DELIM, ANSWER_MARKER, QUESTION_MARKER
from modularqa.con_gen.constraints import QAConstraint
from modularqa.drop.drop_utils import get_number, get_bool
from modularqa.utils.generation import LMGenerator
from modularqa.utils.math_qa import MathQA


class QuestionGenerator:
    """
    Class that generates questions given the qaconstraints
    """

    path_to_modeltokenizer = {}

    def generate_questions(self, qaconstraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], Dict[Any, Any]]:
        raise NotImplementedError("generate_questions not implemented for {}".format(
            self.__class__.__name__))

    @classmethod
    def load_from_json(cls, input_json):
        if input_json["type"] == "lm":
            input_json.pop("type")
            return LMQuestionGenerator.load_from_json(input_json)
        if input_json["type"] == "math_gen":
            input_json.pop("type")
            return MathQuestionGenerator.load_from_json(input_json)
        if input_json["type"] == "dummy":
            input_json.pop("type")
            return DummyGenerator.load_from_json(input_json)
        raise ValueError("Unknown generator type: " + input_json["type"])

    def reset_question_caches(self):
        pass


class DummyGenerator(QuestionGenerator):

    def __init__(self):
        super(DummyGenerator, self).__init__()

    def generate_questions(self, qaconstraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], Dict[Any, Any]]:
        raise NotImplementedError("generate_questions not implemented for {}".format(
            self.__class__.__name__))

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)

    def reset_question_caches(self):
        pass


class LMQuestionGenerator(QuestionGenerator):

    def __init__(self, sample_hints_groups=1, format="c_h_a_q", **kwargs):
        self.sample_hints_groups = sample_hints_groups
        self.cached_questions = {}
        self.format = format
        self.qgen_model = LMGenerator(**kwargs)

    def reset_question_caches(self):
        self.cached_questions.clear()

    def generate_questions(self, qaconstraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], Dict[Any, Any]]:
        if self.format == "c_h_a_q":
            output_seqs = []
            # sub-sample hints based on previously generated questions
            for g in range(self.sample_hints_groups):
                # only sample after the first group
                if g > 0:
                    # count the number of occurences of hints. TODO Stemming?
                    hint_counter = Counter()
                    hints = qaconstraint.qconstraint.hints
                    for hintidx, hint in enumerate(hints):
                        hint_counter[hintidx] = 0
                        for seq in output_seqs:
                            if hint in seq:
                                hint_counter.update([hintidx])
                    # get exp distribution from the counts
                    distribution = [0] * len(hints)
                    for hidx, c in hint_counter.items():
                        distribution[hidx] = math.exp(-c)
                    # sample based on this distribution
                    new_hints = random.choices(hints, weights=distribution, k=len(hints))
                    new_hints = list(set(new_hints))
                else:
                    new_hints = list(set(qaconstraint.qconstraint.hints))

                sequence = qaconstraint.context + HINT_MARKER + \
                           HINTS_DELIM.join(new_hints)
                if qaconstraint.aconstraint.exp_ans is not None:
                    sequence += ANSWER_MARKER + qaconstraint.aconstraint.exp_ans
                sequence += QUESTION_MARKER
                if sequence in self.cached_questions and g == 0:
                    return self.cached_questions[sequence]
                num_samples = math.ceil(self.qgen_model.num_samples / self.sample_hints_groups)
                outputs = self.qgen_model.generate_sequences(sequence, num_samples)
                output_seqs.extend([o for o in outputs if len(o)])
                output_seqs = list(set(output_seqs))

            metadata = {
                "generated_questions": output_seqs
            }
            self.cached_questions[sequence] = output_seqs, metadata
            return output_seqs, metadata
        else:
            raise ValueError("Unknown format: {}!".format(self.format))

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)


class MathQuestionGenerator(QuestionGenerator):

    def __init__(self):
        self.valid_operations = ["count", "diff", "not"]

    def get_potential_dates(self, previous_answers):
        if previous_answers is None:
            return []
        potential_dates = []
        for answer in previous_answers:
            try:
                date_val = parse(answer)
                if date_val:
                    potential_dates.append(answer)
            except Exception:
                # couldn't parse
                pass
        return potential_dates

    def valid_number_pair(self, pair):
        (val1, val2) = pair
        if -1 <= len(str(val1)) - len(str(val2)) <= 1 and val1 != val2:
            return True
        return False

    def valid_date_pair(self, pair):
        (val1, val2) = pair
        if MathQA.date_difference(val1, val2) is not None and \
                MathQA.date_difference(val1, val2, "days") != 0:
            return True

        return False

    def get_potential_numbers(self, previous_answers):
        if previous_answers is None:
            return []
        potential_numbers = []
        for answer in previous_answers:
            num_val = get_number(answer)
            if num_val is not None:
                potential_numbers.append(answer)
        return potential_numbers

    def get_potential_bools(self, previous_answers):
        if previous_answers is None:
            return []
        potential_bools = []
        for answer in previous_answers:
            num_val = get_bool(answer)
            if num_val is not None:
                potential_bools.append(num_val)
        return potential_bools


    def make_predicate(self, predicate, args):
        output = predicate  + "("
        new_args = [str(a).replace(",", "") for a in args]
        output += ", ".join(new_args)
        output += ")"
        return output



    def generate_questions(self, qaconstraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], Dict[Any, Any]]:

        count_questions, count_metadata = self.generate_count_questions(qaconstraint,
                                                                        previous_questions,
                                                                        previous_answers)

        diff_questions, diff_metadata = self.generate_diff_questions(qaconstraint,
                                                                     previous_questions,
                                                                     previous_answers)

        not_questions, not_metadata = self.generate_not_questions(qaconstraint, previous_questions,
                                                                  previous_answers)

        ifthen_questions, ifthen_metadata = self.generate_ifthen_questions(qaconstraint,
                                                                           previous_questions,
                                                                           previous_answers)

        and_questions, and_metadata = self.generate_and_questions(qaconstraint, previous_questions,
                                                                  previous_answers)

        question_list = count_questions + diff_questions + not_questions + ifthen_questions + and_questions
        metadata = {
            "count": count_metadata,
            "diff": diff_metadata,
            "not": not_metadata,
            "ifthen": ifthen_metadata,
            "and": and_metadata
        }
        if len(question_list) == 0:
            msg = "Cannot handle constraint: {} in math questions".format(qaconstraint.to_str())
            metadata["final"] = msg
            return [], metadata
        else:
            return question_list, metadata

    def generate_count_questions(self, qaconstraint, previous_questions, previous_answers):
        if "count" in qaconstraint.qconstraint.hints or "number" in qaconstraint.qconstraint.hints:
            if qaconstraint.aconstraint.exp_ans_type != "number":
                metadata = {
                    "error": "Expected answer is not a number for a math operation!{}".format(
                        qaconstraint.to_json())
                }
                return [], metadata
            if previous_answers is None or len(previous_answers) == 0:
                metadata = {
                    "error": "Count operation has no previous answers!: {}".format(
                        qaconstraint.to_str())
                }
                return [], metadata
            if qaconstraint.qconstraint.use_answer_idxs:
                if len(qaconstraint.qconstraint.use_answer_idxs) != 1:
                    metadata = {
                        "error": "Count constraint should only have one answer to count over!"
                                 "{}".format(qaconstraint.qconstraint.to_str())
                    }
                    return [], metadata
                ans_idx = qaconstraint.qconstraint.use_answer_idxs[0]
                if len(previous_answers) <= ans_idx:
                    raise ValueError("Reference to unknown answer idx in constraint: {}".format(
                        qaconstraint.qconstraint.to_str()))
                ans = previous_answers[ans_idx]
            else:
                ans = previous_answers[-1]
            question = self.make_predicate("count", [ans])
            metadata = {}
            return [question], metadata
        return [], {}


    def get_comparable_pairs(self, use_answer_idxs, hints, previous_answers):
        if use_answer_idxs:
            use_answers = [previous_answers[aidx]
                           for aidx in use_answer_idxs]
        else:
            use_answers = previous_answers
        potential_numbers = self.get_potential_numbers(use_answers)
        if len(potential_numbers) < 2:
            potential_numbers.extend(self.get_potential_numbers(hints))
        potential_dates = self.get_potential_dates(use_answers)
        if len(potential_dates) < 2:
            potential_dates.extend(self.get_potential_dates(hints))

        number_pairs = itertools.combinations(potential_numbers, 2)
        date_pairs = itertools.combinations(potential_dates, 2)

        return ([p for p in number_pairs if self.valid_number_pair(p)],
               [p for p in date_pairs if self.valid_date_pair(p)])


    def generate_diff_questions(self, qaconstraint, previous_questions, previous_answers):
        if "difference" in qaconstraint.qconstraint.hints or "diff" in qaconstraint.qconstraint.hints:
            if qaconstraint.aconstraint.exp_ans_type != "number":
                metadata = {
                    "error": "Expected answer is not a number for a math operation!{}".format(
                        qaconstraint.to_json())
                }
                return [], metadata
            questions = []
            metadata = {}
            # default years
            units = "years"
            if "years" in qaconstraint.qconstraint.hints:
                units = "years"
            if "months" in qaconstraint.qconstraint.hints:
                units = "months"
            if "days" in qaconstraint.qconstraint.hints:
                units = "days"


            comparable_num_pairs, comparable_date_pairs = self.get_comparable_pairs(
                qaconstraint.qconstraint.use_answer_idxs,
                qaconstraint.qconstraint.hints,
                previous_answers
            )

            if len(comparable_num_pairs) + len(comparable_date_pairs) == 0:
                metadata = {
                    "error": "Not enough comparable dates/numbers to compute difference"
                }
                return [], metadata

            for pair in comparable_num_pairs:
                questions.append(self.make_predicate("diff", list(pair)))

            for pair in comparable_date_pairs:
                questions.append(self.make_predicate("diff", list(pair) + [units]))

            return questions, metadata
        return [], {}

    def generate_not_questions(self, qaconstraint, previous_questions, previous_answers):
        if "not" in qaconstraint.qconstraint.hints:
            if qaconstraint.aconstraint.exp_ans_type != "number":
                metadata = {
                    "error": "Expected answer is not a number for a math operation!{}".format(
                        qaconstraint.to_json())
                }
                return [], metadata
            questions = []
            metadata = {}
            if qaconstraint.qconstraint.use_answer_idxs:
                if len(qaconstraint.qconstraint.use_answer_idxs) != 1:
                    metadata["error"] = "Not constraint should only one answer to count over!" \
                                        "{}".format(qaconstraint.qconstraint.to_str())
                    return [], metadata
                potential_numbers = [get_number(previous_answers[aidx])
                                     for aidx in qaconstraint.qconstraint.use_answer_idxs]
            else:
                potential_numbers = self.get_potential_numbers(previous_answers)
            filtered_numbers = [n for n in potential_numbers if get_number(n) <= 100]
            if len(filtered_numbers) < 1:
                metadata = {
                    "error": "Not enough numbers to compute not: {}".format(filtered_numbers)
                }
                return [], metadata
            else:
                for number in filtered_numbers:
                    questions.append(self.make_predicate("not", [number])
            return questions, metadata
        return [], {}

    def generate_and_questions(self, qaconstraint, previous_questions, previous_answers):
        if "and" in qaconstraint.qconstraint.hints and len(qaconstraint.qconstraint.hints) == 1:
            questions = []
            metadata = {}
            if qaconstraint.qconstraint.use_answer_idxs:
                potential_bools = []
                for aidx in qaconstraint.qconstraint.use_answer_idxs:
                    num = get_bool(previous_answers[aidx])
                    if num is not None:
                        potential_bools.append(num)
                if len(potential_bools) < 2:
                    for hint in qaconstraint.qconstraint.hints:
                        num = get_bool(hint)
                        if num is not None:
                            potential_bools.append(num)
            else:
                potential_bools = self.get_potential_bools(previous_answers)

            if len(potential_bools) < 2:
                metadata = {
                    "error": "Not enough bools to compute and"
                }
                return [], metadata
            questions.append(self.make_predicate("and", [self.yes_no(p) for p in potential_bools[-2:]]))
            return questions, metadata
        return [], {}

    def yes_no(self, bool_value):
        if bool_value:
            return "yes"
        else:
            return "no"

    def generate_ifthen_questions(self, qaconstraint, previous_questions, previous_answers):
        # numeric comparison
        if "if_then" in qaconstraint.qconstraint.hints:
            entities = deepcopy(qaconstraint.qconstraint.hints)
            entities.remove("if_then")
            questions = []
            metadata = {}
            comparable_num_pairs, comparable_date_pairs = self.get_comparable_pairs(
                qaconstraint.qconstraint.use_answer_idxs,
                qaconstraint.qconstraint.hints,
                previous_answers
            )
            if len(comparable_num_pairs) + len(comparable_date_pairs) == 0:
                metadata = {
                    "error": "Not enough numbers/dates to compute if_then"
                }
                return [], metadata
            else:
                entities = [e.replace(",", " ") for e in entities]
                for pair in comparable_num_pairs + comparable_date_pairs:
                    questions.append(
                        self.make_predicate("if_then",
                                            [str(pair[0]) + " > " + str(pair[1])] + entities))
                    questions.append(
                        self.make_predicate("if_then",
                                            [str(pair[1]) + " > " + str(pair[0])] + entities))
            return questions, metadata

        # bool comparison
        if "if_then_bool" in qaconstraint.qconstraint.hints:
            entities = deepcopy(qaconstraint.qconstraint.hints)
            entities.remove("if_then_bool")
            questions = []
            metadata = {}
            if qaconstraint.qconstraint.use_answer_idxs:
                potential_bools = []
                for aidx in qaconstraint.qconstraint.use_answer_idxs:
                    num = get_bool(previous_answers[aidx])
                    if num is not None:
                        potential_bools.append(num)
                if len(potential_bools) < 2:
                    for hint in qaconstraint.qconstraint.hints:
                        num = get_bool(hint)
                        if num is not None:
                            potential_bools.append(num)
            else:
                potential_bools = self.get_potential_bools(previous_answers)
            if len(potential_bools) < 2:
                metadata = {
                    "error": "Not enough bools to compute if_then_bool"
                }
                return [], metadata
            if potential_bools[-2] == potential_bools[-1]:
                metadata = {
                    "error": "Same boolean value returned by both questions!"
                }
                return [], metadata
            else:
                entities = [e.replace(",", " ") for e in entities]
                questions.append(
                    self.make_predicate("if_then_bool",
                                        [self.yes_no(potential_bools[-2]) +
                                         " -> " + self.yes_no(potential_bools[-1])] + entities))
                questions.append(
                    self.make_predicate("if_then_bool",
                                        [self.yes_no(potential_bools[-2]) +
                                         " -> " + self.yes_no(potential_bools[-1])] +
                                        list(reversed(entities))))
            return questions, metadata

        if "if_then_str" in qaconstraint.qconstraint.hints:
            entities = deepcopy(qaconstraint.qconstraint.hints)
            entities.remove("if_then_str")
            questions = []
            metadata = {}
            if len(previous_answers) < 2:
                metadata = {
                    "error": "Not enough answers to compute if_then_str"
                }
                return [], metadata
            if qaconstraint.qconstraint.use_answer_idxs:
                potential_strs = [previous_answers[aidx]
                                  for aidx in qaconstraint.qconstraint.use_answer_idxs]
            else:
                potential_strs = previous_answers[-2:]
            potential_strs = [e.replace(",", " ") for e in potential_strs]
            questions.append(
                self.make_predicate("if_then_str",
                                    [str(potential_strs[-2]) + " != " + str(potential_strs[-1])] +
                                    ["no", "yes"]))
            return questions, metadata
        return [], {}

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)
