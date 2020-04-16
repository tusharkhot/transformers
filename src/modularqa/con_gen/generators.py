import itertools
import math
import random
from collections import Counter
from copy import deepcopy
from typing import Tuple, List, Dict, Any

import torch

from modularqa.con_gen.constraints import QAConstraint
from modularqa.drop.drop_utils import get_number
from modularqa.con_gen.constants import HINT_MARKER, HINTS_DELIM, ANSWER_MARKER, QUESTION_MARKER
from modularqa.utils.generation import generate_text_sequence
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead


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

    @staticmethod
    def load_model_tokenizer(model_path):
        if model_path in QuestionGenerator.path_to_modeltokenizer:
            return QuestionGenerator.path_to_modeltokenizer[model_path]
        else:
            config = AutoConfig.from_pretrained(
                model_path,
                cache_dir=None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                do_lower_case=False,
                cache_dir=None,
            )
            print("Loading {} model from: {}".format(config.model_type, model_path))
            model = AutoModelWithLMHead.from_pretrained(
                model_path,
                from_tf=False,
                config=config,
                cache_dir=None,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            QuestionGenerator.path_to_modeltokenizer[model_path] = (model, tokenizer)
            return model, tokenizer

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

    def __init__(self,
                 model_path,
                 model_type=None,
                 length=30,
                 num_samples=20,
                 top_p=0.9,
                 top_k=0,
                 temperature=1.0,
                 sample_hints_groups=1,
                 format="c_h_a_q"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## set up the model
        self.model, self.tokenizer = QuestionGenerator.load_model_tokenizer(model_path)
        self.model_type = model_type if model_type is not None else self.model.config.model_type
        self.model.eval()
        self.length = length
        self.num_samples = num_samples
        self.top_p = top_p
        self.top_k = top_k
        self.format = format
        self.temperature = temperature
        self.sample_hints_groups = sample_hints_groups
        self.cached_questions = {}

    def reset_question_caches(self):
        self.cached_questions.clear()

    def generate_questions(self, qaconstraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], Dict[Any, Any]]:
        if self.format == "c_h_a_q":
            output_seqs = []
            for g in range(self.sample_hints_groups):
                # only sample after the first group
                if g > 0:
                    hint_counter = Counter()
                    hints = qaconstraint.qconstraint.hints
                    for hintidx, hint in enumerate(hints):
                        hint_counter[hintidx] = 0
                        for seq in output_seqs:
                            if hint in seq:
                                hint_counter.update([hintidx])
                    distribution = [0] * len(hints)
                    for hidx, c in hint_counter.items():
                        distribution[hidx] = math.exp(-c)
                    # print(hints)
                    # print(distribution)
                    new_hints = random.choices(hints, weights=distribution, k=len(hints))
                    new_hints = list(set(new_hints))
                    # print("Sampled hints: {} from {}".format(new_hints, hints))
                else:
                    new_hints = list(set(qaconstraint.qconstraint.hints))
                sequence = qaconstraint.context + HINT_MARKER + \
                           HINTS_DELIM.join(new_hints)
                if qaconstraint.aconstraint.exp_ans is not None:
                    sequence += ANSWER_MARKER + qaconstraint.aconstraint.exp_ans
                sequence += QUESTION_MARKER
                if sequence in self.cached_questions and g == 0:
                    return self.cached_questions[sequence]

                num_samples = math.ceil(self.num_samples/self.sample_hints_groups)

                outputs = generate_text_sequence(model=self.model, prompt_text=sequence,
                                                     model_type=self.model_type,
                                                     length=self.length,
                                                     num_samples=num_samples,
                                                     temperature=self.temperature,
                                                     top_k=self.top_k, top_p=self.top_p,
                                                     tokenizer=self.tokenizer, device=self.device)
                # print("\n".join(outputs))
                output_seqs.extend(outputs)
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

    def get_potential_numbers(self, previous_answers):
        if previous_answers is None:
            return []
        potential_numbers = []
        for answer in previous_answers:
            num_val = get_number(answer)
            if num_val is not None:
                potential_numbers.append(num_val)
        return potential_numbers

    def generate_questions(self, qaconstraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], Dict[Any, Any]]:
        if len(qaconstraint.qconstraint.hints) == 1:
            operation = qaconstraint.qconstraint.hints[0]
            if operation not in self.valid_operations:
                raise ValueError("Cannot handle operation: {} in math questions".format(operation))
            if qaconstraint.aconstraint.exp_ans_type != "number":
                raise ValueError("Expected answer is not a number for a math operation!{}".format(
                    qaconstraint.to_json()
                ))
            if operation == "count":
                if previous_answers is None or len(previous_answers) == 0:
                    raise ValueError("Count operation has no previous answers!: {}".format(
                        qaconstraint))
                question = "count(" + previous_answers[-1] + ")"
                metadata = {}
                return [question], metadata
            if operation == "diff":
                questions = []
                metadata = {}
                potential_numbers = self.get_potential_numbers(previous_answers)
                if len(potential_numbers) < 2:
                    metadata = {
                        "error": "Not enough numbers to compute difference"
                    }
                else:
                    number_pairs = itertools.combinations(potential_numbers, 2)
                    for pair in number_pairs:
                        questions.append("diff(" + str(pair[0]) + ", " + str(pair[1]) + ")")
                        questions.append("diff(" + str(pair[1]) + ", " + str(pair[0]) + ")")
                return questions, metadata
            if operation == "not":
                questions = []
                metadata = {}
                potential_numbers = self.get_potential_numbers(previous_answers)
                filtered_numbers = [n for n in potential_numbers if n <= 100]
                if len(potential_numbers) < 1:
                    metadata = {
                        "error": "Not enough numbers to compute not: {}".format(potential_numbers)
                    }
                else:
                    for number in filtered_numbers:
                        questions.append("not(" + str(number) + ")")
                return questions, metadata
        elif len(qaconstraint.qconstraint.hints) == 3 and \
                "if_then" in qaconstraint.qconstraint.hints:
            entities = deepcopy(qaconstraint.qconstraint.hints)
            entities.remove("if_then")
            questions = []
            metadata = {}
            potential_numbers = self.get_potential_numbers(previous_answers)
            if len(potential_numbers) < 2:
                metadata = {
                    "error": "Not enough numbers to compute difference"
                }
            else:
                number_pairs = itertools.combinations(potential_numbers, 2)
                entities = [e.replace(",", " ") for e in entities]
                for pair in number_pairs:
                    questions.append("if_then(" + str(pair[0]) + " > " + str(pair[1]) + ", " +
                                     ", ".join(entities) + ")")
                    questions.append("if_then(" + str(pair[1]) + " > " + str(pair[0]) + ", " +
                                     ", ".join(entities) + ")")
            return questions, metadata
        else:
            raise ValueError("Cannot handle hints: {} in math questions".format(
                qaconstraint.qconstraint.hints))
        return [], {}

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)
