import time
from typing import Tuple, List, Dict, Any

from modularqa.con_gen.constraints import QAConstraint
from modularqa.con_gen.generators import QuestionGenerator
from modularqa.con_gen.verifiers import QuestionVerifier


class QuestionGeneratorVerifier:
    """
    Class that generates + verifies questions
    """

    def __init__(self):
        self.time_per_model = {}
        self.count_per_model = {}

    def generate_questions(self, constraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], List[str],
                                                                        Dict[Any, Any]]:
        raise NotImplementedError("generate_questions not implemented for {}".format(
            self.__class__.__name__))

    @classmethod
    def load_from_json(cls, input_json):
        if input_json["type"] == "basic":
            input_json.pop("type")
            return BasicGeneratorVerifier.load_from_json(input_json)
        raise ValueError("Unknown generator+verifier type: " + input_json["type"])


class BasicGeneratorVerifier(QuestionGeneratorVerifier):

    def __init__(self, qgen_config, qa_config):
        self.qgen = QuestionGenerator.load_from_json(qgen_config)
        self.qa = QuestionVerifier.load_from_json(qa_config)
        super(BasicGeneratorVerifier, self).__init__()

    @classmethod
    def load_from_json(cls, input_json):
        return cls(input_json["qgen"], input_json["qa"])

    def reset_question_caches(self):
        self.qgen.reset_question_caches()
        self.qa.reset_question_caches()

    def generate_questions(self, constraint: QAConstraint,
                           previous_questions: List[str] = None,
                           previous_answers: List[str] = None) -> Tuple[List[str], List[str],
                                                                        Dict[Any, Any]]:
        start = time.time()
        questions, qgen_meta = self.qgen.generate_questions(qaconstraint=constraint,
                                                            previous_questions=previous_questions,
                                                            previous_answers=previous_answers)
        end = time.time()
        qgen_model = self.qgen.__class__.__name__
        if qgen_model not in self.time_per_model:
            self.time_per_model[qgen_model] = 0
            self.count_per_model[qgen_model] = 0
        self.time_per_model[qgen_model] += (end - start)
        self.count_per_model[qgen_model] += 1

        start = time.time()
        sel_question, pred_answers, qver_meta = self.qa.verify_questions(
            qaconstraint=constraint,
            questions=questions,
            previous_questions=previous_questions,
            previous_answers=previous_answers
        )
        end = time.time()
        qa_model = self.qa.__class__.__name__
        if qa_model not in self.time_per_model:
            self.time_per_model[qa_model] = 0
            self.count_per_model[qa_model] = 0
        self.time_per_model[qa_model] += (end - start)
        self.count_per_model[qa_model] += 1
        metadata = {
            "qgen": qgen_meta,
            "qver": qver_meta,
            "prevqs": previous_questions,
            "prevas": previous_answers
        }
        return sel_question, pred_answers, metadata
