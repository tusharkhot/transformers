from typing import List, Tuple, Dict, Any, Optional

from modularqa.con_gen.constants import LIST_JOINER
from modularqa.con_gen.constraints import QAConstraint
from modularqa.drop.drop_utils import get_number, number_match
from modularqa.utils.classifier import LMClassifier
from modularqa.utils.math_qa import MathQA
from modularqa.utils.qa import LMQuestionAnswerer
from modularqa.utils.str_utils import tokenize_str, overlap_score


class QuestionVerifier:
    """
    Class that verifies the question quality
    """
    path_to_modeltokenizer = {}

    def verify_questions(self,
                         qid: str,
                         qaconstraint: QAConstraint,
                         questions: List[str],
                         previous_questions: List[str] = None,
                         previous_answers: List[str] = None) -> Tuple[
        List[str], List[str], Dict[Any, Any]]:
        """
        Return list of verified questions + metadata
        """
        raise NotImplementedError("generate_questions not implemented for {}".format(
            self.__class__.__name__))

    @classmethod
    def load_from_json(cls, input_json):
        if input_json["type"] == "lm":
            input_json.pop("type")
            return LMQuestionVerifier.load_from_json(input_json)
        if input_json["type"] == "lm_bool":
            input_json.pop("type")
            return BoolQuestionVerifier.load_from_json(input_json)
        if input_json["type"] == "lm_spans":
            input_json.pop("type")
            return LMSpansQuestionVerifier.load_from_json(input_json)
        if input_json["type"] == "math_ver":
            input_json.pop("type")
            return MathQuestionVerifier.load_from_json(input_json)
        if input_json["type"] == "dummy":
            input_json.pop("type")
            return DummyVerifier.load_from_json(input_json)
        raise ValueError("Unknown verifier type: " + input_json["type"])

    def reset_question_caches(self):
        pass


class DummyVerifier(QuestionVerifier):

    def __init__(self):
        super(DummyVerifier, self).__init__()

    def verify_questions(self, qid: str, qaconstraint: QAConstraint,
                         questions: List[str],
                         previous_questions: List[str] = None,
                         previous_answers: List[str] = None) -> Tuple[
        List[str], List[str], Dict[Any, Any]]:
        """
        Return list of verified questions + metadata
        """
        raise NotImplementedError("generate_questions not implemented for {}".format(
            self.__class__.__name__))

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)

    def reset_question_caches(self):
        pass


class LMQuestionVerifier(QuestionVerifier):

    def __init__(self, use_all_paras=False, **kwargs):
        self.question_answers = {}
        self.use_all_paras = use_all_paras
        self.qa_model = LMQuestionAnswerer(**kwargs)

    def reset_question_caches(self):
        self.question_answers.clear()

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)

    def answer_score(self, predicted_answer: str, expected_answer: str):
        pred_tokens = tokenize_str(predicted_answer)
        exp_tokens = tokenize_str(expected_answer)
        return overlap_score(pred_tokens, exp_tokens)

    def number_score(self, predicted_answer: str, expected_answer: str):
        pred_num = get_number(predicted_answer)
        exp_num = get_number(expected_answer)
        if exp_num is None:
            raise ValueError("Expected value should be a number: {}".format(expected_answer))
        return 1.0 if number_match(exp_num, pred_num) else 0.0

    def date_score(self, predicted_answer: str, expected_answer: str):
        raise NotImplementedError("Date matching not implemented!")

    def verify_questions(self, qid: str, qaconstraint: QAConstraint,
                         questions: List[str],
                         previous_questions: List[str] = None,
                         previous_answers: List[str] = None) -> Tuple[
        List[str], List[str], Dict[Any, Any]]:
        selected_questions = []
        selected_answers = []
        metadata = {"scored_questions": []}
        for question in questions:
            key = question + "$$" + qaconstraint.context
            if key in self.question_answers:
                answers = self.question_answers[key]
            else:
                if self.use_all_paras:
                    answers = self.qa_model.answer_question_only(question=question, qid=qid)
                else:
                    answers = self.qa_model.answer_question(question=question,
                                                            paragraphs=[qaconstraint.context])
                self.question_answers[key] = answers
            if len(answers) == 0:
                raise ValueError("Atleast empty string should be returned!")
            for answer in answers:
                pred_answer = answer.answer
                exp_answer = qaconstraint.aconstraint.exp_ans
                exp_ans_type = qaconstraint.aconstraint.exp_ans_type
                if "||" in pred_answer:
                    pred_answer = pred_answer[pred_answer.find("||") + 2:]
                if exp_answer is not None:
                    if exp_ans_type == "span" or exp_ans_type is None:
                        ascore = self.answer_score(pred_answer, exp_answer)
                    elif exp_ans_type == "number":
                        ascore = self.number_score(pred_answer, exp_answer)
                    elif exp_ans_type == "date":
                        ascore = self.date_score(pred_answer, exp_answer)
                    else:
                        raise ValueError("Cannot handle answer type:" +
                                         str(qaconstraint.aconstraint.exp_ans_type))
                elif pred_answer == "":
                    ascore = 0.0
                elif exp_ans_type is not None and exp_ans_type == "number":
                    number_val = get_number(pred_answer)
                    if number_val is None:
                        ascore = 0.0
                    else:
                        # todo handle dates
                        ascore = 1.0
                else:
                    ascore = 1.0
                metadata["scored_questions"].append({
                    "q": question,
                    "p": pred_answer,
                    "a": exp_answer,
                    "s": ascore})
                if ascore > 0 and question not in selected_questions:
                    selected_questions.append(question)
                    selected_answers.append(pred_answer)
        return selected_questions, selected_answers, metadata


class LMSpansQuestionVerifier(QuestionVerifier):

    def __init__(self, **kwargs):
        self.question_answers = {}
        # set num_ans_para to 10 if not set
        if "num_ans_para" not in kwargs:
            kwargs["num_ans_para"] = 10
        self.qa_model = LMQuestionAnswerer(**kwargs)

    def reset_question_caches(self):
        self.question_answers.clear()

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)

    def answer_score(self, predicted_answer: List[str], expected_answer: List[str]):
        if len(expected_answer) == 0:
            if len(predicted_answer) == 0:
                return 1.0
            else:
                return 0.0
        total_score = 1.0
        for exp_ans in expected_answer:
            exp_tokens = tokenize_str(exp_ans)
            pred_scores = []
            for pred_ans in predicted_answer:
                pred_tokens = tokenize_str(pred_ans)
                pred_scores.append(overlap_score(pred_tokens, exp_tokens))
            total_score *= max(pred_scores) if len(pred_scores) else 0.0
        return total_score

    def verify_questions(self, qid: str, qaconstraint: QAConstraint,
                         questions: List[str],
                         previous_questions: List[str] = None,
                         previous_answers: List[str] = None) -> Tuple[List[str], List[str],
                                                                      Dict[Any, Any]]:
        selected_questions = []
        selected_answers = []
        metadata = {"scored_questions": []}
        for question in questions:
            key = question + "$$" + qaconstraint.context
            if key in self.question_answers:
                answers = self.question_answers[key]
            else:
                answers = self.qa_model.answer_question(question=question,
                                                        paragraphs=[qaconstraint.context])
                self.question_answers[key] = answers
            answer_texts = [a.answer for a in answers]
            exp_answer = qaconstraint.aconstraint.exp_ans
            if exp_answer is not None:
                answer_score = self.answer_score(answer_texts, exp_answer.split(LIST_JOINER))
            else:
                answer_score = None
            exp_answer_len = qaconstraint.aconstraint.exp_ans_len
            alen_penalty: Optional[int] = None
            if exp_answer_len is not None:
                alen_penalty = abs(exp_answer_len - len(answers))
            else:
                alen_penalty = None

            metadata["scored_questions"].append({
                "q": question,
                "p": answer_texts,
                "a": exp_answer,
                "as": answer_score,
                "alp": alen_penalty
            })
            if question not in selected_questions:
                if answer_score is None or answer_score > 0:
                    if alen_penalty is None or alen_penalty < 2:
                        selected_questions.append(question)
                        selected_answers.append(LIST_JOINER.join(answer_texts))
        return selected_questions, selected_answers, metadata

class BoolQuestionVerifier(QuestionVerifier):

    def __init__(self, **kwargs):
        self.question_answers = {}
        self.classifier = LMClassifier(**kwargs)

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)

    def verify_questions(self, qid: str, qaconstraint: QAConstraint,
                         questions: List[str],
                         previous_questions: List[str] = None,
                         previous_answers: List[str] = None) -> Tuple[List[str], List[str],
                                                                      Dict[Any, Any]]:
        selected_questions = []
        selected_answers = []
        metadata = {"scored_questions": []}
        for question in questions:
            key = question + "$$" + qaconstraint.context
            if key in self.question_answers:
                scores = self.question_answers[key]
            else:
                scores = self.classifier.score_sequence(sequence1=question,
                                                        sequence2=qaconstraint.context)
                self.question_answers[key] = scores

            if scores[0] > scores[1]:
                answer = "no"
            else:
                answer = "yes"


            exp_answer = qaconstraint.aconstraint.exp_ans
            if exp_answer is not None:
                if exp_answer == answer:
                    answer_score = 1
                else:
                    answer_score = 0
            else:
                answer_score = None

            metadata["scored_questions"].append({
                "q": question,
                "p": answer,
                "a": exp_answer,
                "as": answer_score,
                "score": scores
            })
            if question not in selected_questions:
                if answer_score is None or answer_score > 0:
                    selected_questions.append(question)
                    selected_answers.append(answer)
        return selected_questions, selected_answers, metadata

class MathQuestionVerifier(QuestionVerifier):

    def __init__(self):
        self.math_qa = MathQA()

    def verify_questions(self, qid: str, qaconstraint: QAConstraint,
                         questions: List[str],
                         previous_questions: List[str] = None,
                         previous_answers: List[str] = None) -> Tuple[List[str], List[str],
                                                                      Dict[Any, Any]]:
        selected_questions: List[str] = []
        selected_answers: List[str] = []
        metadata = {"scored_questions": []}
        for question in questions:
            pred_answer = self.math_qa.answer_question(question)
            if qaconstraint.aconstraint.exp_ans_type == "number":
                exp_ans_val = get_number(qaconstraint.aconstraint.exp_ans) \
                    if qaconstraint.aconstraint.exp_ans is not None else None
                pred_ans_val = get_number(pred_answer)
                ascore = self.number_score(pred_ans_val, exp_ans_val)
            else:
                exp_ans_val = qaconstraint.aconstraint.exp_ans
                ascore = self.answer_score(pred_answer, exp_ans_val)

            if ascore > 0.0:
                selected_questions.append(question)
                selected_answers.append(pred_answer)
            metadata["scored_questions"].append({
                "q": question,
                "p": pred_answer,
                "a": exp_ans_val,
                "s": ascore
            })
        return selected_questions, selected_answers, metadata

    def answer_score(self, predicted_answer: Optional[str], expected_answer: Optional[str]):
        if expected_answer is None or expected_answer == "":
            return 1.0
        if predicted_answer is None or predicted_answer == "":
            return 0.0
        pred_tokens = tokenize_str(predicted_answer)
        exp_tokens = tokenize_str(expected_answer)
        return overlap_score(pred_tokens, exp_tokens)

    def number_score(self, predicted_answer: float, expected_answer: float):
        if expected_answer is None:
            return 1.0
        if predicted_answer is None:
            return 0.0
        return 1.0 if number_match(expected_answer, predicted_answer) else 0.0

    @classmethod
    def load_from_json(cls, input_json):
        return cls(**input_json)