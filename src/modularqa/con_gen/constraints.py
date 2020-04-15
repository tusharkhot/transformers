from typing import List, Optional


class AnswerConstraint:
    """
    Constraints over the answers
    """

    def __init__(self, exp_ans: Optional[str] = None, exp_ans_len: Optional[int] = None,
                 exp_ans_type: Optional[str] = None):
        assert exp_ans is None or isinstance(exp_ans, str)
        self.exp_ans = exp_ans
        self.exp_ans_len = exp_ans_len
        self.exp_ans_type = exp_ans_type

    def to_json(self):
        return self.__dict__

    def to_str(self):
        return str(self.__dict__)


class QuestionConstraint:
    """
    Constraints over the questions
    """

    def __init__(self, hints: List[str]):
        self.hints = hints

    def to_json(self):
        return self.__dict__

    def to_str(self):
        return "hints: " + str(self.hints)


class QAConstraint:
    """
    Constraints over the question+answer
    """

    def __init__(self, context: str, model: str,
                 qconstraint: QuestionConstraint, aconstraint: AnswerConstraint):
        self.context = context
        self.model = model
        self.qconstraint = qconstraint
        self.aconstraint = aconstraint

    def to_json(self):
        return {
            "context": self.context,
            "model": self.model,
            "qconstraint": self.qconstraint.to_json(),
            "aconstraint": self.aconstraint.to_json()
        }

    def to_str(self):
        return "model: " + self.model + " qcons: " + self.qconstraint.to_str() + \
               " acons: " + self.aconstraint.to_str()

    @classmethod
    def from_json(cls, input_json):
        return cls(input_json["context"], input_json["model"],
                   QuestionConstraint(**input_json["qconstraint"]),
                   AnswerConstraint(**input_json["aconstraint"]))
