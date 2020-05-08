import re

from modularqa.drop.drop_utils import get_subspans, get_number, get_bool


class MathQA:

    def __init__(self):
        self.valid_operation_func = {
            "count": self.answer_count_q,
            "diff": self.answer_diff_q,
            "if_then": self.answer_ifthen_q,
            "if_then_bool": self.answer_ifthen_bool_q,
            "if_then_str": self.answer_ifthen_str_q,
            "not": self.answer_not_q,
            "and": self.answer_and_q
        }

    def answer_question(self, question: str) -> str:
        for operation, func in self.valid_operation_func.items():
            m = re.match("{}\(.*\)".format(operation), question)
            if m:
                return func(question)
        print("Can not answer question: {}".format(question))
        return ""

    def answer_count_q(self, question: str) -> str:
        m = re.match("count\((.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        input_answer = m.group(1)
        if input_answer == "":
            pred_length = 0
        else:
            pred_spans = get_subspans(input_answer)
            pred_length = len(pred_spans)
        return str(pred_length)

    def answer_and_q(self, question: str) -> str:
        m = re.match("and\((.*), (.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num1 = get_bool(m.group(1))
        num2 = get_bool(m.group(2))
        if num1 is None or num2 is None:
            print("Can not parse question: {}".format(question))
            return ""
        if num1 and num2:
            return "yes"
        else:
            return "no"

    def answer_diff_q(self, question: str) -> str:
        m = re.match("diff\((.*), (.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num1 = get_number(m.group(1))
        num2 = get_number(m.group(2))
        if num1 is None or num2 is None:
            print("Can not parse question: {}".format(question))
            return ""
        pred_val = num1 - num2
        return str(pred_val)

    def answer_ifthen_q(self, question: str) -> str:
        m = re.match("if_then\((.*) ([<>]) (.*), (.*), (.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num1 = get_number(m.group(1))
        op = m.group(2)
        num2 = get_number(m.group(3))
        if num1 is None or num2 is None:
            print("Can not parse question: {}".format(question))
            return ""
        ent1 = m.group(4)
        ent2 = m.group(5)
        if (op == ">" and num1 > num2) or (op == "<" and num1 < num2):
            pred_ans = ent1
        else:
            pred_ans = ent2
        return pred_ans

    def answer_ifthen_bool_q(self, question: str) -> str:
        m = re.match("if_then_bool\((.*) -> (.*), (.*), (.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num1 = get_bool(m.group(1))
        num2 = get_bool(m.group(2))
        if num1 is None or num2 is None or num1 == num2:
            print("Can not parse question: {}".format(question))
            return ""
        ent1 = m.group(3)
        ent2 = m.group(4)
        if num1:
            pred_ans = ent1
        else:
            pred_ans = ent2
        return pred_ans


    def answer_ifthen_str_q(self, question: str) -> str:
        m = re.match("if_then_str\((.*) != (.*), (.*), (.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        str1 = m.group(1)
        str2 = m.group(2)
        if str1 == "" or str2 == "":
            print("Can not parse question: {}".format(question))
            return ""
        ent1 = m.group(3)
        ent2 = m.group(4)
        if str1 != str2:
            pred_ans = ent1
        else:
            pred_ans = ent2
        return pred_ans

    def answer_not_q(self, question: str) -> str:
        m = re.match("not\((.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num = get_number(m.group(1))
        if num is None:
            print("Can not parse question: {}".format(question))
            return ""
        pred_val = 100 - num
        return str(pred_val)


if __name__ == '__main__':
    math_qa = MathQA()
    question = "count(23-yd + 14 yd)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(25.0, 17.0)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "if_then(23 > 15, Obama, Biden)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "not(.4)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))
