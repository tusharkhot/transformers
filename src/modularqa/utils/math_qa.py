import re

from dateutil.relativedelta import relativedelta

from modularqa.drop.drop_utils import get_subspans, get_number, get_bool, parse_number
from dateutil.parser import parse
from datetime import datetime

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

    @staticmethod
    def date_difference(date1: str, date2: str, units: str="years"):
        try:
            date1_datetime = parse(date1)
            date2_datetime = parse(date2)
        except Exception:
            # couldn't parse date
            return None
        curr_date = datetime.now()

        # if one doesn't have month set, not usable
        if date1_datetime.year == curr_date.year and date1_datetime.month == curr_date.month:
            return None
        if date2_datetime.year == curr_date.year and date2_datetime.month == curr_date.month:
            return None

        if date1_datetime.year == curr_date.year and date2_datetime.year != curr_date.year:
            # one date is relative and other is not
            date1_datetime = date1_datetime.replace(year=date2_datetime.year)
        elif date2_datetime.year == curr_date.year and date1_datetime.year != curr_date.year:
            # one date is relative and other is not
            date2_datetime = date2_datetime.replace(year=date1_datetime.year)

        if units == "days":
            return (date1_datetime-date2_datetime).days
        if units == "months":
            return relativedelta(date1_datetime, date2_datetime).months
        if units == "years":
            # human annotations are often on just the year value
            return date1_datetime.year - date2_datetime.year
        print("Unknown unit:" + units)
        return None

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
        input_answer = m.group(1).strip()
        if input_answer == "":
            pred_length = 0
        else:
            pred_spans = get_subspans(input_answer)
            pred_length = len(pred_spans)
        return str(pred_length)

    def answer_and_q(self, question: str) -> str:
        m = re.match("and\((.*),(.*)\)", question)
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
        m = re.match("diff\(([^,]*),([^,]*)\)", question)
        if m is None:
            m = re.match("diff\(([^,]*),([^,]*),([^,]*)\)", question)
            if m is None:
                print("Can not parse question: {}".format(question))
                return ""
            else:
                date1 = m.group(1).strip()
                date2 = m.group(2).strip()
                units = m.group(3).strip()
                date_diff = MathQA.date_difference(date1, date2, units)
                if date_diff is not None:
                    pred_val = abs(date_diff)
                else:
                    print("Can not parse question: {}".format(question))
                    return ""
        else:
            num1 = parse_number(m.group(1))
            num2 = parse_number(m.group(2))
            if num1 is None or num2 is None:
                date_diff = MathQA.date_difference(m.group(1), m.group(2))
                if date_diff is None:
                    print("Can not parse question: {}".format(question))
                    return ""
                else:
                    pred_val = abs(date_diff)
            else:
                # never asks for negative difference
                pred_val = abs(num1 - num2)
        return str(pred_val)

    def answer_ifthen_q(self, question: str) -> str:
        m = re.match("if_then\((.*)([<>])(.*),(.*),(.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num1 = parse_number(m.group(1))
        op = m.group(2)
        num2 = parse_number(m.group(3))
        if num1 is None or num2 is None:
            # try date with the smallest unit
            date_diff = MathQA.date_difference(m.group(1), m.group(3), "days")
            if date_diff is None:
                print("Can not parse question: {}".format(question))
                return ""
            else:
                diff_val = date_diff
        else:
            diff_val = num1 - num2
        ent1 = m.group(4).strip()
        ent2 = m.group(5).strip()
        if (op == ">" and diff_val > 0) or (op == "<" and diff_val < 0):
            pred_ans = ent1
        else:
            pred_ans = ent2
        return pred_ans

    def answer_ifthen_bool_q(self, question: str) -> str:
        m = re.match("if_then_bool\((.*)->(.*),(.*),(.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num1 = get_bool(m.group(1))
        num2 = get_bool(m.group(2))
        if num1 is None or num2 is None or num1 == num2:
            print("Can not parse question: {}".format(question))
            return ""
        ent1 = m.group(3).strip()
        ent2 = m.group(4).strip()
        if num1:
            pred_ans = ent1
        else:
            pred_ans = ent2
        return pred_ans


    def answer_ifthen_str_q(self, question: str) -> str:
        m = re.match("if_then_str\((.*)!=(.*),(.*),(.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        str1 = m.group(1).strip()
        str2 = m.group(2).strip()
        if str1 == "" or str2 == "":
            print("Can not parse question: {}".format(question))
            return ""
        ent1 = m.group(3).strip()
        ent2 = m.group(4).strip()
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
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(25.0, 17.0)"
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(25.0 million, 17 thousand)"
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))


    question = "diff(Jun 2 2011, 3rd Aug, days)"
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(7 May 1487, 18 August 1487, days)"
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(29 August 1942, 1 April 1943, months)"
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))

    question = "if_then(23 > 15, Obama, Biden)"
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))

    question = "not(.4)"
    answer = math_qa.answer_question(question)
    print("Q: {} \n A: {}".format(question, answer))
