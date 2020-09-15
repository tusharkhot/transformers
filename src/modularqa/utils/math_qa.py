import re
from typing import List

from dateutil.relativedelta import relativedelta

from modularqa.drop.drop_utils import get_subspans, get_number, get_bool, parse_number
from dateutil.parser import parse
from datetime import datetime

from modularqa.utils.str_utils import tokenize_str


class MathQA:

    def __init__(self):
        self.valid_operation_func = {
            "count": self.answer_count_q,
            "diff": self.answer_diff_q,
            "if_then": self.answer_ifthen_q,
            "if_then_bool": self.answer_ifthen_bool_q,
            "if_then_str": self.answer_ifthen_str_q,
            "not": self.answer_not_q,
            "and": self.answer_and_q,
            "intersect": self.answer_intersect_q
        }

    @staticmethod
    def date_difference(date1: str, date2: str, units: str="years"):
        default_date = datetime(3000, 1, 1)
        try:
            date1_datetime = parse(date1, default=default_date)
            date2_datetime = parse(date2, default=default_date)
        except Exception:
            # couldn't parse date
            return None
        # if one doesn't have month set, not usable
        if date1_datetime.year == default_date.year and date1_datetime.month == default_date.month:
            return None
        if date2_datetime.year == default_date.year and date2_datetime.month == default_date.month:
            return None

        if date1_datetime.year == default_date.year and date2_datetime.year != default_date.year:
            # one date is relative and other is not
            date1_datetime = date1_datetime.replace(year=date2_datetime.year)
        elif date2_datetime.year == default_date.year and date1_datetime.year != default_date.year:
            # one date is relative and other is not
            date2_datetime = date2_datetime.replace(year=date1_datetime.year)

        if units == "days":
            return (date1_datetime-date2_datetime).days
        if units == "months":
            return (date1_datetime.year - date2_datetime.year)*12 + (date1_datetime.month - date2_datetime.month)
        if units == "years":
            # human annotations are often on just the year value
            return date1_datetime.year - date2_datetime.year
        print("Unknown unit:" + units)
        return None

    def answer_question(self, question: str, paragraphs: List[str],
                        num_ans: int = None, return_all_ans: bool = None) -> str:
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

    def answer_intersect_q(self, question: str) -> str:
        m = re.match("intersect\((.*), (.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        span1 = m.group(1).strip()
        span2 = m.group(2).strip()
        span1_toks = tokenize_str(span1.lower())
        span2_toks = tokenize_str(span2.lower())
        shared_toks = set(span1_toks).intersection(set(span2_toks))
        output_str = " ".join([tok for tok in tokenize_str(span1) if tok.lower() in shared_toks])
        return output_str


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
            # try a simple parse
            num1 = get_number(m.group(1))
            num2 = get_number(m.group(2))
            # if it doesn't work, check dates
            if num1 is None or num2 is None:
                date_diff = MathQA.date_difference(m.group(1), m.group(2))
                if date_diff is not None:
                    pred_val = abs(date_diff)
                else:
                    # try a more permissive parse of numbers
                    num1 = parse_number(m.group(1))
                    num2 = parse_number(m.group(2))
                    if num1 is not None and num2 is not None:
                        # never asks for negative difference
                        pred_val = round(abs(num1 - num2), 2)
                    else:
                        print("Can not parse question: {}".format(question))
                        return ""
            else:
                pred_val = round(abs(num1 - num2), 2)

        return str(pred_val)

    def answer_ifthen_q(self, question: str) -> str:
        m = re.match("if_then\((.*)([<>])(.*),(.*),(.*)\)", question)
        if m is None:
            print("Can not parse question: {}".format(question))
            return ""
        num1 = parse_number(m.group(1))
        op = m.group(2)
        num2 = parse_number(m.group(3))
        # try date with the smallest unit
        date_diff = MathQA.date_difference(m.group(1), m.group(3), "days")
        if date_diff is not None:
            diff_val = date_diff
        elif num1 is not None and num2 is not None:
            diff_val = num1 - num2
        else:
            print("Can not parse question: {}".format(question))
            return ""
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
        pred_val = round(100 - num, 2)
        return str(pred_val)


if __name__ == '__main__':
    math_qa = MathQA()
    question = "count(23-yd + 14 yd)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(25.0, 17.0)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(25.0 million, 17 thousand)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(7 May 1487, 18 August 1487)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(Jun 2 2011, 3rd Aug, days)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(7 May 1487, 18 August 1487, days)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(29 August 1942, 1 April 1943, months)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(August 1922, 30 March 1922, months)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))


    question = "if_then(23 > 15, Obama, Biden)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "if_then(July 1918 > 1918, Obama, Biden)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "not(.4)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "diff(110, 40)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))


    question = "diff(20 March 1525, 16 February 1525, days)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question="intersect(Anabolic steroids, anabolic-androgenic steroids)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question="intersect(\"Steve Jobs\"., Steve Jobs.)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))

    question = "if_then(February 2008 < 6 March 2008," \
               " the election of Demetris Christofias as President," \
               " Garoyian becoming Speaker of the House of Representatives)"
    answer = math_qa.answer_question(question, [])
    print("Q: {} \n A: {}".format(question, answer))