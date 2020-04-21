import re
from typing import List, Optional

from modularqa.con_gen.constants import LIST_JOINER


def get_number(answer: str) -> Optional[float]:
    if isinstance(answer, int) or isinstance(answer, float):
        return answer
    answer = answer.strip()
    m = re.match("^([-+]?([0-9,]+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)[^0-9]*$", answer)
    if m:
        return float(m.group(1).replace(",", ""))
    else:
        # if re.match("^[0-9].*", answer):
        #     print("Not considered a number: " + answer)
        return None


def number_match(exp_num, pred_num) -> bool:
    if pred_num is None:
        return False
    return round(pred_num, 1) == round(exp_num, 1)


def get_subspans(answer: str) -> List[str]:
    output_list = []
    for sub_span in answer.split(LIST_JOINER):
        if "," in sub_span:
            output_list.extend(sub_span.split(","))
        elif " and " in sub_span:
            output_list.extend(sub_span.split(" and "))
        elif " & " in sub_span:
            output_list.extend(sub_span.split(" & "))
        else:
            output_list.append(sub_span)
    return output_list


def number_in_para(number, para):
    if "." in number:
        # floating point; consider an extra zero
        return re.match(".*\W" + re.escape(number) + "\W.*", para) is not None or \
               re.match(".*\W" + re.escape(number) + "0\W.*", para) is not None
    else:
        return re.match(".*\W" + re.escape(number) + "\W.*", para) is not None

def format_drop_answer(answer_json):
    if answer_json["number"]:
        return answer_json["number"], -1
    if len(answer_json["spans"]):
        return LIST_JOINER.join(answer_json["spans"]), -1
    # only date possible
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        print("Number, Span or Date not set in {}".format(answer_json))
        return None, -1
    return date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"], -1

