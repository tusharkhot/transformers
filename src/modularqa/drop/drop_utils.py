import re
import string
from typing import List, Optional

from word2number import w2n

from modularqa.con_gen.constants import LIST_JOINER
from modularqa.utils.str_utils import tokenize_str


def get_bool(answer: str) -> bool:
    answer = answer.strip()
    if answer == "yes" or answer == "1" or answer.lower() == "true":
        return True
    return False


def get_number(answer: str) -> Optional[float]:
    if isinstance(answer, int) or isinstance(answer, float):
        return answer
    answer = answer.strip()
    m = re.match("^([-+]?([0-9]+[0-9,]*(\.\d*)?|\.\d+)([eE][-+]?\d+)?)[^0-9]*$", answer)
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


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


def parse_number(word: str):
    tokens = tokenize_str(word)
    for tokidx, token in enumerate(tokens):
        nums = convert_word_to_number(token, try_to_include_more_numbers=True,
                                      normalized_tokens=tokens, token_index=tokidx)
        if nums is not None:
            if isinstance(nums, list):
                return nums[0]
            else:
                return nums
    return None


def convert_word_to_number(word: str, try_to_include_more_numbers=False, normalized_tokens=None,
                           token_index=None):
    """
    Currently we only support limited types of conversion.
    """
    if try_to_include_more_numbers:
        # strip all punctuations from the sides of the word, except for the negative sign
        punctuations = string.punctuation.replace('-', '')
        word = word.strip(punctuations)
        # some words may contain the comma as deliminator
        if word.find(",") < len(word) - 3:
            word = word.replace(",", "")
        elif word.count(",") <= 1:
            word = word.replace(",", ".")
        elif word.find(",") >= 0:
            raise ValueError("Cannot parse number: " + word)
        # word2num will convert hundred, thousand ... to number, but we skip it.
        if word in ["hundred", "thousand", "million", "billion", "trillion"]:
            return None
        try:
            number = w2n.word_to_num(word)
        except ValueError:
            try:
                number = int(word)
            except ValueError:
                try:
                    number = float(word)
                except ValueError:
                    number = get_number(word)
        if number is None:
            if "-" in word or "–" in word:
                fields = word.split("-")
                if len(fields) == 1:
                    fields = word.split("–")
                nums = []
                for f in fields:
                    num = convert_word_to_number(f, try_to_include_more_numbers=False)
                    if num is not None:
                        nums.append(num)
                return nums

        if number is not None and normalized_tokens is not None and token_index is not None:
            if token_index < len(normalized_tokens) - 1:
                next_token = normalized_tokens[token_index + 1]
                if next_token in ["hundred", "thousand", "million", "billion", "trillion"]:
                    number = extend_number_magnitude(number, next_token)
        return number
    else:
        no_comma_word = word.replace(",", "")
        if no_comma_word in WORD_NUMBER_MAP:
            number = WORD_NUMBER_MAP[no_comma_word]
        else:
            try:
                number = int(no_comma_word)
            except ValueError:
                number = None
        return number


def extend_number_magnitude(number, next_token):
    if next_token == "hundred":
        number *= 100
    elif next_token == "thousand":
        number *= 1000
    elif next_token == "million":
        number *= 1000000
    elif next_token == "billion":
        number *= 1000000000
    elif next_token == "thousand":
        number *= 1000000000000
    return number
