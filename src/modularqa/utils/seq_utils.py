from typing import List, Tuple

from modularqa.con_gen.constants import COMPQ_MARKER, INTERQ_MARKER, ANSWER_MARKER, SIMPQ_MARKER
from modularqa.utils.str_utils import tokenize_question


def get_sequence_representation(origq: str, question_seq: List[str], answer_seq: List[str],
                                model_seq: List[str] = None,
                                for_generation=True):
    ret_seq = COMPQ_MARKER + origq
    if for_generation and len(question_seq) != len(answer_seq):
        raise ValueError("Number of generated questions and answers should match before"
                         "question generation. Qs: {} As: {}".format(question_seq, answer_seq))
    elif not for_generation and len(question_seq) != len(answer_seq) + 1:
        raise ValueError("One extra question should be generated than answers"
                         " Qs: {} As: {}".format(question_seq, answer_seq))

    for aidx in range(len(answer_seq)):
        ret_seq += INTERQ_MARKER
        if model_seq is not None and len(model_seq):
            ret_seq += "(" + model_seq[aidx] + ")"
        ret_seq += question_seq[aidx]
        ret_seq += ANSWER_MARKER + answer_seq[aidx]

    if for_generation:
        ret_seq += SIMPQ_MARKER
    else:
        ret_seq += SIMPQ_MARKER + question_seq[-1]

    return ret_seq


def score_question_answer_chain(qchain: List[str], achain: List[str], complexq: str,
                                repeat_ok=True, score_answers = False):
    if len(qchain) > len(achain) + 1 or len(qchain) < len(achain):
        raise ValueError("Mismatch in question chain and answer chain lengths!: {}\n{}".format(
            qchain, achain
        ))

    compq_tokens = tokenize_question(complexq)
    compq_token_len = len(compq_tokens)
    qchain_tokens = []
    qchain_tokens_set = set()
    for q in qchain:
        q_tokens = tokenize_question(q)
        qchain_tokens.append(q_tokens)
        qchain_tokens_set.update(q_tokens)

    achain_tokens = []
    for a in achain:
        a_tokens = tokenize_question(str(a))
        achain_tokens.append(a_tokens)

    missed_comp_tokens = []
    for t in compq_tokens:
        if t not in qchain_tokens_set:
            missed_comp_tokens.append(t)

    new_simp_tokens = []
    curr_achain_tokens = []
    for qidx, qtokens in enumerate(qchain_tokens):
        for qtoken in qtokens:
            if qtoken in compq_tokens:
                if not repeat_ok:
                    compq_tokens.remove(qtoken)
            elif qtoken in curr_achain_tokens:
                if not repeat_ok:
                    curr_achain_tokens.remove(qtoken)
            else:
                new_simp_tokens.append(qtoken)
        if len(achain_tokens) > qidx:
            curr_achain_tokens.extend(achain_tokens[qidx])
    if score_answers:
        unmatched_answers = 0
        for aidx, atokens in enumerate(achain_tokens[:-1]):
            found_match = False
            # check if it overlaps with at least one question
            for qidx in range(aidx + 1, len(qchain_tokens)):
                if set(atokens).intersection(set(qchain_tokens[qidx])):
                    found_match = True
                    break
            # check if it overlaps with the last answer
            if not found_match and set(atokens).intersection(set(achain_tokens[-1])):
                found_match = True
            if not found_match:
                unmatched_answers += 1

    if compq_token_len == 0:
        print("No tokens in {}!!".format(complexq))
        new_token_score = len(new_simp_tokens)
        missed_token_score = len(missed_comp_tokens)
    else:
        new_token_score = len(new_simp_tokens) / compq_token_len
        missed_token_score = len(missed_comp_tokens) / compq_token_len
    if score_answers:
        return new_token_score, missed_token_score, new_simp_tokens, missed_comp_tokens, unmatched_answers
    else:
        return new_token_score, missed_token_score, new_simp_tokens, missed_comp_tokens
