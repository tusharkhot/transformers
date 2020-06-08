import argparse
import json
import re
from copy import deepcopy

from modularqa.con_gen.constants import COMPQ_MARKER, INTERQ_MARKER, ANSWER_MARKER, SIMPQ_MARKER
from modularqa.con_gen.constraints import QAConstraint
from modularqa.con_gen.generator_verifier import QuestionGeneratorVerifier
from modularqa.utils.generation import LMGenerator
from modularqa.utils.seq_utils import score_question_answer_chain


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Generate labelled chains')
    arg_parser.add_argument('--input', type=str, required=True, help="Input JSON file")
    arg_parser.add_argument('--output', type=str, required=True, help="Output TXT file")
    arg_parser.add_argument('--config', type=str, required=True, help="Model configs")
    arg_parser.add_argument('--num_samples', type=int, default=5,
                            help="Number of questions to generate")
    arg_parser.add_argument('--decomposer_model', type=str, required=True,
                            help="Decomposer GPT model path")
    return arg_parser.parse_args()


def load_generator_verifiers(config_file):
    with open(config_file, "r") as config_fp:
        config_json = json.load(config_fp)
    gen_ver = {}
    for model_name, model_config in config_json.items():
        gen_ver[model_name] = QuestionGeneratorVerifier.load_from_json(model_config)
    return gen_ver


def create_labelled_chains(output_json, generator_verifiers,
                           decomposer_generator):
    global num_negative
    global num_chains
    positive_chains = []
    negative_chains = []
    question_pattern = "^\(([^\)]+)\) (.*)$"
    qaconstraints = [QAConstraint.from_json(c) for c in output_json["constraints"]]
    qam_chains = [([], [], [])]
    origq = output_json["question"]
    qid = output_json["id"]
    observed_sequences = set()
    # ignore constraints that are inadmissible given the verifiers. e.g. if boolq model is
    # not specified in the config, do not consider programs that rely on such a model
    # This also handles FAILED and OUTOFSCOPE questions
    for idx, constraint in enumerate(qaconstraints):
        if constraint.model not in generator_verifiers:
            return positive_chains, negative_chains

    for idx, constraint in enumerate(qaconstraints):
        newq_chains = []
        newa_chains = []
        newm_chains = []
        for q_chain, a_chain, m_chain in qam_chains:
            # construct sequence
            current_sequence = COMPQ_MARKER + origq
            for q, a, m in zip(q_chain, a_chain, m_chain):
                current_sequence += INTERQ_MARKER + "(" + m + ") " + q + ANSWER_MARKER + a
            current_sequence += SIMPQ_MARKER
            if current_sequence in observed_sequences:
                continue
            else:
                observed_sequences.add(current_sequence)
            # generate next question
            sequences = decomposer_generator.generate_sequences(sequence=current_sequence)
            sequences = list(set(sequences))
            predicted_questions = []
            predicted_sequences = []
            # extract question
            for new_q in sequences:
                sequence = current_sequence + new_q
                m = re.match(question_pattern, new_q.strip())
                if m:
                    model = m.group(1)
                    if constraint.model == model:
                        # only create +ve/-ve example when it matches expected model.
                        # alternatively, only create questions that match this model ?
                        predicted_question = m.group(2)
                        predicted_questions.append(predicted_question)
                        predicted_sequences.append(sequence)
                    else:
                        negative_chains.append(sequence)
                elif new_q == "[EOQ]":
                    # stopped too early
                    negative_chains.append(sequence)
                else:
                    print(
                        "Generated question: {} doesn't match expected pattern: {}\n"
                        "Output: {}".format(new_q, question_pattern, sequence))
                    negative_chains.append(sequence)
                    continue
            verifier = generator_verifiers[constraint.model].qa
            subquestions, subanswers, metadata = verifier.verify_questions(qid,
                                                                           constraint,
                                                                           predicted_questions,
                                                                           previous_questions=q_chain,
                                                                           previous_answers=a_chain)
            for question, predicted_seq in zip(predicted_questions, predicted_sequences):
                if question in subquestions:
                    # add to positive examples
                    positive_chains.append(predicted_seq)
                else:
                    negative_chains.append(predicted_seq)
            for q, a in zip(subquestions, subanswers):
                newq_chain = q_chain + [q]
                newa_chain = a_chain + [a]
                newm_chain = m_chain + [constraint.model]
                newa_chains.append(newa_chain)
                newq_chains.append(newq_chain)
                newm_chains.append(newm_chain)

        qam_chains = zip(newq_chains, newa_chains, newm_chains)
    # check if the model continues to generate questions
    for q_chain, a_chain, m_chain in qam_chains:
        current_sequence = COMPQ_MARKER + origq
        for q, a, m in zip(q_chain, a_chain, m_chain):
            current_sequence += INTERQ_MARKER + "(" + m + ") " + q + ANSWER_MARKER + a
        current_sequence += SIMPQ_MARKER
        # generate next question
        sequences = decomposer_generator.generate_sequences(sequence=current_sequence)
        sequences = list(set(sequences))
        for new_q in sequences:
            sequence = current_sequence + new_q
            if new_q == "[EOQ]":
                new_tok_score, missed_tok_score, new_toks, missed_toks, unmatched_answers = \
                    score_question_answer_chain(q_chain, a_chain, origq, repeat_ok=False,
                                                score_answers=True)
                if unmatched_answers > 0:
                    negative_chains.append(sequence)
                else:
                    positive_chains.append(sequence)
            else:
                negative_chains.append(sequence)

    return positive_chains, negative_chains


if __name__ == '__main__':
    args = parse_arguments()
    generator_verifiers = load_generator_verifiers(args.config)
    model_path = args.decomposer_model
    generator = LMGenerator(model_path=model_path, num_samples=args.num_samples, top_p=0.95)
    counter = 0
    num_pos = 0
    num_neg = 0
    with open(args.input, "r") as input_fp, open(args.output, "w") as output_fp:
        for line in input_fp:
            counter += 1
            input_json = json.loads(line)
            output_json = deepcopy(input_json)
            for m, gen_ver in generator_verifiers.items():
                gen_ver.reset_question_caches()
            positives, negatives = create_labelled_chains(input_json, generator_verifiers,
                                                          generator)
            num_pos += len(positives)
            num_neg += len(negatives)
            for pos in positives:
                output_fp.write(pos + "\t" + "1\n")
            for neg in negatives:
                output_fp.write(neg + "\t" + "0\n")
            if counter % 100 == 0:
                print("Num Lines: {}".format(counter))
                print("Num Pos: {}".format(num_pos))
                print("Num Neg: {}".format(num_neg))
    print("Num Lines: {}".format(counter))
    print("Num Pos: {}".format(num_pos))
    print("Num Neg: {}".format(num_neg))
