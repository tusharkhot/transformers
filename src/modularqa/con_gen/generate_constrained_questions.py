import argparse
import json
from copy import deepcopy

from tqdm import tqdm

from modularqa.con_gen.generator_verifier import QuestionGeneratorVerifier
from modularqa.con_gen.constraints import QAConstraint


num_chains = 0
num_dropped = 0


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Generate question chains from constrained programs')
    arg_parser.add_argument('--input', type=str, required=True, help="Input JSON file")
    arg_parser.add_argument('--output', type=str, required=True, help="Output TXT file")
    arg_parser.add_argument('--config', type=str, required=True, help="Model configs")
    arg_parser.add_argument('--max_new_score', type=float, required=False, default=-1.0,
                            help="Drop question chains in generation when new token score exceeds this threshold")
    return arg_parser.parse_args()


def load_generator_verifiers(config_file):
    with open(config_file, "r") as config_fp:
        config_json = json.load(config_fp)
    gen_ver = {}
    for model_name, model_config in config_json.items():
        gen_ver[model_name] = QuestionGeneratorVerifier.load_from_json(model_config)
    return gen_ver


def create_question_chains(output_json, generator_verifiers):
    global num_dropped
    global num_chains
    qaconstraints = [QAConstraint.from_json(c) for c in output_json["constraints"]]
    qa_chains = [(None, None)]
    origq = output_json["question"]
    qid = output_json["id"]
    for idx, constraint in enumerate(qaconstraints):
        model = constraint.model
        output_json["constraints"][idx]["subquestions"] = []
        output_json["constraints"][idx]["subanswers"] = []
        if model in generator_verifiers:
            output_json["constraints"][idx]["metadata"] = []
            generator_verifier = generator_verifiers[model]
            newq_chains = []
            newa_chains = []
            for q_chain, a_chain in qa_chains:
                subquestions, subanswers, metadata = generator_verifier.generate_questions(
                    qid=qid,
                    constraint=constraint,
                    previous_questions=q_chain,
                    previous_answers=a_chain
                )
                if len(subquestions):
                    output_json["constraints"][idx]["subquestions"].append(subquestions)
                    output_json["constraints"][idx]["subanswers"].append(subanswers)
                metadata["dropped_chains"] = []
                for q, a in zip(subquestions, subanswers):
                    if q_chain is not None:
                        newq_chain = q_chain + [q]
                        newa_chain = a_chain + [a]
                    else:
                        newq_chain = [q]
                        newa_chain = [a]
                    # new_tok_score, missed_tok_score, new_toks, missed_toks = \
                    #     score_question_answer_chain(newq_chain, newa_chain, origq, repeat_ok=True)
                    # if max_new_score < 0 or new_tok_score < max_new_score:
                    newa_chains.append(newa_chain)
                    newq_chains.append(newq_chain)
                    # else:
                    #     num_dropped += 1
                    #     metadata["dropped_chains"].append({
                    #         "qchain": newq_chain,
                    #         "achain": newa_chain,
                    #         "new_toks": new_toks,
                    #         "missed_toks": missed_toks
                    #     })
                output_json["constraints"][idx]["metadata"].append(metadata)
            qa_chains = zip(newq_chains, newa_chains)
        else:
            output_json["constraints"][idx]["metadata"] = {"error": "Model Not Found!!"}

    output_json["qa_chains"] = list(zip(*qa_chains))
    if len(output_json["qa_chains"]):
        if len(output_json["qa_chains"]) == 1 and output_json["qa_chains"][0][0] is None:
            output_json["qa_chains"] = []
        else:
            num_chains += len(output_json["qa_chains"][0])
    return output_json


if __name__ == '__main__':
    args = parse_arguments()
    generator_verifiers = load_generator_verifiers(args.config)
    counter = 0
    with open(args.input, "r") as input_fp, open(args.output, "w") as output_fp:
        for line in tqdm(input_fp):
            input_json = json.loads(line)
            output_json = deepcopy(input_json)
            for m, gen_ver in generator_verifiers.items():
                gen_ver.reset_question_caches()
            output_json = create_question_chains(output_json, generator_verifiers)
            counter += 1
            if counter % 100 == 0:
                for m, gen_ver in generator_verifiers.items():
                    print("=========")
                    for m in gen_ver.time_per_model.keys():
                        print(m, gen_ver.time_per_model[m], gen_ver.count_per_model[m])

            output_fp.write(json.dumps(output_json) + "\n")

    print("Num chains: {}".format(num_chains))
    print("Dropped chains: {}".format(num_dropped))
