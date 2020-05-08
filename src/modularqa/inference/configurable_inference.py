### An example controller that uses our existing bert and gpt models
import argparse
import json
import logging

from modularqa.inference.constants import MODEL_NAME_CLASS, READER_NAME_CLASS
from modularqa.inference.dataset_readers import DatasetReader
from modularqa.inference.model_search import (
    ModelController,
    BestFirstDecomposer, QuestionGeneratorData)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Convert HotPotQA dataset into SQUAD format')
    arg_parser.add_argument('--input', type=str, required=True, help="Input QA file")
    arg_parser.add_argument('--output', type=str, required=True, help="Output file")
    arg_parser.add_argument('--config', type=str, required=True, help="Model configs")
    arg_parser.add_argument('--reader', type=str, required=True, help="Dataset reader",
                            choices=READER_NAME_CLASS.keys())
    arg_parser.add_argument('--debug', action='store_true', default=False,
                            help="Debug output")
    arg_parser.add_argument('--demo', action='store_true', default=False,
                            help="Demo mode")
    return arg_parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    args = parse_arguments()

    ################################
    # ## step 1: initialize models #
    ################################
    with open(args.config, "r") as input_fp:
        config_map = json.load(input_fp)

    print("loading participant models (might take a while)...")
    for key, value in config_map["models"].items():
        class_name = value.pop("name")
        if class_name not in MODEL_NAME_CLASS:
            raise ValueError("No class mapped to model name: {} in MODEL_NAME_CLASS:{}".format(
                class_name, MODEL_NAME_CLASS))
        model = MODEL_NAME_CLASS[class_name](**value)
        if key in config_map:
            raise ValueError("Overriding key: {} with value: {} using instantiated model of type:"
                             " {}".format(key, config_map[key], class_name))
        config_map[key] = model.query

    ###############################################################
    # ## step 2: add to controller spec and initialize controller #
    ###############################################################

    ## instantiating
    controller = ModelController(config_map, QuestionGeneratorData)
    decomposer = BestFirstDecomposer(controller)
    reader: DatasetReader = READER_NAME_CLASS[args.reader]()
    #############################
    # ## step 3: run decomposer #
    #############################
    print("Running decomposer on examples")
    output_json = {}

    if args.demo:
        while True:
            qid = input("QID: ")
            question = input("Question: ")
            example = {
                "qid": qid,
                "query": question,
                "question": question
            }
            final_state = decomposer.find_answer_decomp(example, debug=args.debug)
            if final_state is None:
                print("FAILED!")
            else:
                data = final_state._data
                chain = example["question"]
                for q, a in zip(data["question_seq"], data["answer_seq"]):
                    chain += " Q: {} A: {}".format(q, a)
                chain += " S: " + str(final_state._score)
                print(chain)
    else:
        for example in reader.read_examples(args.input):
            final_state = decomposer.find_answer_decomp(example, debug=args.debug)
            if final_state is None:
                print(example["question"] + " FAILED!")
                output_json[example["qid"]] = ""
            else:
                data = final_state._data
                chain = example["question"]
                for q, a in zip(data["question_seq"], data["answer_seq"]):
                    chain += " Q: {} A: {}".format(q, a)
                chain += " S: " + str(final_state._score)
                print(chain)
                output_json[example["qid"]] = final_state._data["answer_seq"][-1]

        with open(args.output, "w") as output_fp:
            json.dump(output_json, output_fp)