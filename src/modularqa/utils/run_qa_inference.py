import argparse
import json

from modularqa.utils.qa import LMQuestionAnswerer


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Run QA inference')
    arg_parser.add_argument('--dataset_type', type=str, required=False,
                            default="squad", choices=["hotpot", "drop", "squad"],
                            help="Dataset type")
    arg_parser.add_argument('--input', type=str, required=False, help="Input File")
    arg_parser.add_argument('--output', type=str, required=False, help="Output File")
    arg_parser.add_argument('--output_type', type=str, required=False, default="single",
                            choices=["single", "list"])
    arg_parser.add_argument('--qa', type=str, required=True,
                            help="Squad QA Model")
    arg_parser.add_argument('--qa_args', type=str, required=False,
                            action='append', default=[],
                            help="Set particular variables for the QA model. Format: key:value."
                                 " E.g. \"num_ans:20\".")
    arg_parser.add_argument('--demo', action='store_true', default=False,
                            help="Demo mode")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    kwargs = {}
    # initialize kwargs from cmd line
    for qa_arg in args.qa_args:
        key, value = qa_arg.split(":")
        try:
            value_f = float(value)
            kwargs[key] = value_f
        except ValueError:
            try:
                value_b = bool(value)
                kwargs[key] = value_b
            except ValueError:
                kwargs[key] = value
    qa_model = LMQuestionAnswerer(model_path=args.qa, **kwargs)

    if args.demo:
        while True:
            para = input("Para:")
            question = input("Question:")
            answers = qa_model.answer_question(question=question, paragraphs=[para])
            for ans in answers:
                print(ans.answer, ans.score)
        quit()

    # load json from file
    with open(args.input, "r") as input_fp:
        input_json = json.load(input_fp)
    # populate dataset with (qid, question, [paras]) tuples
    dataset = []
    if args.dataset_type == "squad":
        for data_item in input_json["data"]:
            for paragraph in data_item["paragraphs"]:
                para = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    qid = qa["id"]
                    dataset.append((qid, question, [para]))
    elif args.dataset_type == "drop":
        for paraid, item in input_json.items():
            para = item["passage"]
            for qa_pair in item["qa_pairs"]:
                question = qa_pair["question"]
                qid = qa_pair["query_id"]
                dataset.append((qid, question, [para]))
    else:
        raise ValueError("Cannot handle dataset type: {}".format(args.dataset_type))

    # compute predictions
    predictions = {}
    for (qid, question, paras) in dataset:
        ans = qa_model.answer_question(question=question, paragraphs=paras)
        if args.output_type == "single":
            predictions[qid] = ans[0].answer
        else:
            predictions[qid] = [a.answer for a in ans]

    with open(args.output, "w") as output_fp:
        json.dump(predictions, output_fp, indent=2)
