import json
import sys

from modularqa.retrievers.file_retriever import FileRetriever


def output_for_skiff_demo(file_retriever, output_file):
    # OUTPUT FOR SKIFF DEMO
    dropq_arr = [
        ("5add39e75542997545bbbcc4",
         "Little Big Girl was a Simpsons episode directed by the animator and artist of what nationality?"),
        ("5adcf2b55542992c1e3a24cd",
         "How many children's books has the writer of the sitcom Maid Marian and her Merry Men written ?"),
        ("5ab916e155429919ba4e23a4",
         "12 Years a Slave starred what British actor born 10 July 1977)"),
        ("fc10427d-d468-437c-8bc5-bb3953f9ac5b",
         "Which ancestral group is smaller: Irish or Italian?"),
        ("17169f44-cb53-4809-9dc7-19872cc70480",
         "How many months after 25 villages belonging to Memmingen rebelled were the Twelve Articles agreed on by the Upper Swabian Peasants Confederation?"),
        ("7e372248-75c4-4165-acc4-f048d9d5f189",
         "How many percent of the national population does not live in Bangkok?"),
        ("6a764941-9eab-4958-a752-e0959efa8aba",
         "How many years did it take for the services sector to rebound?")
    ]
    output_json = []
    for id, question in dropq_arr:
        output_json.append({
            "question": question,
            "paragraphs": file_retriever.retrieve_paragraphs(id, question)
        })
    with open(output_file, "w") as output_fp:
        json.dump(output_json, output_fp, indent=2)


output_file = sys.argv[1]

file_retriever = FileRetriever(
    hotpotqa_file="../launchpadqa/data/hotpotqa/hotpot_dev_distractor_v1.json",
    drop_file="../launchpadqa/data/drop_inference_sets/drop_dataset_dev.json")
output_for_skiff_demo(file_retriever, output_file)
