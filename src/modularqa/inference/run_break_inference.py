import csv
import re
import sys

from modularqa.utils.generation import LMGenerator


def load_questions(break_file):
    qid_question_map = {}
    with open(break_file, "r") as break_fp:
        break_csv = csv.reader(break_fp)
        header = next(break_csv)
        for row in break_csv:
            orig_q = row[header.index("question_text")]
            qid = row[header.index("question_id")]
            question = row[header.index("question_text")]
            qid_question_map[qid] = question
    print("Found {} questions".format(len(qid_question_map)))
    return qid_question_map


break_csv = sys.argv[1]
output_csv = sys.argv[2]
model_path = sys.argv[3]

qid_question_map = load_questions(break_csv)
lm_generator = LMGenerator(model_path=model_path,
                           length=40,
                           num_samples=1,
                           num_beams=4,
                           do_sample=False)

with open(output_csv, "w") as output_writer:
    output_writer.write("qid, decomposition\n")
    for (qid, question) in qid_question_map.items:
        maxqs = 10
        curr_seq = " QC: " + question
        question_seq = []
        while maxqs > 0:
            maxqs = maxqs - 1
            output, score = lm_generator.generate_sequences(curr_seq + " QS: ")
            m = re.match("\(([a-zA-Z]+)\) (.*)", output[0])
            if m:
                outq = m.group(2)
                question_seq.append(outq)
                curr_seq + " QI: " + outq
            else:
                if output[0] != "[EOQ]":
                    print("No pattern match for {}".format(output[0]))
                break
        output_writer.write(qid + "," + " ; ".join(question_seq) + "\n")
