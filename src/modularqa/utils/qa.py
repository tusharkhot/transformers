from time import time
from typing import List

import numpy as np
import torch

from examples.run_squad import to_list
from transformers import SquadV2Processor, squad_convert_examples_to_features
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, \
    compute_predictions_logits
from transformers.data.processors.squad import SquadResult


class QAAnswer(object):

    def __init__(self, answer_text, score, para_text):
        self.answer = answer_text
        self.score = score
        self.para_text = para_text

    def __str__(self):
        return "Ans:{} Score:{} Para:{}".format(self.answer, self.score, self.para_text)


def answer_question(question: str,
                    paragraphs: List[str],
                    model,
                    model_type,
                    tokenizer,
                    device, length,
                    num_ans_para,
                    single_para=False,
                    normalize=False,
                    return_unique_list=False) -> List[QAAnswer]:
    """
    Answer question using BERT
    :param question: input question
    :param title_doc_map: map from document title to a list of sentences corresponding to the
    document.
    :return: the answer to question (Returns "" is the question is unanswerable)
    """
    predicted_spans = []
    model.eval()
    # start = time()
    prediction_json = {}
    if single_para:
        for i, curr_paragraph in enumerate(paragraphs):
            examples, features, dataset = get_example_features_dataset(paragraphs, question,
                                                                       tokenizer,
                                                                       length)
            # print("Time to generate e,f,d: {}".format(time()-start))
            # start = time()
            curr_para_json = get_predictions(examples=examples,
                                             features=features,
                                             dataset=dataset, model_type=model_type,
                                             tokenizer=tokenizer,
                                             device=device,
                                             model=model, num_ans_per_para=num_ans_para)
            # print("Time to generate predictions: {}".format(time()-start))
            prediction_json[str(i)] = []
            for key, predictions in curr_para_json.items():
                prediction_json[str(i)].extend(predictions)
    else:
        examples, features, dataset = get_example_features_dataset(paragraphs, question, tokenizer,
                                                                   length)
        # print("Time to generate e,f,d: {}".format(time()-start))
        # start = time()
        prediction_json = get_predictions(examples=examples,
                                          features=features,
                                          dataset=dataset, model_type=model_type,
                                          tokenizer=tokenizer,
                                          device=device,
                                          model=model, num_ans_per_para=num_ans_para)
        # print("Time to generate predictions: {}".format(time()-start))

    for key, predictions in prediction_json.items():
        # simple 'hack' to get the paragraph
        para = paragraphs[int(key)]
        for pred in predictions:
            # only consider answers upto "unanswerable" per para
            if pred["text"] == "" or (pred["text"] == "empty" and pred["start_logit"] == 0):
                break
            else:
                predicted_spans.append((pred["text"], pred["probability"], para))

    if len(predicted_spans):
        answer_list = [QAAnswer(text, prob, para)
                       for text, prob, para in sorted(predicted_spans, key=lambda x: -x[1])]
        ## returns text (even if empty) and -log prob
        if normalize:
            # should have probability 1, there -log prob(1)  = 0
            denom = np.sum([ans.score for ans in answer_list])
            for ans in answer_list:
                ans.score = -np.log(ans.score / denom)
        else:
            for ans in answer_list:
                ans.score = -np.log(ans.score)

        if return_unique_list:
            observed_answers = []
            unique_list = []
            for bert_answer in answer_list:
                found_match = False
                answer_text = bert_answer.answer
                for obs_ans in observed_answers:
                    if obs_ans in answer_text or answer_text in obs_ans:
                        # ignore duplicate
                        found_match = True
                        break
                if not found_match:
                    unique_list.append(bert_answer)
                observed_answers.append(answer_text)
            answer_list = unique_list

        return answer_list
    else:
        if return_unique_list:
            return []
        else:
            return [QAAnswer("", 10.0, "")]


def get_example_features_dataset(paragraphs: List[str], question, tokenizer, seq_length):
    processor = SquadV2Processor()
    input_data = []
    for i, p in enumerate(paragraphs):
        input_data.append({
            "title": "",
            "paragraphs": [
                {
                    "context": p,
                    "qas": [
                        {
                            "id": str(i),
                            "question": question,
                            "answers": [""]
                        }
                    ]
                }
            ]
        })
    examples = processor._create_examples(input_data, "dev")
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        doc_stride=256,
        max_query_length=512,
        is_training=False,
        return_dataset="pt",
        threads=1,
        pad_to_max=False  # we will take care of padding
    )
    return examples, features, dataset


def get_predictions(examples, features, dataset, model_type, model, tokenizer, device,
                    num_ans_per_para):
    all_results = []
    # get the list of tensors
    dataset = dataset.tensors
    dataset = tuple(t.to(device) for t in dataset)
    with torch.no_grad():
        inputs = {
            "input_ids": dataset[0],
            "attention_mask": dataset[1],
            "token_type_ids": dataset[2],
        }
        # print("*** Example ***")
        # print("input_ids: %s" % " ".join([str(x) for x in dataset[0]]))
        # print("input_mask: %s" % " ".join([str(x) for x in dataset[1]]))

        if model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]

        example_indices = dataset[3]

        # XLNet and XLM use more arguments for their predictions
        if model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": dataset[4], "p_mask": dataset[5]})

        outputs = model(**inputs)

    for i, example_index in enumerate(example_indices):
        eval_feature = features[example_index.item()]
        unique_id = int(eval_feature.unique_id)

        output = [to_list(output[i]) for output in outputs]

        # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
        # models only use two.
        if len(output) >= 5:
            start_logits = output[0]
            start_top_index = output[1]
            end_logits = output[2]
            end_top_index = output[3]
            cls_logits = output[4]

            result = SquadResult(
                unique_id,
                start_logits,
                end_logits,
                start_top_index=start_top_index,
                end_top_index=end_top_index,
                cls_logits=cls_logits,
            )

        else:
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

        all_results.append(result)

    # XLNet and XLM use a more complex post-processing procedure
    if model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model,
                                                          "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model,
                                                      "config") else model.module.config.end_n_top

        predictions, nbest_predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            n_best_size=num_ans_per_para,
            max_answer_length=30,
            output_prediction_file="/dev/null",
            output_nbest_file="/dev/null",
            output_null_log_odds_file="/dev/null",
            start_n_top=start_n_top,
            end_n_top=end_n_top,
            version_2_with_negative=True,
            tokenizer=tokenizer,
            verbose_logging=False
        )
    else:
        predictions, nbest_predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size=num_ans_per_para,
            max_answer_length=30,
            output_prediction_file="/dev/null",
            output_nbest_file="/dev/null",
            output_null_log_odds_file="/dev/null",
            version_2_with_negative=True,
            tokenizer=tokenizer,
            verbose_logging=False,
            do_lower_case=False,
            null_score_diff_threshold=0.0,
        )

    return nbest_predictions
