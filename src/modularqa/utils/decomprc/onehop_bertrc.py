import collections
import logging
import sys
from typing import List

import numpy as np
import six
import torch
from torch.nn.functional import sigmoid
from tqdm import tqdm

from modularqa.retrievers.retriever import Retriever
from modularqa.utils.decomprc.modeling_bert import BertForQuestionAnswering, BertConfig
from modularqa.utils.decomprc.multipara_prepro import get_dataloader
from modularqa.utils.decomprc.tokenization import FullTokenizer, BasicTokenizer
from modularqa.utils.qa import QAAnswer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class OneHopBertRC:

    def __init__(self, model_path, config_path, vocab_path,
                 only_gold_para=False, return_all_ans=False,
                 seq_length=512, num_ans_para=1, single_para=False, merge_select_para=False,
                 return_unique_list=False, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.load_models(model_path, config_path, self.device)
        self.tokenizer = FullTokenizer(vocab_file=vocab_path, do_lower_case=True)
        self.seq_length = seq_length
        self.num_ans_para = num_ans_para
        self.single_para = single_para
        self.return_unique_list = return_unique_list
        self.merge_select_para = merge_select_para
        self.return_all_ans = return_all_ans
        self.retriever = Retriever.load_retriever(**kwargs)

    def load_models(self, checkpoint_paths, config_path, device):
        bert_config = BertConfig.from_json_file(config_path)
        model = []
        for i, checkpoint in enumerate(checkpoint_paths.split(',')):
            model.append(BertForQuestionAnswering(bert_config, 4))
            print("Loading from", checkpoint)
            state_dict = torch.load(checkpoint, map_location='cpu')
            filter = lambda x: x[7:] if x.startswith('module.') else x
            state_dict = {filter(k): v for (k, v) in state_dict.items()}
            model[-1].load_state_dict(state_dict)
            model[-1].eval()
            model[-1].to(device)
        return model

    def answer_question_only(self, question: str, qid: str, normalize=False,
                             num_ans: int = None, return_all_ans: bool = None):
        """
        Answer question using title+paras for the question corresponding to this QID
        :param question: input question
        :param qid: question id
        :return: answer
        """
        paragraphs = self.retriever.retrieve_paragraphs(qid, question)
        return self.answer_question(question, paragraphs, normalize=normalize,
                                    num_ans_para=num_ans)

    def answer_question(self, question: str,
                        paragraphs: List[str],
                        num_ans_para=None,
                        normalize=False,
                        return_unique_list=False):
        if num_ans_para is None:
            num_ans_para = self.num_ans_para
        dataloader, examples, eval_features = get_dataloader("qa", [question], paragraphs,
                                                             self.tokenizer, batch_size=1)
        predictions = self.get_qa_prediction(dataloader, examples, eval_features, num_ans_para)
        # print(predictions)

        answer_list = [QAAnswer(t["text"], t["logit"], t["para"]) for t in predictions[0]]
        if len(answer_list):
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

    def get_qa_prediction(self, dataloader, examples, eval_features, num_ans_para):
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits", "switch"])
        all_results = []

        def _get_raw_results(model1):
            raw_results = []
            for batch in dataloader:
                example_indices = batch[-1]
                if self.device != -1:
                    batch_to_feed = [t.to(self.device) for t in batch[:-1]]
                else:
                    batch_to_feed = batch[:-1]
                with torch.no_grad():
                    batch_start_logits, batch_end_logits, batch_switch = model1(batch_to_feed)

                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    switch = batch_switch[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    raw_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits,
                                                 switch=switch))
            return raw_results

        all_raw_results = [_get_raw_results(m) for m in self.models]
        for i in range(len(all_raw_results[0])):
            result = [all_raw_result[i] for all_raw_result in all_raw_results]
            assert all([r.unique_id == result[0].unique_id for r in result])
            start_logits = sum([np.array(r.start_logits) for r in result]).tolist()
            end_logits = sum([np.array(r.end_logits) for r in result]).tolist()
            switch = sum([np.array(r.switch) for r in result]).tolist()
            all_results.append(RawResult(unique_id=result[0].unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         switch=switch))

        return self.format_predictions(examples, eval_features, all_results, num_ans_para)

    def format_predictions(self, examples, features, results, num_ans_para):

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "logit", "no_answer_logit"])

        example_index_to_features = collections.defaultdict(list)
        for feature in features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in results:
            unique_id_to_result[result.unique_id] = result

        predictions_list = []
        for example_index, example in enumerate(examples):
            # print(example_index)
            prelim_predictions = []
            for (feature_index, feature) in enumerate(example_index_to_features[example_index]):
                result = unique_id_to_result[feature.unique_id]
                switch = np.argmax(result.switch[:3])
                if switch > 0:
                    prelim_predictions.append(_PrelimPrediction(
                        feature_index=feature_index, start_index=-switch, end_index=-switch,
                        logit=result.switch[switch] - result.switch[3],
                        no_answer_logit=result.switch[3]))
                    continue

                scores = []
                start_logits = result.start_logits[:len(feature.tokens)]
                end_logits = result.end_logits[:len(feature.tokens)]
                for (i, s) in enumerate(start_logits):
                    for (j, e) in enumerate(end_logits[i:i + 10]):
                        scores.append(((i, i + j), s + e - result.switch[3]))

                scores = sorted(scores, key=lambda x: x[1], reverse=True)

                for (start_index, end_index), score in scores:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > 10:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            logit=score, no_answer_logit=result.switch[3]))

            predictions = []
            para_predictions = {}
            for pred in sorted(prelim_predictions, key=lambda x: x.logit, reverse=True):
                feature = example_index_to_features[example_index][pred.feature_index]
                para = " ".join(feature.doc_tokens)
                # keep processing other para predictions
                if para in para_predictions:
                    if para_predictions[para] == num_ans_para:
                        continue
                else:
                    para_predictions[para] = 0

                if pred.start_index == pred.end_index == -1:
                    final_text = "yes"
                    sp_fact = " ".join(feature.doc_tokens)
                elif pred.start_index == pred.end_index == -2:
                    final_text = "no"
                    sp_fact = " ".join(feature.doc_tokens)
                else:
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    try:
                        orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    except Exception:
                        print("Error during postprocessing")
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text)
                    sp_fact = " ".join(feature.doc_tokens[:orig_doc_start] + ["@@"] + \
                                       orig_tokens + ["@@"] + feature.doc_tokens[orig_doc_end + 1:])

                predictions.append({'text': final_text, 'logit': pred.logit, 'para': sp_fact,
                                    'noans': pred.no_answer_logit})
                para_predictions[para] += 1
            predictions_list.append(predictions)
            # print(predictions_list)
        return predictions_list


def get_final_text(pred_text, orig_text, do_lower_case=True):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


if __name__ == "__main__":
    q = "What is McCain's job?"
    p = ["Obama is the president of USA.", "Obama is a senator", "Biden is an actor."]
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    vocab_file = sys.argv[3]
    onehopqa = OneHopBertRC(model_path=model_path, config_path=config_path, vocab_path=vocab_file,
                            num_ans_para=2)
    preds = onehopqa.answer_question(question=q, paragraphs=p)
    for pred in preds:
        print(pred)
