import torch
from torch.nn.functional import softmax

from modularqa.inference.model_search import ParticipantModel
from modularqa.utils.seq_utils import get_sequence_representation
from modularqa.utils.seq_utils import score_question_answer_chain
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, InputExample, \
    glue_convert_examples_to_features, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class QualityCheckerExample(ParticipantModel):

    def query(self, state, debug=False):
        """Checks the quality of a given question. In this case, it does
        nothing. This shows how you might build functions that do something,
        e.g., change the score of an example, but that don't change the state.

        :param state: pass
        """
        ## state data
        data = state.data
        ## copy the state
        new_state = state.copy()
        new_state._next = "qa"
        if debug: print("<QUALITYCHECK>: %s" % data["question_seq"][-1])

        ## checks last question (stored in data["question_seq"]) and sees
        ## if it has any overlap with original question (stored in data["query"])
        initial_question_words = set(data["query"].split())
        if not [i for i in data["question_seq"][-1].split() \
                if i not in initial_question_words]:
            ## if so, manipulates the score to be infinity
            new_state.last_output = "terrible quality!"
            new_state._score = float('inf')

        else:
            new_state.last_output = "good quality!"

        return new_state


class ChainOverlapScorer(ParticipantModel):

    def query(self, state, debug=False):
        ## state data
        data = state.data
        ## copy the state
        new_state = state.copy()
        new_state._next = "qa"
        origq = data["query"]
        qchain = data["question_seq"]
        achain = data["answer_seq"]
        score_answers = False
        if qchain[-1] == "[EOQ]":
            score_answers = True
            new_state._next = "EOQ"
            qchain.pop(-1)

        if debug:
            print("<QUALITYCHECK>: Qs: {} As: {} Q: {}".format(
                ", ".join(qchain), ", ".join(achain), origq))
        if score_answers:
            new_tok_score, missed_tok_score, new_toks, missed_toks, unmatched_answers = \
                score_question_answer_chain(qchain, achain, origq, repeat_ok=False,
                                            score_answers=score_answers)
            if unmatched_answers > 0:
                if debug:
                    print("Unmatched answers! Rejecting!"
                          "Qs: {} As: {} Q: {}".format(", ".join(qchain), ", ".join(achain), origq))
                new_state._score = float("inf")
            else:
                new_state._score = new_tok_score + missed_tok_score
        else:
            new_tok_score, missed_tok_score, new_toks, missed_toks = \
                score_question_answer_chain(qchain, achain, origq, repeat_ok=False,
                                            score_answers=score_answers)
            new_state._score = new_tok_score

        new_state.last_output = "Missed: {} New: {}".format(",".join(missed_toks),
                                                            ",".join(new_toks))

        return new_state


class BertQualityChecker(ParticipantModel):

    def __init__(self, model_path: str, max_seq_length=256):
        config = AutoConfig.from_pretrained(
            model_path,
            cache_dir=None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            do_lower_case=False,
            cache_dir=None,
        )
        print("Loading {} model from: {}".format(config.model_type, model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            from_tf=False,
            config=config,
            cache_dir=None,
        )
        # config_class, model_class, tokenizer_class = \
            #     (BertConfig, BertForSequenceClassification, BertTokenizer)
        # print("Loading BERT model from: {}".format(model_path))
        # config = config_class.from_pretrained(model_path)
        # self.tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=False)
        # self.model = model_class.from_pretrained(model_path, from_tf=False, config=config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_seq_length = max_seq_length

    def create_dataset(self, sequence):
        examples = [InputExample(guid="1", text_a=sequence, text_b=None, label="1")]
        features = glue_convert_examples_to_features(examples, label_list=["0", "1"],
                                                max_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                output_mode="classification",
                                                task="sst-2")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = (all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def classify_dataset(self, dataset):
        dataset = tuple(t.to(self.device) for t in dataset)
        inputs = {'input_ids': dataset[0],
                  'attention_mask': dataset[1],
                  'token_type_ids': dataset[2],
                  'labels': dataset[3]}
        outputs = self.model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        probs = softmax(logits)
        return probs.detach().cpu().numpy()[0]

    def query(self, state, debug=False):
        ## state data
        data = state.data
        ## copy the state
        new_state = state.copy()
        new_state._next = "qa"
        origq = data["query"]
        qchain = data["question_seq"]
        achain = data["answer_seq"]
        score_answers = False
        if qchain[-1] == "[EOQ]":
            new_state._next = "EOQ"
            return new_state

        if debug:
            print("<QUALITYCHECK>: Qs: {} As: {} Q: {}".format(
                ", ".join(qchain), ", ".join(achain), origq))

        sequence = get_sequence_representation(origq, qchain, achain, for_generation=False)
        dataset = self.create_dataset(sequence)
        output_probs = self.classify_dataset(dataset)

        #print(sequence, output_probs)
        new_state._score += output_probs[0] # higher is worse; so take prob of 0
        new_state.last_output = "Score: {}".format(output_probs)

        return new_state
