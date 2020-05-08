from copy import deepcopy

import torch
from torch.nn.functional import softmax

from modularqa.inference.model_search import ParticipantModel
from modularqa.utils.seq_utils import get_sequence_representation
from modularqa.utils.seq_utils import score_question_answer_chain
from transformers import InputExample, \
    glue_convert_examples_to_features, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification



class LMClassifier():

    def __init__(self, model_path: str, model_type=None, max_seq_length=256,
                 task="sst-2", label_list=["0", "1"]):
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
        self.model_type = model_type if model_type is not None else self.model.config.model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.task = task

    def create_dataset(self, sequence1, sequence2=None):
        examples = [InputExample(guid="1", text_a=sequence1, text_b=sequence2,
                                 label=self.label_list[-1])]
        features = glue_convert_examples_to_features(examples, label_list=self.label_list,
                                                     max_length=self.max_seq_length,
                                                     tokenizer=self.tokenizer,
                                                     output_mode="classification",
                                                     task=self.task)

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
                  'labels': dataset[3]}
        if self.model_type != "distilbert":
            inputs["token_type_ids"] = (
                dataset[2] if self.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        outputs = self.model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        probs = softmax(logits)
        return probs.detach().cpu().numpy()[0]

    def score_sequence(self, sequence1: str, sequence2:str=None):
        dataset = self.create_dataset(sequence1=sequence1, sequence2=sequence2)
        scores = self.classify_dataset(dataset)
        return scores