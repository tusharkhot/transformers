import json
import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class ConditionalTextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, conditional_split: str,
                 output_dir: str, overwrite_cache=False,
                 block_size=512):
        lines = []
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            output_dir,
            "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            if file_path.endswith(".txt") or file_path.endswith(".tsv"):
                with open(file_path, encoding="utf-8") as f:
                    lines = [line for line in f if (len(line) > 0 and not line.isspace())]
            else:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        input_json = json.loads(line)
                        if "train_seq" in input_json:
                            lines.append(input_json["train_seq"])
                        elif "train_seqs" in input_json:
                            lines.extend(input_json["train_seqs"])
                        else:
                            raise ValueError("No training sequences in input json: {}".format(line))

            self.examples = []
            split_str = conditional_split
            for text in lines:
                if split_str is not None and len(split_str) > 0:
                    rhs_start_idx = text.index(split_str) + len(split_str)
                else:
                    rhs_start_idx = 0
                lhs_str = text[:rhs_start_idx]
                rhs_str = text[rhs_start_idx:]

                tokenized_lhs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lhs_str))
                tokenized_rhs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(rhs_str))
                # DONT use padding token. Internal code uses that for masking
                eos_token = tokenizer.eos_token_id or tokenizer.bos_token_id
                bos_token = tokenizer.bos_token_id or tokenizer.eos_token_id
                self._truncate_seq_pair(tokenized_lhs, tokenized_rhs,
                                        max_length=block_size - 1)
                attention_mask = [1] * len(tokenized_lhs)

                self.examples.append((tokenized_lhs,  # input
                                      tokenized_rhs + [eos_token], #labels
                                      attention_mask,  # mask
                                      [bos_token] + tokenized_rhs # decoder
                                      ))
                if len(self.examples) < 5:
                    print(lhs_str)
                    print(rhs_str)
                    print(self.examples[-1][0])
                    print(self.examples[-1][1])
                    print(self.examples[-1][3])
            logger.info("Saving features into cached file %s", cached_features_file)
            os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. tokens_a is always truncated from the beginning and tokens_b is
        # truncated from the end.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item][0], dtype=torch.long),
                torch.tensor(self.examples[item][1], dtype=torch.long),
                torch.tensor(self.examples[item][2], dtype=torch.long),
                torch.tensor(self.examples[item][3], dtype=torch.long))