import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead

PREPROCESSING_FUNCTIONS = {
    "ctrl": None,
    "xlm": None,
    "xlnet": None,
    "transfo-xl": None,
}


class LMGenerator:
    path_to_modeltokenizer = {}

    def __init__(self,
                 model_path,
                 model_type=None,
                 length=30,
                 num_samples=20,
                 top_p=0.9,
                 top_k=None,
                 num_beams=None,
                 add_bos=False,
                 temperature=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## set up the model
        self.model, self.tokenizer = LMGenerator.load_model_tokenizer(model_path, device=self.device)
        self.model_type = model_type if model_type is not None else self.model.config.model_type
        self.model.eval()
        self.length = length
        self.num_samples = num_samples
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.num_beams = num_beams
        self.add_bos = add_bos

    def generate_sequences(self, sequence, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        outputs = generate_text_sequence(model=self.model, prompt_text=sequence,
                                         model_type=self.model_type,
                                         length=self.length,
                                         num_samples=num_samples,
                                         temperature=self.temperature,
                                         top_k=self.top_k, top_p=self.top_p,
                                         add_bos=self.add_bos,
                                         num_beams=self.num_beams,
                                         tokenizer=self.tokenizer, device=self.device)

        return outputs

    @staticmethod
    def load_model_tokenizer(model_path, device):
        if model_path in LMGenerator.path_to_modeltokenizer:
            return LMGenerator.path_to_modeltokenizer[model_path]
        else:
            config = AutoConfig.from_pretrained(
                model_path,
                cache_dir=None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                do_lower_case=False,
                cache_dir=None,
            )
            print("Loading {} model from: {}".format(config.model_type, model_path))
            model = AutoModelWithLMHead.from_pretrained(
                model_path,
                from_tf=False,
                config=config,
                cache_dir=None,
            )
            model.to(device)
            LMGenerator.path_to_modeltokenizer[model_path] = (model, tokenizer)
            return model, tokenizer


def generate_text_sequence(model, tokenizer, model_type, prompt_text, device,
                           length=30, num_samples=1, temperature=1, top_k=None, num_beams=None,
                           top_p=None, stop_token=None, add_bos=False):
    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        raise ValueError("Model: {} requires preprocessing. Not implemented!".format(model_type))
    else:
        if add_bos:
            prompt_text = tokenizer.bos_token + prompt_text
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False,
                                          max_length=tokenizer.max_len - length,
                                          return_tensors="pt")

    encoded_prompt = encoded_prompt.to(device)
    if model.config.is_encoder_decoder:
        max_len = length
    else:
        max_len = length + len(encoded_prompt[0])
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_beams=num_beams, no_repeat_ngram_size=3,
        num_return_sequences=num_samples,
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=model.config.pad_token_id or model.config.eos_token_id
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        if model.config.is_encoder_decoder:
            # remove the first input token
            generated_sequence = generated_sequence[1:]
        else:
            generated_sequence = generated_sequence[len(encoded_prompt[0]):]
        if add_bos:
            generated_sequence = generated_sequence[1:]
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after the stop token
        if stop_token is None:
            stop_token = tokenizer.eos_token
        if stop_token in text:
            text = text[: text.find(stop_token)]

        generated_sequences.append(text)

    return generated_sequences
