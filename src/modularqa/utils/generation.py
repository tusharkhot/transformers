PREPROCESSING_FUNCTIONS = {
    "ctrl": None,
    "xlm": None,
    "xlnet": None,
    "transfo-xl": None,
}


def generate_text_sequence(model, tokenizer, model_type, prompt_text, device,
                           length=30, num_samples=1, temperature=1, top_k=0,
                           top_p=0.0, stop_token=None):
    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        raise ValueError("Model: {} requires preprocessing. Not implemented!".format(model_type))
    else:
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False,
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
        num_return_sequences=num_samples,
        decoder_start_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
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
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        print(text)
        # Remove all text after the stop token
        if stop_token is None:
            stop_token = tokenizer.eos_token
        if stop_token in text:
            text = text[: text.find(stop_token)]

        generated_sequences.append(text)

    return generated_sequences
