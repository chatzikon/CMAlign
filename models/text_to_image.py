import torch


def txt_to_img(model, tokenizer, texts):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")

    input_tokens = tokenizer(texts).to(device)
    target_attributes = input_tokens['input_ids']
    text_attrs = target_attributes
    text_attrs = text_attrs.transpose(0, 1)

    pad_id = tokenizer.tokenizer.pad_token_id

    pad_mask = (text_attrs == pad_id).transpose(0,1)
    generated_images = model.generate_from_text(text_attrs, pad_mask)

    return generated_images