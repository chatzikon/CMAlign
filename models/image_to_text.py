def img_to_txt(image, model, tokenizer):

    dataset='Flickr30k'

    generated_texts = model.generate_from_image(image, dataset, tokenizer)


    return generated_texts