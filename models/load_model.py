import torch
from models.vae import MultimodalVAE
from transformers import  AutoTokenizer


class Tokenizer:
    def __init__(self, max_length, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.max_length=max_length

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

def load_model():

    dataset='Flickr30k'
    latent_dim=64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")

    model_checkpoint=('/home/chatziko/PycharmProjects/PythonProject/Multimodal-VAE/results/'
                      'results_flickr_both_modalities_kl_1_latent_dim_var/latent_64/'
                      'final_model_kl_coef_1.0_lr_0.01_latent_dim_64.pt')
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)

    num_attributes=32

    tokenizer = Tokenizer(32, AutoTokenizer.from_pretrained("facebook/bart-base"))


    vocab_size = tokenizer.tokenizer.vocab_size

    model = MultimodalVAE(device,tokenizer, vocab_size, dataset,
        latent_dim=latent_dim, num_attributes=num_attributes,
        temperature=1.0).to(device)


    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    return model, tokenizer
