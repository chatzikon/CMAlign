# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import wandb
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_config, flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, set_seed
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from datasets.utils import image_transform, resize_and_pad_image, to_tensor_and_normalize

# set_seed(10)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch


def tsne_calc(z_images,z_texts):
    tsne = TSNE(n_components=4, perplexity=2, max_iter=1000, random_state=42, method="exact")

    #X1=z_images

    # print(X1.shape)
    # print("NaN:", np.isnan(X1).any())
    # print("Inf:", np.isinf(X1).any())
    # print("global std:", np.std(X1))
    # print("per-feature zero std:", np.sum(np.std(X1, axis=0) == 0))
    # print("unique rows:", np.unique(X1, axis=0).shape[0])
    #
    # X2=z_texts
    # print(X2.shape)
    # print("NaN:", np.isnan(X2).any())
    # print("Inf:", np.isinf(X2).any())
    # print("global std:", np.std(X2))
    # print("per-feature zero std:", np.sum(np.std(X2, axis=0) == 0))
    # print("unique rows:", np.unique(X2, axis=0).shape[0])

    print('img')
    z_img_tsne = tsne.fit_transform(z_images)
    print('txt')
    z_txt_tsne = tsne.fit_transform(z_texts)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'wspace': 0.3})



    ax1.scatter(
        z_img_tsne[:, 0],
        z_img_tsne[:, 1],
        # c=colors[label],
        #label=f'Image ({label})',
        alpha=0.7,
        s=80
    )

    ax1.set_title('Image Embeddings', fontsize=16)
    ax1.set_xlabel('t-SNE 1', fontsize=14)
    ax1.set_ylabel('t-SNE 2', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        z_txt_tsne[:, 0],
        z_txt_tsne[:, 1],
        # c=colors[label],
        # label=f'Text ({label})',
        alpha=0.7,
        s=80
    )

    ax2.set_title('Text Embeddings', fontsize=16)
    ax2.set_xlabel('t-SNE 1', fontsize=14)
    ax2.set_ylabel('t-SNE 2', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(True, alpha=0.3)

    num_lines = min(30, 12)


    for i in range(num_lines):
        con = ConnectionPatch(
            xyA=(z_img_tsne[i, 0], z_img_tsne[i, 1]),
            xyB=(z_txt_tsne[i, 0], z_txt_tsne[i, 1]),
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax2,
            alpha=0.4,
            linestyle="--",
            linewidth=0.8
        )
        fig.add_artist(con)

    plt.suptitle('Cross-Modal Latent Space Alignment', fontsize=18, y=1.02)
    save_dir='./'
    plt.savefig(
        os.path.join(save_dir, 'paired_tsne_2d_gender_colored_lines.png'),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.5
    )
    plt.close()


logger = get_logger(__name__, log_level="INFO")

if __name__ == '__main__':



    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name,
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.float16

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path,
                                                         add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    print(config.model.showo.load_from_showo)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type)
    model.eval()

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)

    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    if not (config.mmu_image_path.endswith('.jpg') or config.mmu_image_path.endswith('.png')):
        file_list = [os.path.join(config.mmu_image_path, fn) for fn in os.listdir(config.mmu_image_path)]
    else:
        file_list = [config.mmu_image_path]

    config.question = config.question.split(' *** ')

    sys_prompt_ids = text_tokenizer("system\nYou are a helpful assistant.<|im_end|>",
                                    add_special_tokens=False)['input_ids']
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

    z_images=[]
    z_texts=[]

    for step, image_path in enumerate(tqdm(file_list)):
        print(image_path)
        image_ori = Image.open(image_path).convert("RGB")
        # not center cropping
        # image = resize_and_pad_image(image, target_resolution=(config.dataset.preprocessing.resolution,
        #                                                        config.dataset.preprocessing.resolution))
        # image = to_tensor_and_normalize(image)
        # center crop
        image = image_transform(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
        image = image.unsqueeze(0)

        #image_latents, features = vae_model.sample(image.unsqueeze(2)).squeeze(2).to(weight_type)
        image_latents, features = vae_model.sample(image.unsqueeze(2))
        image_latents=image_latents.squeeze(2).to(weight_type)
        image_embeds_und = model.image_embedder_und(image_latents)
        image_embeds_gen = model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
        image_embeds_und = model.und_trans(image_embeds_und)['last_hidden_state']
        image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        batch_size = 1
        responses = ['' for j in range(len(file_list))]
        images = [image]
        for j, question in enumerate(config.question):
            input_ids = text_tokenizer(question, add_special_tokens=False).input_ids
            text_tokens_a = torch.tensor([showo_token_ids['bos_id']] + sys_prompt_ids + role_a).to(device)[None, :]
            text_tokens_b = torch.tensor([showo_token_ids['boi_id'], showo_token_ids['eoi_id']] + input_ids + role_b).to(device)[None, :]
            text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
            text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)




            if config.model.showo.add_time_embeds:
                time_embeds = model.time_embed(torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype)
                if hasattr(model, 'time_embed_proj'):
                    time_embeds = model.time_embed_proj(time_embeds)
                input_embeds = torch.cat([
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    time_embeds,
                    image_embeds,
                    text_embeds_b[:, 1:]
                ], dim=1).to(weight_type)
                modality_positions = torch.tensor([text_tokens_a.shape[1] + 2, num_mmu_image_tokens])[None, None, :].to(device)
            else:
                input_embeds = torch.cat([
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    image_embeds,
                    text_embeds_b[:, 1:]
                ], dim=1).to(weight_type)
                modality_positions = torch.tensor([text_tokens_a.shape[1] + 1, num_mmu_image_tokens])[None, None, :].to(device)

            attention_mask = omni_attn_mask_naive(
                B=input_embeds.size(0),
                LEN=input_embeds.size(1),
                modalities=modality_positions,
                device=device, inverted=True
            ).to(input_embeds.dtype)

            output_tokens = model.mmu_generate(input_embeds=input_embeds,
                                                     attention_mask=attention_mask,
                                                     top_k=top_k,
                                                     max_new_tokens=300,
                                                     eos_token=text_tokenizer.eos_token_id)

            output_tokens = torch.stack(output_tokens).squeeze()[None]

        text = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        responses[j] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'

        z_texts.append(output_tokens.squeeze().detach().cpu().numpy())
        z_images.append(features.squeeze().flatten().detach().cpu().numpy())
        print('yo')

        #
        # images = torch.cat(images, dim=0)
        # images = denorm(images)
        # pil_images = [Image.fromarray(image) for image in images]
        #
        # wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
        # wandb.log({"Multimodal understanding responses": wandb_images}, step=step)



    all_feats = z_images

    import numpy as np


    # number of unique embeddings
    print("unique rows:", np.unique(all_feats, axis=0).shape[0])

    # compare with first image
    for i in range(1, len(all_feats)):
        diff = np.abs(all_feats[0] - all_feats[i])

        print(f"\nImage 0 vs Image {i}")
        print("max diff :", diff.max())
        print("mean diff:", diff.mean())
        print("all equal:", np.allclose(all_feats[0], all_feats[i]))



    z_images = np.array(z_images)

    #pad
    # max_len = max(len(x) for x in z_texts)
    # z_texts = np.array([
    #     np.pad(x, (0, max_len - len(x)))
    #     for x in z_texts
    # ])

    #pooling
    import torch
    import torch.nn as nn
    import numpy as np


    target_len = min(len(x) for x in z_texts)

    pool = nn.AdaptiveAvgPool1d(target_len)

    processed = []

    for arr in z_texts:
        x = torch.tensor(arr, dtype=torch.float32)

        # shape: (N,C,L) or (C,L)
        x = x.unsqueeze(0).unsqueeze(0)

        x = pool(x)

        x = x.squeeze().numpy()

        processed.append(x)

    z_texts = np.stack(processed)

    print(z_texts.shape)
    print(z_images.shape)

    tsne_calc(z_images, z_texts)

