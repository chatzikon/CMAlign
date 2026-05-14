import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from omegaconf import OmegaConf

# Adapt these import paths to where you copied Show-o2 in your project.
# If your Show-o2 folder is under models/Showo/show-o2, you may need to expose it via PYTHONPATH.
from models.Showo.show_o2.models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.Showo.show_o2.models.misc import get_text_tokenizer
from models.Showo.show_o2.utils import get_hyper_params, path_to_llm_name, load_state_dict
from models.Showo.show_o2.datasets.utils import image_transform


class Showo2Service:
    def __init__(self, config_path: str):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        #self.config = OmegaConf.load(config_path)
        self.config = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_type = torch.float32

        if self.config.model.vae_model.type != "wan21":
            raise NotImplementedError(
                f"Unsupported VAE type: {self.config.model.vae_model.type}"
            )

        self.vae_model = WanVAE(
            vae_pth=self.config.model.vae_model.pretrained_model_path,
            dtype=self.weight_type,
            device=self.device,
        )

        llm_path = self.config.model.showo.llm_model_path

        self.text_tokenizer, self.showo_token_ids = get_text_tokenizer(
            llm_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=path_to_llm_name[llm_path],
        )

        self.config.model.showo.llm_vocab_size = len(self.text_tokenizer)

        if self.config.model.showo.load_from_showo:
            self.model = Showo2Qwen2_5.from_pretrained(
                self.config.model.showo.pretrained_model_path,
                use_safetensors=False,
            ).to(self.device)
        else:
            self.model = Showo2Qwen2_5(**self.config.model.showo).to(self.device)
            state_dict = load_state_dict(self.config.model_path)
            self.model.load_state_dict(state_dict)

        self.model.to(self.weight_type)
        self.model.eval()

        if self.config.model.showo.add_time_embeds:
            self.config.dataset.preprocessing.num_t2i_image_tokens += 1
            self.config.dataset.preprocessing.num_mmu_image_tokens += 1
            self.config.dataset.preprocessing.num_video_tokens += 1

        (
            self.num_t2i_image_tokens,
            self.num_mmu_image_tokens,
            self.num_video_tokens,
            self.max_seq_len,
            self.max_text_len,
            self.image_latent_dim,
            self.patch_size,
            self.latent_width,
            self.latent_height,
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.boi_id,
            self.eoi_id,
            self.bov_id,
            self.eov_id,
            self.img_pad_id,
            self.vid_pad_id,
            self.guidance_scale,
        ) = get_hyper_params(
            self.config,
            self.text_tokenizer,
            self.showo_token_ids,
        )

        self.sys_prompt_ids = self.text_tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>",
            add_special_tokens=False,
        )["input_ids"]

        self.role_a = self.text_tokenizer(
            "\n<|im_start|>user\n",
            add_special_tokens=False,
        )["input_ids"]

        self.role_b = self.text_tokenizer(
            "\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )["input_ids"]

    @torch.inference_mode()
    def image_to_text(
        self,
        image: Image.Image,
        question: str = "Please describe this image in detail.",
        max_new_tokens: int = 300,
        top_k: int = 1,
    ) -> str:
        image_ori = image.convert("RGB")

        image_tensor = image_transform(
            image_ori,
            resolution=self.config.dataset.preprocessing.resolution,
        ).to(self.device)

        image_tensor = image_tensor.unsqueeze(0)

        image_latents = self.vae_model.sample(
            image_tensor.unsqueeze(2)
        ).squeeze(2).to(self.weight_type)

        image_embeds_und = self.model.image_embedder_und(image_latents)
        image_embeds_gen = self.model.image_embedder_gen(image_latents)

        image_embeds_und = image_embeds_und + self.model.position_embedding(
            self.model.image_position_ids
        )

        image_embeds_und = self.model.und_trans(
            image_embeds_und
        )["last_hidden_state"]

        image_embeds = self.model.fusion_proj(
            torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
        )

        input_ids = self.text_tokenizer(
            question,
            add_special_tokens=False,
        ).input_ids

        text_tokens_a = torch.tensor(
            [self.showo_token_ids["bos_id"]] + self.sys_prompt_ids + self.role_a,
            device=self.device,
            dtype=torch.long,
        )[None, :]

        text_tokens_b = torch.tensor(
            [self.showo_token_ids["boi_id"], self.showo_token_ids["eoi_id"]]
            + input_ids
            + self.role_b,
            device=self.device,
            dtype=torch.long,
        )[None, :]

        text_embeds_a = self.model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self.model.showo.model.embed_tokens(text_tokens_b)

        if self.config.model.showo.add_time_embeds:
            time_embeds = self.model.time_embed(
                torch.tensor([[1.0]], device=self.device),
                text_embeds_a.dtype,
            )

            if hasattr(self.model, "time_embed_proj"):
                time_embeds = self.model.time_embed_proj(time_embeds)

            input_embeds = torch.cat(
                [
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    time_embeds,
                    image_embeds,
                    text_embeds_b[:, 1:],
                ],
                dim=1,
            ).to(self.weight_type)

            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 2, self.num_mmu_image_tokens],
                device=self.device,
            )[None, None, :]
        else:
            input_embeds = torch.cat(
                [
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    image_embeds,
                    text_embeds_b[:, 1:],
                ],
                dim=1,
            ).to(self.weight_type)

            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 1, self.num_mmu_image_tokens],
                device=self.device,
            )[None, None, :]

        attention_mask = omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=self.device,
            inverted=True,
        ).to(input_embeds.dtype)

        output_tokens = self.model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            eos_token=self.text_tokenizer.eos_token_id,
        )

        output_tokens = torch.stack(output_tokens).squeeze()[None]

        text = self.text_tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
        )[0]

        return text.strip()