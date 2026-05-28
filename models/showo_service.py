import gc
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from models.Showo.models import Showo, MAGVITv2, CLIPVisionTower
from models.Showo.training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu_vit
from models.Showo.training.utils import image_transform
from models.Showo.llava.llava import conversation as conversation_lib


class ShowoService:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        config = self.config

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.showo.llm_model_path,
            padding_side="left"
        )

        uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob
        )

        vq_model = MAGVITv2.from_pretrained(
            config.model.vq_model.vq_model_name
        ).to(self.device)
        vq_model.eval()

        vision_tower_name = "openai/clip-vit-large-patch14-336"
        vision_tower = CLIPVisionTower(vision_tower_name).to(self.device)
        vision_tower.eval()

        clip_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        model = Showo.from_pretrained(
            config.model.showo.pretrained_model_path
        ).to(self.device)
        model.eval()

        return model, tokenizer, uni_prompting, vq_model, vision_tower, clip_processor

    def _unload_model(self, *objects):
        for obj in objects:
            del obj

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def image_to_text(self, image: Image.Image, question: str) -> str:
        model = tokenizer = uni_prompting = vq_model = vision_tower = clip_processor = None

        try:
            model, tokenizer, uni_prompting, vq_model, vision_tower, clip_processor = self._load_model()

            device = self.device
            image = image.convert("RGB")

            pixel_values = clip_processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"].to(device)

            conv = conversation_lib.conv_templates["phi1.5"].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt().strip()

            text_tokenizer = uni_prompting.text_tokenizer

            input_ids = text_tokenizer(
                prompt,
                return_tensors="pt"
            ).input_ids.to(device)

            SYSTEM_PROMPT = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
            SYSTEM_PROMPT_LEN = 28

            input_ids_system = text_tokenizer(
                SYSTEM_PROMPT,
                return_tensors="pt",
                padding="longest"
            ).input_ids.to(device)

            assert input_ids_system.shape[-1] == SYSTEM_PROMPT_LEN

            mmu_id = int(uni_prompting.sptids_dict["<|mmu|>"])
            soi_id = int(uni_prompting.sptids_dict["<|soi|>"])
            eoi_id = int(uni_prompting.sptids_dict["<|eoi|>"])

            mmu_tok = torch.full((1, 1), mmu_id, dtype=torch.long, device=device)
            soi_tok = torch.full((1, 1), soi_id, dtype=torch.long, device=device)
            eoi_tok = torch.full((1, 1), eoi_id, dtype=torch.long, device=device)

            input_ids_llava = torch.cat([
                mmu_tok,
                input_ids_system,
                soi_tok,
                eoi_tok,
                input_ids
            ], dim=1)

            with torch.inference_mode():
                text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

                image_embeddings = vision_tower(pixel_values)
                image_embeddings = model.mm_projector(image_embeddings)

                part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
                part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]

                input_embeddings = torch.cat(
                    (part1, image_embeddings, part2),
                    dim=1
                )

                attention_mask = create_attention_mask_for_mmu_vit(
                    input_embeddings,
                    system_prompt_len=SYSTEM_PROMPT_LEN
                )

                attention_mask = attention_mask.to(dtype=text_embeddings.dtype)

                outputs = model.mmu_generate(
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask[0].unsqueeze(0),
                    max_new_tokens=200,
                    top_k=1,
                    eot_token=text_tokenizer.eos_token_id
                )

            outputs = torch.stack(outputs).squeeze()[None]

            text = text_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]

            return text.strip()

        finally:
            self._unload_model(
                model,
                tokenizer,
                uni_prompting,
                vq_model,
                vision_tower,
                clip_processor
            )