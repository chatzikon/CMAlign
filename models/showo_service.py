import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from models.Showo.models import Showo, MAGVITv2, CLIPVisionTower
from models.Showo.training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu_vit
from models.Showo.training.utils import image_transform

from models.Showo.llava.llava import conversation as conversation_lib
from models.Showo.training.prompting_utils import create_attention_mask_for_mmu_vit


class ShowoService:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.showo.llm_model_path,
            padding_side="left"
        )

        # prompting
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob
        )

        # VQ model
        self.vq_model = MAGVITv2.from_pretrained(
            config.model.vq_model.vq_model_name
        ).to(self.device)
        self.vq_model.eval()

        # vision tower
        vision_tower_name = "openai/clip-vit-large-patch14-336"
        self.vision_tower = CLIPVisionTower(vision_tower_name).to(self.device)
        self.clip_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)




        # main model
        self.model = Showo.from_pretrained(
            config.model.showo.pretrained_model_path
        ).to(self.device)

        self.model.eval()

    def image_to_text(self, image: Image.Image, question: str) -> str:
        import torch

        device = self.device
        model = self.model

        # --- 1. Prepare image ---
        image = image.convert("RGB")

        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device)

        # --- 2. Build conversation prompt (IMPORTANT) ---

        conv = conversation_lib.conv_templates["phi1.5"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt().strip()

        # --- 3. Tokenize ---
        tokenizer = self.uni_prompting.text_tokenizer

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        SYSTEM_PROMPT = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

        SYSTEM_PROMPT_LEN = 28  # must match training

        input_ids_system = tokenizer(
            SYSTEM_PROMPT,
            return_tensors="pt",
            padding="longest"
        ).input_ids.to(device)

        assert input_ids_system.shape[-1] == SYSTEM_PROMPT_LEN

        # --- 4. Build LLaVA-style sequence ---
        mmu_id = int(self.uni_prompting.sptids_dict["<|mmu|>"])
        soi_id = int(self.uni_prompting.sptids_dict["<|soi|>"])
        eoi_id = int(self.uni_prompting.sptids_dict["<|eoi|>"])

        mmu_tok = torch.full((1, 1), mmu_id, dtype=torch.long, device=device)
        soi_tok = torch.full((1, 1), soi_id, dtype=torch.long, device=device)
        eoi_tok = torch.full((1, 1), eoi_id, dtype=torch.long, device=device)

        input_ids_llava = torch.cat([
            mmu_tok,
            input_ids_system,
            soi_tok,
            eoi_tok,  # placeholder for image
            input_ids
        ], dim=1)

        # --- 5. Get embeddings ---
        with torch.no_grad():
            text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

            image_embeddings = self.vision_tower(pixel_values)
            image_embeddings = model.mm_projector(image_embeddings)

        # --- 6. Insert image embeddings ---
        part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
        part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]

        input_embeddings = torch.cat(
            (part1, image_embeddings, part2),
            dim=1
        )

        # --- 7. Create correct attention mask (CRITICAL) ---

        attention_mask = create_attention_mask_for_mmu_vit(
            input_embeddings,
            system_prompt_len=SYSTEM_PROMPT_LEN
        )

        # ensure correct dtype
        attention_mask = attention_mask.to(dtype=text_embeddings.dtype)

        # --- 8. Generate ---
        with torch.no_grad():
            outputs = model.mmu_generate(
                input_embeddings=input_embeddings,
                attention_mask=attention_mask[0].unsqueeze(0),
                max_new_tokens=100,
                top_k=1,
                eot_token=tokenizer.eos_token_id
            )

        # --- 9. Decode ---
        outputs = torch.stack(outputs).squeeze()[None]

        text = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return text.strip()