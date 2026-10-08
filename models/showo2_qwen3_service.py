import copy
import gc
import math
import types
from pathlib import Path

import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from models.Showo.show_o2.models import (
    Showo2Qwen2_5,
    WanVAE,
)

from models.Showo.show_o2.datasets.utils import (
    resize_and_pad_image,
    to_tensor_and_normalize,
)

class Showo2ToQwenAdapter(nn.Module):

    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(
            1536,
            2560,
        )

    def forward(self, x):
        return self.proj(x)


def spatial_align_showo_to_qwen(
    showo_features,
    qwen_features,
):

    batch_size, num_showo_tokens, dim = (
        showo_features.shape
    )

    _, num_qwen_tokens, _ = (
        qwen_features.shape
    )

    showo_side = int(
        math.sqrt(num_showo_tokens)
    )

    qwen_side = int(
        math.sqrt(num_qwen_tokens)
    )

    if showo_side * showo_side != num_showo_tokens:
        raise ValueError(
            f"Show-o token count "
            f"{num_showo_tokens} is not square."
        )

    if qwen_side * qwen_side != num_qwen_tokens:
        raise ValueError(
            f"Qwen token count "
            f"{num_qwen_tokens} is not square."
        )

    x = showo_features.transpose(
        1,
        2,
    )

    x = x.reshape(
        batch_size,
        dim,
        showo_side,
        showo_side,
    )

    x = F.adaptive_avg_pool2d(
        x,
        (
            qwen_side,
            qwen_side,
        ),
    )

    x = (
        x
        .flatten(2)
        .transpose(1, 2)
    )

    return x


def get_showo2_features(
    image,
    model,
    vae,
):

    image = resize_and_pad_image(
        image,
        (432, 432),
    )

    image_tensor = (
        to_tensor_and_normalize(
            image,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(
            device=next(
                model.parameters()
            ).device,
            dtype=next(
                model.parameters()
            ).dtype,
        )
    )

    # WanVAE expects:
    #
    # B, C, T, H, W
    #
    # We have a single image,
    # therefore T = 1.
    image_tensor = image_tensor.unsqueeze(2)

    with torch.no_grad():

        latent = vae.sample(
            image_tensor,
            deterministic=True,
        )

        if (
            latent.ndim == 5
            and latent.shape[2] == 1
        ):
            latent = latent.squeeze(2)

        visual_dtype = (
            model
            .image_embedder_und
            .proj
            .weight
            .dtype
        )

        visual_device = (
            model
            .image_embedder_und
            .proj
            .weight
            .device
        )

        latent = latent.to(
            device=visual_device,
            dtype=visual_dtype,
        )

        image_embeds_und = (
            model.image_embedder_und(
                latent
            )
        )

        image_embeds_gen = (
            model.image_embedder_gen(
                latent
            )
        )

        image_embeds_und = (
            image_embeds_und
            +
            model.position_embedding(
                model.image_position_ids
            )
        )

        image_embeds_und = (
            model.und_trans(
                image_embeds_und
            )["last_hidden_state"]
        )

        fused = model.fusion_proj(
            torch.cat(
                [
                    image_embeds_und,
                    image_embeds_gen,
                ],
                dim=-1,
            )
        )

    return fused

class Showo2Qwen3Service:

    def __init__(
        self,
        wan_vae_path,
        stage2_checkpoint,
        showo_model="showlab/show-o2-1.5B",
        qwen_model="Qwen/Qwen3-VL-4B-Instruct"
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.dtype = torch.bfloat16

        self.wan_vae_path = Path(
            wan_vae_path
        )

        self.stage2_checkpoint = Path(
            stage2_checkpoint
        )

        self.showo_model = showo_model
        self.qwen_model = qwen_model

        self._inference_lock = threading.Lock()

        # -----------------------------------------
        # Verify required local files exist
        # -----------------------------------------

        if not self.wan_vae_path.exists():
            raise FileNotFoundError(
                f"Wan VAE checkpoint not found: "
                f"{self.wan_vae_path}"
            )

        if not self.stage2_checkpoint.exists():
            raise FileNotFoundError(
                f"Stage-2 checkpoint not found: "
                f"{self.stage2_checkpoint}"
            )

        print(
            "Loading Stage-2 checkpoint:",
            self.stage2_checkpoint,
        )

        checkpoint = torch.load(
            self.stage2_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        # -----------------------------------------
        # Store trained weights
        # -----------------------------------------

        self.fusion_state = (
            checkpoint["fusion_proj"]
        )

        self.adapter_state = (
            checkpoint["adapter"]
        )

        print(
            "Stage:",
            checkpoint.get("stage"),
        )

        print(
            "Epoch:",
            checkpoint.get("epoch"),
        )

        print(
            "Validation loss:",
            checkpoint.get("val_loss"),
        )

        # -----------------------------------------
        # Build adapter
        # -----------------------------------------

        self.adapter = (
            Showo2ToQwenAdapter()
        )

        self.adapter.load_state_dict(
            self.adapter_state
        )

        self.adapter.eval()

        print(
            "Adapter loaded successfully."
        )

        print("\nPreloading resident models into CPU memory...")

        # --------------------------------------------------
        # Qwen processor
        # --------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(
            self.qwen_model
        )

        print("Qwen3 processor loaded.")

        # --------------------------------------------------
        # Show-o2 visual model
        #
        # Load it ONCE and leave it on CPU.
        # --------------------------------------------------

        self.showo_visual = (
            Showo2Qwen2_5.from_pretrained(
                self.showo_model,
                use_safetensors=False,
                load_llm=False,
                load_siglip_pretrained=False,
            )
        )

        self.showo_visual.fusion_proj.load_state_dict(
            self.fusion_state
        )

        self.showo_visual = self.showo_visual.to(
            device="cpu",
            dtype=self.dtype,
        )

        self.showo_visual.eval()

        for parameter in self.showo_visual.parameters():
            parameter.requires_grad = False

        print("Show-o2 visual model resident on CPU.")

        # --------------------------------------------------
        # Wan VAE
        #
        # Also load ONCE on CPU.
        # --------------------------------------------------

        self.vae = WanVAE(
            vae_pth=str(self.wan_vae_path),
            dtype=self.dtype,
            device="cpu",
        )

        print("Wan VAE resident on CPU.")

        # --------------------------------------------------
        # Qwen3-VL
        #
        # Important: NO .to(self.device) here.
        # It remains in CPU RAM until a request arrives.
        # --------------------------------------------------

        self.qwen = (
            Qwen3VLForConditionalGeneration
            .from_pretrained(
                self.qwen_model,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        )

        self.qwen.eval()

        for parameter in self.qwen.parameters():
            parameter.requires_grad = False

        print("Qwen3-VL resident on CPU.")

        print("\nAll resident models loaded.")

    def _move_vae(
            self,
            device,
    ):

        device = torch.device(device)

        self.vae.model = self.vae.model.to(
            device
        )

        self.vae.mean = self.vae.mean.to(
            device
        )

        self.vae.std = self.vae.std.to(
            device
        )

        self.vae.scale = [
            self.vae.mean,
            1.0 / self.vae.std,
        ]

        self.vae.device = device






    def _activate_showo2(self):

        print(
            "\nMoving Show-o2 + WanVAE "
            "CPU -> GPU..."
        )

        self.showo_visual = (
            self.showo_visual.to(
                device=self.device,
                dtype=self.dtype,
            )
        )

        self._move_vae(
            self.device
        )

        print(
            "Show-o2 + WanVAE are on GPU."
        )

    def _offload_showo2(self):

        print(
            "Moving Show-o2 + WanVAE "
            "GPU -> CPU..."
        )

        self.showo_visual = (
            self.showo_visual.to("cpu")
        )

        self._move_vae("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            "Show-o2 + WanVAE are back on CPU."
        )

    def _activate_qwen(self):

        print(
            "\nMoving Qwen3-VL CPU -> GPU..."
        )

        self.qwen = self.qwen.to(
            self.device
        )

        print(
            "Qwen3-VL is on GPU."
        )

    def _offload_qwen(self):

        print(
            "Moving Qwen3-VL GPU -> CPU..."
        )

        self.qwen = self.qwen.to(
            "cpu"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            "Qwen3-VL is back on CPU."
        )


    def _load_showo2_visual(self):

        self._activate_showo2()

        return (
            self.showo_visual,
            self.vae,
        )

    def extract_showo_features(
            self,
            image: Image.Image,
    ):

        model, vae = (
            self._load_showo2_visual()
        )

        try:

            image = image.convert("RGB")

            print(
                "\nExtracting Show-o2 features..."
            )

            with torch.no_grad():

                features = (
                    get_showo2_features(
                        image,
                        model,
                        vae,
                    )
                )

            print(
                "Show-o2 feature shape:",
                tuple(features.shape),
            )

            features = (
                features
                .detach()
                .float()
                .cpu()
            )

            return features

        finally:

            self._offload_showo2()

            gc.collect()

    def _extract_showo_features_active(
            self,
            image: Image.Image,
    ):

        image = image.convert("RGB")

        with torch.no_grad():
            features = get_showo2_features(
                image,
                self.showo_visual,
                self.vae,
            )

        # Keep the features in bfloat16 on CPU.
        #
        # There is no reason to convert them to
        # float32 for batch inference because they
        # are converted back to bfloat16 for the
        # adapter anyway.
        features = (
            features
            .detach()
            .to(
                device="cpu",
                dtype=self.dtype,
            )
        )

        return features

    def _prepare_qwen_inputs(
        self,
        processor,
        image: Image.Image,
        prompt: str,
    ):

        # This is the exact fixed resolution
        # used during your adapter training.
        qwen_image = resize_and_pad_image(
            image.convert("RGB"),
            (448, 448),
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=[qwen_image],
            return_tensors="pt",

            # We already resized to 448 × 448.
            do_resize=False,
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(
                    self.device
                )

        return inputs

    def _generate_from_showo_features_active(
            self,
            image: Image.Image,
            z_showo_raw,
            question: str,
            alpha: float,
            max_new_tokens: int,
    ):

        beta = 1.0 - alpha

        image = image.convert("RGB")

        qwen = self.qwen
        processor = self.processor

        # -----------------------------------------
        # Qwen input
        # -----------------------------------------

        inputs = self._prepare_qwen_inputs(
            processor,
            image,
            question,
        )

        # -----------------------------------------
        # Native Qwen visual features
        # -----------------------------------------

        with torch.no_grad():

            (
                native_image_embeds,
                native_deepstack,
            ) = qwen.get_image_features(
                pixel_values=
                inputs["pixel_values"],

                image_grid_thw=
                inputs["image_grid_thw"],
            )

        z_qwen = (
            native_image_embeds[0]
            .unsqueeze(0)
        )

        # -----------------------------------------
        # Show-o2 feature -> GPU
        # -----------------------------------------

        z_showo_raw = z_showo_raw.to(
            device=self.device,
            dtype=self.dtype,
        )

        # -----------------------------------------
        # Spatial alignment
        # -----------------------------------------

        z_showo_aligned = (
            spatial_align_showo_to_qwen(
                z_showo_raw,
                z_qwen,
            )
        )

        # -----------------------------------------
        # Trained adapter
        # -----------------------------------------

        with torch.no_grad():

            z_showo_adapted = self.adapter(
                z_showo_aligned
            )

        if (
                z_showo_adapted.shape
                != z_qwen.shape
        ):
            raise RuntimeError(
                "Cannot fuse representations. "
                f"Show-o2="
                f"{tuple(z_showo_adapted.shape)}, "
                f"Qwen="
                f"{tuple(z_qwen.shape)}"
            )

        # -----------------------------------------
        # Fusion
        # -----------------------------------------

        fused = (
                alpha * z_qwen
                +
                beta * z_showo_adapted
        )

        fused = fused.to(
            device=z_qwen.device,
            dtype=z_qwen.dtype,
        )

        # -----------------------------------------
        # Temporarily replace Qwen's final
        # visual representation
        # -----------------------------------------

        original_get_image_features = (
            qwen.model.get_image_features
        )

        def fused_get_image_features(
                model_self,
                pixel_values,
                image_grid_thw=None,
                **kwargs,
        ):

            return (
                (fused[0],),
                native_deepstack,
            )

        qwen.model.get_image_features = (
            types.MethodType(
                fused_get_image_features,
                qwen.model,
            )
        )

        qwen.model.rope_deltas = None

        try:

            with torch.no_grad():

                generated_ids = qwen.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            generated_only = generated_ids[
                :,
                inputs["input_ids"].shape[1]:
            ]

            output_text = (
                processor.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            )

        finally:

            qwen.model.get_image_features = (
                original_get_image_features
            )

            qwen.model.rope_deltas = None

        # Delete per-image GPU tensors.
        #
        # We deliberately DON'T call
        # torch.cuda.empty_cache() here.
        #
        # PyTorch can reuse this memory for the
        # next image, which is more efficient.

        del inputs
        del native_image_embeds
        del native_deepstack
        del z_qwen
        del z_showo_raw
        del z_showo_aligned
        del z_showo_adapted
        del fused
        del generated_ids
        del generated_only

        return output_text.strip()

    def extract_qwen_features(
        self,
        image: Image.Image,
        prompt: str = "Describe this image.",
    ):

        print(
            "\nLoading Qwen3-VL..."
        )

        processor = AutoProcessor.from_pretrained(
            self.qwen_model
        )

        qwen = (
            Qwen3VLForConditionalGeneration
            .from_pretrained(
                self.qwen_model,

                dtype=self.dtype,

                low_cpu_mem_usage=True,

                attn_implementation="sdpa",
            )
            .to(self.device)
        )

        qwen.eval()

        print(
            "Qwen3-VL loaded."
        )

        inputs = self._prepare_qwen_inputs(
            processor,
            image,
            prompt,
        )

        print(
            "Extracting Qwen3 visual features..."
        )

        with torch.no_grad():
            native_image_embeds, native_deepstack = (
                qwen.get_image_features(
                    pixel_values=
                    inputs["pixel_values"],

                    image_grid_thw=
                    inputs["image_grid_thw"],
                )
            )

        # One image only.
        # native_image_embeds is a tuple/list containing
        # the visual embeddings for each image.
        qwen_features = (
            native_image_embeds[0]
            .unsqueeze(0)
        )

        print(
            "Qwen3 feature shape:",
            tuple(
                qwen_features.shape
            ),
        )

        features_cpu = (
            qwen_features
            .detach()
            .float()
            .cpu()
        )

        del qwen_features
        del native_image_embeds
        del native_deepstack
        del inputs
        del qwen

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            "Qwen3-VL removed from GPU."
        )

        return features_cpu

    def images_to_text(
            self,
            image_paths,
            question: str,
            alpha: float = 0.5,
            max_new_tokens: int = 384,
            chunk_size: int = 128,
    ):

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0.0 and 1.0"
            )

        if chunk_size < 1:
            raise ValueError(
                "chunk_size must be >= 1"
            )

        with self._inference_lock:

            return self._images_to_text_impl(
                image_paths=image_paths,
                question=question,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
            )

    def _images_to_text_impl(
            self,
            image_paths,
            question: str,
            alpha: float,
            max_new_tokens: int,
            chunk_size: int,
    ):

        image_paths = [
            Path(path)
            for path in image_paths
        ]

        results = [
            None
            for _ in image_paths
        ]

        total = len(image_paths)

        # =========================================
        # Process one chunk at a time
        # =========================================

        for chunk_start in range(
                0,
                total,
                chunk_size,
        ):

            chunk_end = min(
                chunk_start + chunk_size,
                total,
            )

            chunk_paths = image_paths[
                chunk_start:chunk_end
            ]

            print(
                f"\nProcessing images "
                f"{chunk_start + 1}-{chunk_end} "
                f"of {total}"
            )

            chunk_features = [
                None
                for _ in chunk_paths
            ]

            # =====================================
            # PHASE 1:
            # SHOW-O2
            #
            # Move model ONCE for entire chunk.
            # =====================================

            self._activate_showo2()

            try:

                for local_index, path in enumerate(
                        chunk_paths
                ):

                    global_index = (
                            chunk_start
                            + local_index
                    )

                    try:

                        with Image.open(path) as image:

                            features = (
                                self
                                ._extract_showo_features_active(
                                    image
                                )
                            )

                        chunk_features[
                            local_index
                        ] = features

                    except Exception as exc:

                        results[
                            global_index
                        ] = {
                            "status": "error",
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        }

            finally:

                self._offload_showo2()

                gc.collect()

            # =====================================
            # PHASE 2:
            # QWEN
            #
            # Only activate Qwen if at least one
            # image survived Show-o2.
            # =====================================

            if any(
                    feature is not None
                    for feature in chunk_features
            ):

                self._activate_qwen()

                self.adapter = self.adapter.to(
                    device=self.device,
                    dtype=self.dtype,
                )

                try:

                    for local_index, (
                            path,
                            showo_features,
                    ) in enumerate(
                        zip(
                            chunk_paths,
                            chunk_features,
                        )
                    ):

                        if showo_features is None:
                            continue

                        global_index = (
                                chunk_start
                                + local_index
                        )

                        try:

                            with Image.open(path) as image:

                                caption = (
                                    self
                                    ._generate_from_showo_features_active(
                                        image=image,
                                        z_showo_raw=
                                        showo_features,
                                        question=question,
                                        alpha=alpha,
                                        max_new_tokens=
                                        max_new_tokens,
                                    )
                                )

                            results[
                                global_index
                            ] = {
                                "status": "success",
                                "caption": caption,
                            }

                        except Exception as exc:

                            results[
                                global_index
                            ] = {
                                "status": "error",
                                "error": (
                                    f"{type(exc).__name__}: "
                                    f"{exc}"
                                ),
                            }

                        finally:

                            # Release this image's stored
                            # Show-o2 representation as soon
                            # as Qwen has finished with it.

                            chunk_features[
                                local_index
                            ] = None

                finally:

                    # Adapter is small, but leave the GPU
                    # in the same clean state as the rest
                    # of the service.

                    self.adapter = (
                        self.adapter.to("cpu")
                    )

                    self._offload_qwen()

                    gc.collect()

            del chunk_features

            gc.collect()

        return results

    def compare_visual_representations(
        self,
        image: Image.Image,
        prompt: str = "Describe this image.",
    ):

        # --------------------------------------------------
        # 1. Extract raw Show-o2 representation
        # --------------------------------------------------

        z_showo_raw = self.extract_showo_features(
            image
        )

        print(
            "\nRaw Show-o2 shape:",
            tuple(z_showo_raw.shape),
        )

        # --------------------------------------------------
        # 2. Extract native Qwen3 representation
        # --------------------------------------------------

        z_qwen = self.extract_qwen_features(
            image,
            prompt=prompt,
        )

        print(
            "Native Qwen3 shape:",
            tuple(z_qwen.shape),
        )

        # --------------------------------------------------
        # 3. Match the spatial/token dimensions
        #
        #    729 tokens -> 196 tokens
        # --------------------------------------------------

        z_showo_aligned = (
            spatial_align_showo_to_qwen(
                z_showo_raw,
                z_qwen,
            )
        )

        print(
            "Spatially aligned Show-o2 shape:",
            tuple(z_showo_aligned.shape),
        )

        # --------------------------------------------------
        # 4. Apply trained 1536 -> 2560 adapter
        # --------------------------------------------------

        self.adapter.eval()

        with torch.no_grad():

            z_showo_adapted = self.adapter(
                z_showo_aligned
            )

        print(
            "Adapted Show-o2 shape:",
            tuple(z_showo_adapted.shape),
        )

        # --------------------------------------------------
        # 5. Verify exact compatibility
        # --------------------------------------------------

        if z_showo_adapted.shape != z_qwen.shape:
            raise RuntimeError(
                "Feature shape mismatch: "
                f"Show-o2={tuple(z_showo_adapted.shape)}, "
                f"Qwen={tuple(z_qwen.shape)}"
            )

        # --------------------------------------------------
        # 6. Some useful sanity checks
        # --------------------------------------------------

        qwen_norm = (
            z_qwen
            .norm(dim=-1)
            .mean()
            .item()
        )

        showo_norm = (
            z_showo_adapted
            .norm(dim=-1)
            .mean()
            .item()
        )

        mse = F.mse_loss(
            z_showo_adapted,
            z_qwen,
        ).item()

        cosine = (
            F.cosine_similarity(
                z_showo_adapted,
                z_qwen,
                dim=-1,
            )
            .mean()
            .item()
        )

        print("\n===================================")
        print("VISUAL REPRESENTATION CHECK")
        print("===================================")

        print(
            f"Qwen mean token norm: "
            f"{qwen_norm:.6f}"
        )

        print(
            f"Show-o2 mean token norm: "
            f"{showo_norm:.6f}"
        )

        print(
            f"Show-o2 / Qwen norm ratio: "
            f"{showo_norm / qwen_norm:.6f}"
        )

        print(
            f"MSE: {mse:.6f}"
        )

        print(
            f"Mean cosine similarity: "
            f"{cosine:.6f}"
        )

        return (
            z_showo_adapted,
            z_qwen,
        )

    def _load_adapter_for_inference(self):

        self.adapter = self.adapter.to(
            device=self.device,
            dtype=self.dtype,
        )

        return self.adapter

    def image_to_text(
            self,
            image: Image.Image,
            question: str,
            alpha: float = 0.5,
            max_new_tokens: int = 384,
    ):

        with self._inference_lock:
            return self._image_to_text_impl(
                image=image,
                question=question,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
            )


    def _image_to_text_impl(
        self,
        image: Image.Image,
        question: str,
        alpha: float = 0.5,
        max_new_tokens: int = 384,
    ):

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0.0 and 1.0"
            )

        beta = 1.0 - alpha

        image = image.convert("RGB")

        print("\n========================================")
        print("STEP 1: SHOW-O2 FEATURES")
        print("========================================")

        # --------------------------------------------------
        # 1. Extract Show-o2 representation.
        #
        # This method already:
        #   - loads Show-o2 + WanVAE
        #   - extracts [1,729,1536]
        #   - moves result to CPU
        #   - removes Show-o2 + VAE from GPU
        # --------------------------------------------------

        z_showo_raw = self.extract_showo_features(
            image
        )

        print(
            "Raw Show-o2:",
            tuple(z_showo_raw.shape),
        )

        # --------------------------------------------------
        # 2. Load Qwen3-VL
        # --------------------------------------------------

        print("\n========================================")
        print("STEP 2: LOAD QWEN3-VL")
        print("========================================")

        self._activate_qwen()

        qwen = self.qwen
        processor = self.processor

        qwen.eval()

        for parameter in qwen.parameters():
            parameter.requires_grad = False

        print("Qwen3-VL loaded.")

        # --------------------------------------------------
        # 3. Prepare image + textual prompt for Qwen
        # --------------------------------------------------

        inputs = self._prepare_qwen_inputs(
            processor,
            image,
            question,
        )

        # --------------------------------------------------
        # 4. Extract native Qwen visual representation
        # --------------------------------------------------

        print("\n========================================")
        print("STEP 3: QWEN VISUAL FEATURES")
        print("========================================")

        with torch.no_grad():

            (
                native_image_embeds,
                native_deepstack,
            ) = qwen.get_image_features(
                pixel_values=
                    inputs["pixel_values"],

                image_grid_thw=
                    inputs["image_grid_thw"],
            )

        # One image.
        z_qwen = (
            native_image_embeds[0]
            .unsqueeze(0)
        )

        print(
            "Native Qwen:",
            tuple(z_qwen.shape),
        )

        # --------------------------------------------------
        # 5. Move Show-o2 representation back to GPU
        # --------------------------------------------------

        z_showo_raw = z_showo_raw.to(
            device=self.device,
            dtype=self.dtype,
        )

        # --------------------------------------------------
        # 6. Spatial alignment
        #
        # [1,729,1536]
        #       ↓
        # [1,196,1536]
        # --------------------------------------------------

        z_showo_aligned = (
            spatial_align_showo_to_qwen(
                z_showo_raw,
                z_qwen,
            )
        )

        print(
            "Aligned Show-o2:",
            tuple(z_showo_aligned.shape),
        )

        # --------------------------------------------------
        # 7. Trained Stage-2 adapter
        #
        # [1,196,1536]
        #       ↓
        # [1,196,2560]
        # --------------------------------------------------

        adapter = (
            self._load_adapter_for_inference()
        )

        with torch.no_grad():

            z_showo_adapted = adapter(
                z_showo_aligned
            )

        print(
            "Adapted Show-o2:",
            tuple(z_showo_adapted.shape),
        )

        if z_showo_adapted.shape != z_qwen.shape:
            raise RuntimeError(
                "Cannot fuse representations. "
                f"Show-o2={tuple(z_showo_adapted.shape)}, "
                f"Qwen={tuple(z_qwen.shape)}"
            )

        # --------------------------------------------------
        # 8. Fusion
        #
        # alpha = Qwen contribution
        # beta  = Show-o2 contribution
        # --------------------------------------------------

        fused = (
            alpha * z_qwen
            +
            beta * z_showo_adapted
        )

        fused = fused.to(
            device=z_qwen.device,
            dtype=z_qwen.dtype,
        )

        print("\n========================================")
        print("STEP 4: FUSION")
        print("========================================")

        print(
            f"alpha Qwen3  = {alpha:.2f}"
        )

        print(
            f"beta Show-o2 = {beta:.2f}"
        )

        print(
            "Fused shape:",
            tuple(fused.shape),
        )

        # --------------------------------------------------
        # 9. Replace only Qwen's FINAL visual embeddings.
        #
        # Keep native DeepStack features unchanged.
        # --------------------------------------------------

        original_get_image_features = (
            qwen.model.get_image_features
        )

        def fused_get_image_features(
            model_self,
            pixel_values,
            image_grid_thw=None,
            **kwargs,
        ):

            fused_image_embeds = (
                fused[0],
            )

            return (
                fused_image_embeds,
                native_deepstack,
            )

        qwen.model.get_image_features = (
            types.MethodType(
                fused_get_image_features,
                qwen.model,
            )
        )

        # Qwen caches rotary-position information.
        # Reset it before using our replacement
        # visual representation.
        qwen.model.rope_deltas = None

        # --------------------------------------------------
        # 10. Generate text
        # --------------------------------------------------

        print("\n========================================")
        print("STEP 5: GENERATION")
        print("========================================")

        try:

            with torch.no_grad():

                generated_ids = qwen.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            # Remove the original input/prompt tokens.
            generated_only = generated_ids[
                :,
                inputs["input_ids"].shape[1]:
            ]

            output_text = (
                processor.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            )

        finally:

            # Very important:
            # restore normal Qwen behavior even if
            # generation raises an exception.
            qwen.model.get_image_features = (
                original_get_image_features
            )

            qwen.model.rope_deltas = None

            self._offload_qwen()

        print("\nGenerated text:")
        print(output_text)

        # --------------------------------------------------
        # 11. Cleanup
        # --------------------------------------------------

        del adapter
        del z_showo_raw
        del z_showo_aligned
        del z_showo_adapted
        del z_qwen
        del fused
        del native_image_embeds
        del native_deepstack
        del inputs


        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_text.strip()