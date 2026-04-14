# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only ARC-Hunyuan-Video-7B model compatible with HuggingFace weights.

Architecture: ARCHunyuanVideoForConditionalGeneration
- Vision encoder: HunYuanVisionTransformer (27-layer ViT)
- Audio encoder: Whisper-large-v3 encoder + MLP projection
- LLM backbone: HunYuanDenseV1ForCausalLM (32-layer dense transformer)
- Mixing: Vision + audio embeddings are added element-wise per frame
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import sinusoids

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.arc_hunyuan_video import (
    ARCHunyuanVideoAudioConfig,
    ARCHunyuanVideoConfig,
)
from vllm.transformers_utils.processors.arc_hunyuan_video import (
    ARCHunyuanVideoProcessor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .hunyuan_vision import HunYuanVisionTransformer
from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)

# === Input Schemas === #


class ARCHunyuanVideoPixelInputs(TensorSchema):
    """Video frame pixel inputs.

    Dimensions:
        - np: Number of total patches across all frames
        - cps: Channels * patch_size * patch_size
        - nf: Number of frames
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "cps"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nf", 3),
    ]


class ARCHunyuanVideoEmbeddingInputs(TensorSchema):
    """Pre-computed video embedding inputs.

    Dimensions:
        - nf: Number of features
        - hs: Hidden size
        - ni: Number of items
    """

    type: Literal["image_embeds"]

    image_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


ARCHunyuanVideoInputs: TypeAlias = (
    ARCHunyuanVideoPixelInputs | ARCHunyuanVideoEmbeddingInputs
)


# === Audio Encoder === #


class ARCHunyuanVideoSpeechEncoderAttention(nn.Module):
    """Whisper-style encoder self-attention for the audio encoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = MMEncoderAttention(
            num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=num_heads,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)

        is_2d = q.dim() == 2
        if is_2d:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        out = self.attn(q, k, v)

        if is_2d:
            out = out.squeeze(0)

        output, _ = self.out_proj(out)
        return output


class ARCHunyuanVideoSpeechMLP(nn.Module):
    """Whisper-style MLP for the audio encoder."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str = "gelu",
        prefix: str = "",
    ):
        super().__init__()
        self.activation_fn = get_act_fn(act_fn)
        self.fc1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class ARCHunyuanVideoSpeechEncoderLayer(nn.Module):
    """Single Whisper encoder layer for the audio encoder."""

    def __init__(
        self,
        config: ARCHunyuanVideoAudioConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = ARCHunyuanVideoSpeechEncoderAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.mlp = ARCHunyuanVideoSpeechMLP(
            embed_dim=config.d_model,
            ffn_dim=config.encoder_ffn_dim,
            act_fn=config.activation_function,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ARCHunyuanVideoSpeechEncoder(nn.Module):
    """Whisper-based speech encoder for ARC-Hunyuan-Video.

    Encodes audio mel spectrograms into hidden representations.
    Structure: conv1 → conv2 → positional embedding → N transformer layers → layer_norm
    """

    def __init__(
        self,
        config: ARCHunyuanVideoAudioConfig,
        prefix: str = "",
    ):
        super().__init__()
        embed_dim = config.d_model

        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, stride=2, kernel_size=3, padding=1)

        self.layers = nn.ModuleList(
            [
                ARCHunyuanVideoSpeechEncoderLayer(config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        with torch.no_grad():
            self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
            self.embed_positions.weight.copy_(
                sinusoids(*self.embed_positions.weight.shape)
            )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode audio features.

        Args:
            input_features: Mel spectrogram features of shape
                [num_segments, num_mel_bins, time_steps].

        Returns:
            Hidden states of shape [num_segments, seq_len, d_model].
        """
        # Apply convolutions
        hidden_states = nn.functional.gelu(self.conv1(input_features))
        hidden_states = nn.functional.gelu(self.conv2(hidden_states))

        # [batch, d_model, time] -> [batch, time, d_model]
        hidden_states = hidden_states.transpose(-1, -2)

        # Add positional embeddings
        hidden_states = (
            hidden_states + self.embed_positions.weight[: hidden_states.size(-2), :]
        ).to(hidden_states.dtype)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class ARCHunyuanVideoAudioProjection(nn.Module):
    """MLP projection from audio encoder output to LLM hidden size.

    Structure: LayerNorm → Linear → GELU → Linear
    Weight mapping: mlp2.0=LayerNorm, mlp2.1=Linear1, mlp2.3=Linear2
    (mlp2.2=GELU has no parameters)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Use nn.Sequential with same indices as original for weight compat
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# === Multimodal Processing === #


def _arc_hunyuan_video_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
):
    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)
    num_frames = image_grid_thw.shape[0]
    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes("video", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes("video", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        input_features=MultiModalFieldConfig.shared("video", num_frames),
        audio_duration=MultiModalFieldConfig.shared(
            "video", num_frames, keep_on_cpu=True
        ),
    )


class ARCHunyuanVideoMultiModalDataParser(MultiModalDataParser):
    def _parse_video_data(
        self,
        data,
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_arc_hunyuan_video_field_config,
            )
        return super()._parse_video_data(data)


class ARCHunyuanVideoProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(ARCHunyuanVideoConfig)

    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> ARCHunyuanVideoProcessor:
        hf_config = self.get_hf_config()
        tokenizer = self.ctx.tokenizer

        # Load WhisperFeatureExtractor from the model path
        try:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.ctx.model_config.model
            )
        except Exception:
            feature_extractor = WhisperFeatureExtractor()

        return ARCHunyuanVideoProcessor(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_size=hf_config.vision_config.force_image_size,
            max_num_frame=hf_config.max_num_frame,
        )

    def get_data_parser(self):
        return ARCHunyuanVideoMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config

        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        force_image_size = vision_config.force_image_size

        grid_h = force_image_size // patch_size // spatial_merge_size
        grid_w = force_image_size // patch_size // spatial_merge_size
        tokens_per_frame = grid_h * (grid_w + 1) + 2

        max_frames = hf_config.max_num_frame
        # +2 per frame for im_start/im_end wrapper tokens
        max_video_tokens = (tokens_per_frame + 2) * max_frames

        return {"video": max_video_tokens}

    def get_num_tokens_per_frame(self) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        force_image_size = vision_config.force_image_size
        grid_h = force_image_size // patch_size // spatial_merge_size
        grid_w = force_image_size // patch_size // spatial_merge_size
        return grid_h * (grid_w + 1) + 2


class ARCHunyuanVideoDummyInputsBuilder(
    BaseDummyInputsBuilder[ARCHunyuanVideoProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_videos = mm_counts.get("video", 0)
        # Each video represented by <image> tokens
        return "<image>" * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        hf_config = self.info.get_hf_config()
        image_size = hf_config.vision_config.force_image_size
        num_videos = mm_counts.get("video", 1)

        # Create dummy video frames (1 frame per video)
        frames_per_video = 1
        dummy_frame = np.zeros(
            (frames_per_video, image_size, image_size, 3), dtype=np.uint8
        )

        return {
            "video": [dummy_frame] * num_videos,
        }


class ARCHunyuanVideoMultiModalProcessor(
    BaseMultiModalProcessor[ARCHunyuanVideoProcessingInfo],
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processor = self.info.get_hf_processor(**mm_kwargs)

        # Reference prompt format:
        #   "<|startoftext|>" + video_prefix + "\n" + question + "...<sep>"
        # Add <|startoftext|> prefix if not already present
        hf_config = self.info.get_hf_config()
        bos_token_id = hf_config.text_config.bos_token_id
        tokenizer = processor.tokenizer
        bos_token = tokenizer.convert_ids_to_tokens(bos_token_id)
        if bos_token and not prompt.startswith(bos_token):
            prompt = bos_token + prompt

        # Add <sep> suffix if not already present
        sep_token = "<sep>"
        if not prompt.rstrip().endswith(sep_token):
            prompt = prompt.rstrip() + sep_token

        videos_raw = mm_data.get("videos")

        # Extract video metadata if available (from video loader)
        videos = None
        video_metadata = None
        if videos_raw is not None:
            if not isinstance(videos_raw, list):
                videos_raw = [videos_raw]
            videos = []
            for item in videos_raw:
                if isinstance(item, tuple) and len(item) == 2:
                    video, metadata = item
                    videos.append(video)
                    if metadata is not None:
                        video_metadata = metadata
                else:
                    videos.append(item)

        return processor(
            text=prompt,
            videos=videos,
            video_metadata=video_metadata,
            **tok_kwargs,
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_id
        im_start_id = hf_config.image_start_token_id
        im_end_id = hf_config.image_end_token_id

        tokens_per_frame = self.info.get_num_tokens_per_frame()

        def get_replacement_video(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            # Each frame in the video produces tokens_per_frame tokens
            num_frames = grid_thw.shape[0] if grid_thw.ndim == 2 else 1

            # Reference format: each frame gets its own im_start/im_end block
            # matching the reference's per-frame image handling where each
            # <image> token in the prompt expands to one frame's embeddings.
            full_seq = []
            for _ in range(num_frames):
                full_seq.extend(
                    [im_start_id] + [image_token_id] * tokens_per_frame + [im_end_id]
                )
            return PromptUpdateDetails.select_token_id(full_seq, image_token_id)

        return [
            PromptReplacement(
                modality="video",
                target=[image_token_id],
                replacement=partial(get_replacement_video, modality="video"),
            ),
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _arc_hunyuan_video_field_config(hf_inputs)


# === Main Model === #


@MULTIMODAL_REGISTRY.register_processor(
    ARCHunyuanVideoMultiModalProcessor,
    info=ARCHunyuanVideoProcessingInfo,
    dummy_inputs=ARCHunyuanVideoDummyInputsBuilder,
)
class ARCHunyuanVideoForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
):
    """ARC-Hunyuan-Video-7B model for video understanding.

    Combines a vision encoder (ViT), audio encoder (Whisper), and
    LLM backbone (HunYuan Dense V1) for video+audio understanding.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vision_model.vit.": "visual.",
            "vision_model.perceive.": "visual.perceive.",
            "speech_encoder.": "audio_encoder.",
            "mlp2.": "audio_projection.mlp.",
        }
    )

    supports_encoder_tp_data = True

    def get_xdrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> torch.Tensor:
        """Compute 4D XDRoPE positions for multimodal inputs."""
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw"},
        )
        # Flatten batched grid_thw to per-frame list.
        # gather_kwargs returns one tensor per video item (shape [N, 3] for
        # N frames). We need one [t, h, w] entry per im_start token.
        all_frame_grids: list[list[int]] = []
        for item in kwargs.get("image_grid_thw", []):
            grid = item.tolist() if hasattr(item, "tolist") else item
            if isinstance(grid[0], list):
                all_frame_grids.extend(grid)
            else:
                all_frame_grids.append(grid)
        image_grid_thw = all_frame_grids

        hf_config = self.config
        image_start_token_id = hf_config.image_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        xd_num = len(hf_config.text_config.xdrope_section)

        input_tokens_tensor = torch.tensor(input_tokens)
        image_start_indices = torch.argwhere(
            input_tokens_tensor == image_start_token_id
        ).squeeze(1)

        p_index = torch.arange(len(input_tokens_tensor))
        w_index = torch.arange(len(input_tokens_tensor))
        h_index = torch.arange(len(input_tokens_tensor))
        t_index = torch.arange(len(input_tokens_tensor))

        for image_index in range(len(image_start_indices)):
            pos = image_start_indices[image_index] + 2
            t, h, w = image_grid_thw[image_index]
            _, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )

            token_num = (llm_grid_w + 1) * llm_grid_h
            w_index[pos : pos + token_num].copy_(
                torch.arange(0, llm_grid_w + 1)
                .reshape(1, -1)
                .expand(llm_grid_h, -1)
                .reshape(-1)
            )
            h_index[pos : pos + token_num].copy_(
                torch.arange(0, llm_grid_h)
                .reshape(-1, 1)
                .expand(-1, llm_grid_w + 1)
                .reshape(-1)
            )
            t_index[pos : pos + token_num] = image_index

        if xd_num == 4:
            llm_positions = torch.stack([p_index, w_index, h_index, t_index])
        elif xd_num == 3:
            llm_positions = torch.stack([w_index, h_index, t_index])
        else:
            llm_positions = torch.stack([p_index, w_index, h_index, t_index])

        return llm_positions

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("video") or modality.startswith("image"):
            # Reference: prompt = "<|startoftext|>" + video_prefix + "\n" + question + "\n..." + "<sep>"  # noqa: E501
            # The <image> tokens are the per-frame placeholders.
            # <|startoftext|> and <sep> are handled via chat template / tokenizer.
            return "<image>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: ARCHunyuanVideoConfig = vllm_config.model_config.hf_config
        self.config = config

        # Vision encoder (reuse HunYuanVisionTransformer)
        with self._mark_tower_model(vllm_config, {"video", "image"}):
            self.visual = HunYuanVisionTransformer(
                config.vision_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        # Audio encoder (standalone Whisper encoder)
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = ARCHunyuanVideoSpeechEncoder(
                config.audio_config,
                prefix=maybe_prefix(prefix, "audio_encoder"),
            )

        # Audio projection MLP
        self.audio_projection = ARCHunyuanVideoAudioProjection(
            in_features=config.audio_config.d_model,
            out_features=config.text_config.hidden_size,
        )

        # LLM backbone (reuse HunYuanDenseV1ForCausalLM)
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model.model"),
                architectures=[
                    "HunYuanDenseV1ForCausalLM",
                    "HunYuanMoEV1ForCausalLM",
                ],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> ARCHunyuanVideoInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        if pixel_values is not None:
            if len(pixel_values.shape) == 3:
                last_dim = pixel_values.shape[-1]
                pixel_values = pixel_values.reshape(-1, last_dim)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.reshape(-1, 3)

            return ARCHunyuanVideoPixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return ARCHunyuanVideoEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

        return None

    def _process_vision_input(
        self, video_input: ARCHunyuanVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        """Process video frames through the vision encoder."""
        grid_thw = video_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "image_embeds":
            return (video_input["image_embeds"].type(self.visual.dtype),)

        pixel_values = video_input["pixel_values"]
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)
        return tuple(image_embeds)

    def _process_audio_input(self, **kwargs: object) -> torch.Tensor | None:
        """Process audio features through Whisper encoder + MLP projection."""
        input_features = kwargs.get("input_features")

        if input_features is None:
            return None

        if isinstance(input_features, list):
            input_features = torch.stack(input_features)

        # input_features shape: [num_segments, mel_bins, time_steps]
        # Reshape for whisper encoder: [num_segments, mel_bins, time_steps]
        audio_features = input_features.squeeze(0)
        if audio_features.ndim == 2:
            audio_features = audio_features.unsqueeze(0)

        # Reshape to [num_chunks, mel_bins, time_per_chunk]
        if audio_features.shape[1] != self.audio_encoder.num_mel_bins:
            audio_features = audio_features.reshape(
                -1,
                self.audio_encoder.num_mel_bins,
                audio_features.shape[-1],
            )

        # Run through Whisper encoder
        speech_embeds = self.audio_encoder(audio_features)

        # speech_embeds shape: [num_chunks, seq_len, d_model]
        # Reshape to [1, total_seq_len, d_model]
        speech_embeds = speech_embeds.reshape(1, -1, speech_embeds.shape[-1])

        # Project to LLM hidden size
        speech_embeds = self.audio_projection(speech_embeds)

        return speech_embeds

    def _create_mixed_embeddings(
        self,
        vision_embeds: tuple[torch.Tensor, ...],
        audio_embeds: torch.Tensor | None,
        duration: int | None,
    ) -> tuple[torch.Tensor, ...]:
        """Mix vision and audio embeddings via element-wise addition.

        Each frame's vision embeddings get corresponding audio embeddings
        added to them.
        """
        if audio_embeds is None:
            return vision_embeds

        max_num_frame = self.config.max_num_frame
        num_frames = len(vision_embeds)

        if duration is None:
            duration = num_frames

        # Reshape audio: [1, total_tokens, hidden] -> [duration, 50, hidden]
        # (50 tokens per second from Whisper encoder)
        audio_embeds_reshaped = audio_embeds.reshape(
            audio_embeds.shape[0], -1, 50, audio_embeds.shape[-1]
        )
        audio_no_pad = audio_embeds_reshaped[:, :duration].squeeze(0)

        # Handle long videos
        if duration > max_num_frame:
            per_audio_tokens = math.ceil(audio_no_pad.shape[0] / max_num_frame * 50)
            num_audio_tokens_sum = per_audio_tokens * max_num_frame
            audio_no_pad = audio_no_pad.reshape(-1, audio_no_pad.shape[-1])

            if num_audio_tokens_sum != audio_no_pad.shape[0]:
                zero_padding = torch.zeros(
                    num_audio_tokens_sum - audio_no_pad.shape[0],
                    audio_no_pad.shape[-1],
                    dtype=audio_no_pad.dtype,
                    device=audio_no_pad.device,
                )
                audio_no_pad = torch.cat((audio_no_pad, zero_padding), dim=0)

            audio_no_pad = audio_no_pad.reshape(
                max_num_frame, -1, audio_no_pad.shape[-1]
            )

        mixed_embeds = []
        for i, vis_embed in enumerate(vision_embeds):
            if i < audio_no_pad.shape[0]:
                audio_frame = audio_no_pad[i]
                # Pad or trim audio to match vision token count
                vis_tokens = vis_embed.shape[0]
                aud_tokens = audio_frame.shape[0]

                if aud_tokens < vis_tokens:
                    padding = torch.zeros(
                        vis_tokens - aud_tokens,
                        audio_frame.shape[-1],
                        dtype=audio_frame.dtype,
                        device=audio_frame.device,
                    )
                    audio_frame = torch.cat((audio_frame, padding), dim=0)
                elif aud_tokens > vis_tokens:
                    audio_frame = audio_frame[:vis_tokens]

                mixed_embeds.append(vis_embed + audio_frame)
            else:
                mixed_embeds.append(vis_embed)

        return tuple(mixed_embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Generate embeddings from video+audio multimodal data."""
        # Process vision
        video_input = self._parse_and_validate_video_input(**kwargs)
        if video_input is None:
            return []

        vision_embeds = self._process_vision_input(video_input)

        # Process audio
        audio_embeds = self._process_audio_input(**kwargs)
        duration = kwargs.get("audio_duration")
        if isinstance(duration, torch.Tensor):
            duration = int(duration.item())

        # Reference alignment logic:
        #   if duration < pixel_values.shape[0]:
        #       pixel_values = pixel_values[:duration]
        #   if duration <= max_num_frame:
        #       duration = pixel_values.shape[0]
        num_frames = len(vision_embeds)
        max_num_frame = self.config.max_num_frame
        if duration is not None and duration < num_frames:
            vision_embeds = vision_embeds[:duration]
            num_frames = duration
        if duration is not None and duration <= max_num_frame:
            duration = len(vision_embeds)

        # Mix vision and audio
        mixed_embeds = self._create_mixed_embeddings(
            vision_embeds, audio_embeds, duration
        )

        return mixed_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model.model",
            connector=["visual.perceive", "audio_projection"],
            tower_model=["visual.", "audio_encoder."],
        )
