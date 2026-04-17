# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration and tokenizer classes for ARC-Hunyuan-Video-7B model."""

import sys
import types

from transformers import PretrainedConfig, PreTrainedTokenizerFast


class ARCHunyuanVideoTokenizer(PreTrainedTokenizerFast):
    """Tokenizer for ARC-Hunyuan-Video-7B.

    A thin wrapper around PreTrainedTokenizerFast that knows the vocab file
    is named ``hunyuan.tiktoken`` (tiktoken BPE format). Transformers'
    built-in TikTokenConverter handles the actual conversion.
    """

    vocab_files_names = {"vocab_file": "hunyuan.tiktoken"}
    model_input_names = ["input_ids", "attention_mask"]


def _register_arc_hunyuan_video_tokenizer():
    """Make ARCHunyuanVideoTokenizer discoverable by AutoTokenizer.

    ``AutoTokenizer.from_pretrained`` resolves ``tokenizer_class`` names via
    ``tokenizer_class_from_name``, which iterates
    ``TOKENIZER_MAPPING_NAMES`` and imports from
    ``transformers.models.<model_type>``.  We inject our class into that
    lookup so that the ``"ARCHunyuanVideoTokenizer"`` reference in the HF
    ``tokenizer_config.json`` resolves without requiring a custom
    transformers fork.
    """
    from transformers.models.auto.tokenization_auto import (
        TOKENIZER_MAPPING_NAMES,
    )

    key = "arc_hunyuan_video"
    if key not in TOKENIZER_MAPPING_NAMES:
        TOKENIZER_MAPPING_NAMES[key] = (
            "ARCHunyuanVideoTokenizer",
            "ARCHunyuanVideoTokenizer",
        )

    # Provide a synthetic module so that ``importlib.import_module``
    # inside ``tokenizer_class_from_name`` succeeds.
    mod_name = "transformers.models.arc_hunyuan_video"
    if mod_name not in sys.modules:
        mod = types.ModuleType(mod_name)
        mod.ARCHunyuanVideoTokenizer = ARCHunyuanVideoTokenizer
        sys.modules[mod_name] = mod


# Register at import time so the tokenizer is available before
# AutoTokenizer.from_pretrained is called.
_register_arc_hunyuan_video_tokenizer()


class ARCHunyuanVideoVisionConfig(PretrainedConfig):
    model_type = "arc_hunyuan_video_vision"

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=1152,
        intermediate_size=4304,
        interpolate_mode="bilinear",
        rms_norm_eps=1e-05,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_channels=3,
        num_hidden_layers=27,
        out_hidden_size=4096,
        patch_size=16,
        spatial_merge_size=2,
        anyres_pooling_size=2,
        force_image_size=640,
        max_image_size=2048,
        max_vit_seq_len=1600,
        attention_head_dim=72,
        attention_bias=True,
        mlp_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.interpolate_mode = interpolate_mode
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.anyres_pooling_size = anyres_pooling_size
        self.force_image_size = force_image_size
        self.max_image_size = max_image_size
        self.max_vit_seq_len = max_vit_seq_len
        self.attention_head_dim = attention_head_dim
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias

        # Computed: effective merge size for token count calculation
        # force_image_size / patch_size / spatial_merge_size / anyres_pooling_size
        self.text_hidden_size = out_hidden_size


class ARCHunyuanVideoAudioConfig(PretrainedConfig):
    model_type = "arc_hunyuan_video_audio"

    def __init__(
        self,
        d_model=1280,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layers=32,
        encoder_layerdrop=0.0,
        num_mel_bins=128,
        max_source_positions=1500,
        activation_function="gelu",
        scale_embedding=False,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_hidden_layers=32,
        vocab_size=51866,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_layerdrop = encoder_layerdrop
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.activation_function = activation_function
        self.scale_embedding = scale_embedding
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size


class ARCHunyuanVideoTextConfig(PretrainedConfig):
    model_type = "arc_hunyuan_video_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=20480,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=127961,
        bos_token_id=127959,
        eos_token_id=127960,
        eod_token_id=127957,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=128,
        use_qk_norm=True,
        use_rotary_pos_emb=True,
        position_embedding_xdrope=True,
        xdrope_section=None,
        image_token_id=127968,
        im_start_id=127962,
        im_end_id=127963,
        im_newline_id=127964,
        num_media_embeds=257,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.position_embedding_xdrope = position_embedding_xdrope
        self.xdrope_section = xdrope_section or [0.25, 0.25, 0.25, 0.25]
        self.image_token_id = image_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.im_newline_id = im_newline_id
        self.num_media_embeds = num_media_embeds

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ARCHunyuanVideoConfig(PretrainedConfig):
    model_type = "arc_hunyuan_video"
    sub_configs = {
        "vision_config": ARCHunyuanVideoVisionConfig,
        "text_config": ARCHunyuanVideoTextConfig,
        "audio_config": ARCHunyuanVideoAudioConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        max_num_frame=150,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        if isinstance(audio_config, dict):
            self.audio_config = self.sub_configs["audio_config"](**audio_config)
        elif audio_config is None:
            self.audio_config = self.sub_configs["audio_config"]()
        else:
            self.audio_config = audio_config

        self.max_num_frame = max_num_frame

        # Token IDs from text_config
        self.image_token_id = self.text_config.image_token_id
        self.image_start_token_id = self.text_config.im_start_id
        self.image_end_token_id = self.text_config.im_end_id

        self.vision_config.text_hidden_size = self.text_config.hidden_size

        # ARC model: the perceive Conv2d uses spatial_merge_size (=2) as
        # kernel/stride.  The effective merge for token-count calculations is
        # spatial_merge_size * anyres_pooling_size (=4), but we must NOT
        # overwrite spatial_merge_size here because HunYuanVisionPatchMerger
        # uses it directly as the Conv2d kernel size and must match the
        # checkpoint weights [2304, 1152, 2, 2].

        # Ensure XDRoPE configuration is set for the text model.
        # Reference: convert_config_to_legacy sets rope_scaling with
        # alpha=1000.0 and mrope_section=[0.25, 0.25, 0.25, 0.25].
        # vllm's get_rope() needs rope_type="xdrope" to create the
        # correct XDRotaryEmbedding (DynamicNTKAlpha + XDRoPE sections).
        if self.text_config.rope_scaling is None and getattr(
            self.text_config, "position_embedding_xdrope", False
        ):
            head_dim = getattr(self.text_config, "head_dim", None) or (
                self.text_config.hidden_size // self.text_config.num_attention_heads
            )
            rotary_dim = head_dim
            xdrope_section_abs = [
                int(x * rotary_dim // 2) for x in self.text_config.xdrope_section
            ]
            self.text_config.rope_scaling = {
                "rope_type": "xdrope",
                "alpha": 1000.0,
                "xdrope_section": xdrope_section_abs,
            }

        # The HF config.json has max_position_embeddings=1024 as the base
        # RoPE window. With dynamic NTK-alpha scaling (alpha=1000), the
        # model supports much longer sequences. Update to the actual
        # supported context length so vLLM's max_model_len validation
        # accepts it.
        if self.text_config.max_position_embeddings < 20480:
            self.text_config.max_position_embeddings = 20480

        self._attn_implementation = kwargs.pop("attn_implementation", None)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config"))
            is not None
            and key not in ["dtype", "_attn_implementation_internal",
                            "architectures"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
            "architectures",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)
