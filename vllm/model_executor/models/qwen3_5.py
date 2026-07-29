# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
    _is_shared_expert_fse_compatible,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    # Qwen3.5 ships the GDN in_proj checkpoints separately (qwen3-next
    # pre-fuses them); fuse them on top of the qwen3-next QKV/gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # FSE must match construction (Qwen3NextSparseMoeBlock): reroute the
        # shared expert into the extra fused slot only when AITER FSE is both
        # requested and compatible with the quant spec.
        if "moe" in self.config.model_type:
            weights = maybe_fuse_shared_experts(
                weights,
                enabled=rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
                and _is_shared_expert_fse_compatible(self.quant_config),
                n_routed_experts=self.config.num_experts,
                n_shared_experts=1,
                ckpt_prefix="mlp.shared_expert",
            )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3.5 (text-only) Multi-head Sequence Classification
########################################################


class Qwen3_5ClassifierHead(nn.Module):
    """Shared MLP feature extractor + N independent binary Linear heads.

    Applied on top of last-token pooled hidden states. Produces raw logits of
    shape ``[B, num_models]``. Independent-sigmoid activation is applied later
    by the ``ClassifierPoolerHead`` via ``PoolerMultiLabelClassify``.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        num_models: int,
        classifier_bias: bool = False,
        classifier_dropout: float = 0.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # NOTE: ``shared_mlp`` MUST be an ``nn.Sequential`` with the Linear at
        # index 0 to match checkpoint keys ``shared_mlp.0.weight`` etc.
        layers: list[nn.Module] = [
            nn.Linear(
                hidden_size, mlp_hidden_size, bias=classifier_bias, dtype=dtype
            ),
            nn.GELU(),
        ]
        if classifier_dropout > 0.0:
            layers.append(nn.Dropout(classifier_dropout))
        self.shared_mlp = nn.Sequential(*layers)
        # Per-model binary heads: N x Linear(mlp_hidden, 1) — matches
        # checkpoint keys ``heads.<i>.weight``.
        self.heads = nn.ModuleList(
            [
                nn.Linear(mlp_hidden_size, 1, bias=classifier_bias, dtype=dtype)
                for _ in range(num_models)
            ]
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # Cast pooled features into the head's parameter dtype for stable
        # matmul (mirrors the training-time behaviour).
        head_dtype = self.shared_mlp[0].weight.dtype
        features = self.shared_mlp(pooled.to(head_dtype))  # [B, H']
        # Stack per-head scalar logits into [B, num_models].
        return torch.stack(
            [head(features).squeeze(-1) for head in self.heads], dim=1
        )


class Qwen3_5ForSequenceClassification(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
):
    """Qwen3.5 (text-only) multi-head sequence classification model.

    Backbone: :class:`Qwen3_5Model` (mixed linear+full attention text tower).
    Head:     :class:`Qwen3_5ClassifierHead` (shared MLP + N binary heads).
    Pooling:  LAST token; independent-sigmoid activation on the raw logits.

    Checkpoint layout (produced by the HF trainer using
    ``Qwen3_5ForSequenceClassification``)::

        backbone.embed_tokens.weight, backbone.layers.*, backbone.norm.weight
        shared_mlp.0.weight [+ .bias]
        heads.<i>.weight    [+ .bias]        (i in 0..num_models-1)

    which is remapped onto vLLM as::

        model.embed_tokens.weight, model.layers.*, model.norm.weight
        classifier.shared_mlp.0.weight
        classifier.heads.<i>.weight
    """

    # Mark this as a pooling model so vLLM skips lm_head construction and
    # dispatches to the pooler for classify tasks.
    is_pooling_model = True

    # Reuse the packed-module fusion mapping (QKV / gate_up / GDN in_proj) so
    # loading works even under TP.
    packed_modules_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping

    # Remap checkpoint keys: ``backbone.<x>`` -> ``model.<x>``; classifier keys
    # (``shared_mlp.*`` and ``heads.*``) are renamed to live under
    # ``classifier.*`` so they load into ``self.classifier``.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "backbone.": "model.",
            "shared_mlp.": "classifier.shared_mlp.",
            "heads.": "classifier.heads.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        # Same guard as Qwen3_5ForCausalLMBase.
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        hf_config = vllm_config.model_config.hf_config
        self.config = hf_config

        # Backbone: text-only Qwen3.5 model.
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Classification head knobs (read from the top-level HF config).
        num_models: int = int(
            getattr(hf_config, "num_models", getattr(hf_config, "num_labels", 3))
        )
        hidden_size: int = int(
            getattr(
                hf_config,
                "hidden_size",
                vllm_config.model_config.hf_text_config.hidden_size,
            )
        )
        mlp_hidden_size: int = int(getattr(hf_config, "mlp_hidden_size", hidden_size))
        classifier_bias: bool = bool(getattr(hf_config, "classifier_bias", False))
        classifier_dropout: float = float(
            getattr(hf_config, "classifier_dropout", 0.0)
        )

        # Build the multi-head classifier in the model's head dtype so weights
        # load cleanly from the (usually bfloat16) checkpoint.
        self.classifier = Qwen3_5ClassifierHead(
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            num_models=num_models,
            classifier_bias=classifier_bias,
            classifier_dropout=classifier_dropout,
            dtype=vllm_config.model_config.head_dtype,
        )

        # Pooler: LAST-token pooling -> classifier -> sigmoid.
        # ``resolve_classifier_act_fn`` will pick ``PoolerMultiLabelClassify``
        # (sigmoid) because the config has
        # ``problem_type="multi_label_classification"``.
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=LastPool(),
            classifier=self.classifier,
        )

    # ---- Mamba / hybrid state plumbing (mirrors Qwen3NextForCausalLM) ----

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    # ---- forward / loading ---------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
