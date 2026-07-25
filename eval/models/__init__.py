"""Model registry for EnvShip-Bench baselines."""

from .sequence_models import LSTM2L, LSTM4L, GRU2L, BiLSTM2L
from .tcn import TCNModel
from .transformer_variants import Transformer3L, Transformer6L
from .transformer_nar import TransformerNAR
from .traisformer_local import TrAISformerLocal
from .context_models import (
    LSTMSocialPool, LSTMSocialAttn,
    LSTMEnvDesc, LSTMEnvRaster,
    LSTMSocialEnv,
)
from .sequence_models_v2 import LSTM2L_v2, BiLSTM2L_v2
from .context_models_v2 import (
    LSTMSocialAttnV2, LSTMEnvDescV2, LSTMSocialEnvV2,
)
from .sequence_models_v3 import BiLSTM2L_v3
from .context_models_v3 import LSTMEnvDescV3, LSTMSocialEnvV3
from .context_models_v4 import (
    LSTMEnvSDF, LSTMEnvSpatialAttn,
    LSTMEnvBinarySpatialAttn, LSTMSocialEnvSDF,
)
from .context_models_v5 import LSTMSocialAttnV5

MODEL_REGISTRY = {
    # ── Sequence RNN v1 ───────────────────────────────────
    "lstm_2l":    LSTM2L,
    "lstm_4l":    LSTM4L,
    "gru_2l":     GRU2L,
    "bilstm_2l":  BiLSTM2L,
    # ── TCN ──────────────────────────────────────────────
    "tcn":        TCNModel,
    # ── Transformer v1 (broken AR) ───────────────────────
    "transformer_3l": Transformer3L,
    "transformer_6l": Transformer6L,
    # ── Transformer NAR (new) ────────────────────────────
    "transformer_nar": TransformerNAR,
    # ── TrAISformer (discrete token) ─────────────────────
    "traisformer": TrAISformerLocal,
    # ── Context-aware v1 ─────────────────────────────────
    "lstm_social_pool":  LSTMSocialPool,
    "lstm_social_attn":  LSTMSocialAttn,
    "lstm_env_desc":     LSTMEnvDesc,
    "lstm_env_raster":   LSTMEnvRaster,
    "lstm_social_env":   LSTMSocialEnv,
    # ── Sequence RNN v2 ──────────────────────────────────
    "lstm_2l_v2":    LSTM2L_v2,
    "bilstm_2l_v2":  BiLSTM2L_v2,
    # ── Context-aware v2 (per-step attn/FiLM) ────────────
    "lstm_social_attn_v2": LSTMSocialAttnV2,
    "lstm_env_desc_v2":    LSTMEnvDescV2,
    "lstm_social_env_v2":  LSTMSocialEnvV2,
    # ── Sequence RNN v3 (BiLSTM + enc cross-attn) ────────
    "bilstm_2l_v3":       BiLSTM2L_v3,
    # ── Context-aware v3 (enc cross-attn + FiLM) ─────────
    "lstm_env_desc_v3":   LSTMEnvDescV3,
    "lstm_social_env_v3": LSTMSocialEnvV3,
    # ── Context-aware v4 (SDF input + spatial attention) ─
    "lstm_env_sdf":          LSTMEnvSDF,
    "lstm_env_spatial_attn": LSTMEnvSpatialAttn,
    "lstm_env_binary_spatial_attn": LSTMEnvBinarySpatialAttn,
    "lstm_social_env_sdf":   LSTMSocialEnvSDF,
    # ── Context-aware v5 (improved social: CPA/TCPA + gating + per-step attn) ─
    "lstm_social_attn_v5":   LSTMSocialAttnV5,
}

CONTEXT_MODELS = {
    "lstm_social_pool", "lstm_social_attn",
    "lstm_env_desc", "lstm_env_raster",
    "lstm_social_env",
    # v2
    "lstm_social_attn_v2", "lstm_env_desc_v2", "lstm_social_env_v2",
    # v3
    "lstm_env_desc_v3", "lstm_social_env_v3",
    # v4
    "lstm_env_sdf", "lstm_env_spatial_attn",
    "lstm_env_binary_spatial_attn", "lstm_social_env_sdf",
    # v5
    "lstm_social_attn_v5",
}

RASTER_MODELS = {"lstm_env_raster", "lstm_social_env",
                 "lstm_env_sdf", "lstm_env_spatial_attn",
                 "lstm_env_binary_spatial_attn", "lstm_social_env_sdf"}
