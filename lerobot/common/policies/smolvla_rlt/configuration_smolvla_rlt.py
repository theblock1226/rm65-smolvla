"""Configuration for SmolVLA with RL Token (RLT) policy.

RLT extracts a compact RL token from frozen SmolVLA embeddings and trains
a lightweight actor-critic on top via online RL, enabling the robot to
improve from its own practice beyond imitation learning.

Reference: "RL Token" (Physical Intelligence, 2025)
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("smolvla_rlt")
@dataclass
class SmolVLARLTConfig(PreTrainedConfig):
    # ── SmolVLA base model ──────────────────────────────────────────────
    smolvla_pretrained_path: str = ""  # Path to pretrained SmolVLA checkpoint

    # These mirror SmolVLAConfig defaults for the frozen VLA
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 48
    num_steps: int = 10  # VLA denoising steps
    num_vlm_layers: int = 16
    num_expert_layers: int = -1
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    attention_mode: str = "cross_attn"
    freeze_vision_encoder: bool = True
    use_cache: bool = True
    prefix_length: int = -1
    add_image_special_tokens: bool = False
    pad_language_to: str = "longest"
    n_obs_steps: int = 1

    # ── RLT encoder-decoder ─────────────────────────────────────────────
    # Measured from local RM65 SmolVLA (usb_smolvla / astra_smolvla) extract_embeddings:
    #   vlm [B, L, 960], expert [B, T, 720]  (not the stock 576 / 432)
    vlm_hidden_dim: int = 960
    expert_hidden_dim: int = 720
    rlt_hidden_dim: int = 256      # z_rl bottleneck dimension
    rlt_encoder_layers: int = 4
    rlt_decoder_layers: int = 4
    rlt_num_heads: int = 8
    rlt_dropout: float = 0.1

    # ── Actor MLP ───────────────────────────────────────────────────────
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    ref_dropout: float = 0.5      # Probability of zeroing ref_actions during actor training

    # ── Critic MLP (twin Q) ─────────────────────────────────────────────
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])

    # ── RM65 action space (6 joints + gripper) ─────────────────────────
    action_dim: int = 7
    state_dim: int = 7
    n_action_steps_rl: int = 10    # RL action chunk size
    action_stride: int = 2         # Subsample VLA's 50-step chunk by stride 2

    # ── RM65 cameras / frozen VLA family ────────────────────────────────
    # USB insert → front+mono + usb_smolvla
    # desk/slot/stack → front+astra + astra_smolvla
    # (Inference remapping camera1/2/3 is separate; RLT Stage2 uses these names.)
    camera_set: str = "stack"  # desk|slot|stack|usb|default — see robot_adapter.CAMERA_SETS
    smolvla_family: str = "astra"  # usb | astra
    smolvla_usb_path: str = "usb_smolvla"
    smolvla_astra_path: str = "astra_smolvla"

    # ── RL hyperparameters ──────────────────────────────────────────────
    discount: float = 0.99
    bc_weight: float = 0.1         # BC regularization coefficient
    utd_ratio: int = 5             # Update-to-data ratio
    critic_per_actor: int = 2      # Critic updates per actor update
    target_tau: float = 0.005      # EMA target network update rate
    replay_buffer_capacity: int = 100_000
    rl_batch_size: int = 256

    # ── Training mode ───────────────────────────────────────────────────
    # "rlt_training" = Stage 1 (encoder-decoder on demos)
    # "online_rl"    = Stage 2 (actor-critic on robot)
    # "inference"    = Deployment
    mode: str = "rlt_training"

    # ── Stage 1 optimizer ───────────────────────────────────────────────
    rlt_lr: float = 1e-4
    rlt_weight_decay: float = 1e-5
    rlt_warmup_steps: int = 200
    rlt_total_steps: int = 5000

    # ── Stage 2 optimizer ───────────────────────────────────────────────
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # ── Normalization ───────────────────────────────────────────────────
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Paths for loading stage checkpoints
    rlt_encoder_path: str = ""     # Trained encoder checkpoint (for Stage 2 / inference)
    rlt_decoder_path: str = ""     # Trained decoder checkpoint (optional, for evaluation)
    actor_path: str = ""           # Trained actor checkpoint (for inference)
    critic_path: str = ""          # Trained critic checkpoint (optional)

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("rlt_training", "online_rl", "inference"):
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of: rlt_training, online_rl, inference")
        if self.smolvla_family not in ("usb", "astra"):
            raise ValueError(f"smolvla_family must be 'usb' or 'astra', got {self.smolvla_family!r}")
        if self.camera_set not in ("desk", "slot", "stack", "usb", "default"):
            raise ValueError(
                f"camera_set must be desk|slot|stack|usb|default, got {self.camera_set!r}"
            )
        # Keep family ↔ camera consistent when user only sets one of them
        if self.smolvla_family == "usb" and self.camera_set != "usb":
            self.camera_set = "usb"
        if self.smolvla_family == "astra" and self.camera_set == "usb":
            # USB camera set belongs with usb family; fall back to stack (front+astra)
            self.camera_set = "stack"
        if not self.smolvla_pretrained_path:
            self.smolvla_pretrained_path = (
                self.smolvla_usb_path if self.smolvla_family == "usb" else self.smolvla_astra_path
            )

    def camera_names(self) -> list[str]:
        from lerobot.common.policies.smolvla_rlt.robot_adapter import CAMERA_SETS

        return list(CAMERA_SETS.get(self.camera_set, CAMERA_SETS["default"]))

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.rlt_lr,
            weight_decay=self.rlt_weight_decay,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.n_action_steps_rl))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def n_action_steps(self) -> int:
        return self.n_action_steps_rl
