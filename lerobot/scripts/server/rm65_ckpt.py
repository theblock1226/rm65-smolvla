"""Helpers to load RM65 SmolVLA checkpoints offline (local VLM + normalizer stats)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
LOCAL_VLM = ROOT / "base_models" / "SmolVLM2-500M-Video-Instruct"
COMPAT_DROP_KEYS = {
    "use_peft",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "pretrained_path",
    "pretrained_revision",
    "compile_model",
    "compile_mode",
}


def sanitize_ckpt_config(ckpt_dir: Path | str) -> dict:
    ckpt_dir = Path(ckpt_dir)
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    dropped = sorted(k for k in COMPAT_DROP_KEYS if k in cfg)
    for k in dropped:
        cfg.pop(k, None)
    prev_vlm = cfg.get("vlm_model_name")
    if LOCAL_VLM.is_dir():
        cfg["vlm_model_name"] = str(LOCAL_VLM)
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
    return {"dropped_keys": dropped, "vlm_was": prev_vlm, "vlm_now": cfg.get("vlm_model_name")}


def load_dataset_stats_from_ckpt(ckpt_dir: Path | str) -> dict:
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_dir)
    candidates: list[Path] = []
    pre = ckpt_dir / "policy_preprocessor.json"
    if pre.is_file():
        meta = json.loads(pre.read_text())
        for step in meta.get("steps", []):
            sf = step.get("state_file")
            if sf and "normalizer" in str(sf):
                candidates.append(ckpt_dir / sf)
    candidates.append(ckpt_dir / "policy_preprocessor_step_6_normalizer_processor.safetensors")
    candidates.append(ckpt_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    candidates.append(ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}

    raw = load_file(str(path))
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        if "." not in key:
            continue
        feat, stat = key.rsplit(".", 1)
        if stat not in ("mean", "std", "min", "max"):
            continue
        stats.setdefault(feat, {})[stat] = tensor
    return stats


def trim_stats_to_config(stats: dict, config) -> dict:
    out = {}
    for key, ft in {**config.input_features, **config.output_features}.items():
        if key not in stats:
            continue
        need = int(ft.shape[0]) if len(ft.shape) >= 1 else None
        entry = {}
        for sk, val in stats[key].items():
            t = val.clone()
            if need is not None and t.ndim >= 1 and t.shape[0] != need and t.shape[0] > need:
                t = t[:need]
            entry[sk] = t
        out[key] = entry
    return out


def load_offline_smolvla_config(ckpt_dir: Path | str):
    """Parse a checkpoint config without rewriting it.

    Drops trainer-only fields this fork's ``SmolVLAConfig`` does not declare, and
    points ``vlm_model_name`` at the local SmolVLM tree when the Hub id is not a directory.
    """
    import tempfile
    from dataclasses import fields

    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig

    ckpt_dir = Path(ckpt_dir)
    raw = json.loads((ckpt_dir / "config.json").read_text())
    allowed = {f.name for f in fields(SmolVLAConfig)} | {f.name for f in fields(PreTrainedConfig)} | {"type"}
    dropped = sorted(set(raw) - allowed)
    raw = {k: v for k, v in raw.items() if k in allowed}
    vlm = str(raw.get("vlm_model_name", ""))
    if LOCAL_VLM.is_dir() and not Path(vlm).is_dir():
        raw["vlm_model_name"] = str(LOCAL_VLM)
    with tempfile.TemporaryDirectory(prefix="smolvla_cfg_") as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps(raw))
        config = PreTrainedConfig.from_pretrained(tmp)
    logger.info(
        "offline SmolVLA config dropped=%s vlm=%s",
        dropped,
        getattr(config, "vlm_model_name", None),
    )
    return config


def load_smolvla_for_server(pretrained_name_or_path: str, device: str, *, sanitize: bool = True):
    """Sanitize + load SmolVLA with normalizer stats (for PolicyServer)."""
    from lerobot.common.policies.factory import get_policy_class
    from lerobot.configs.policies import PreTrainedConfig

    ckpt = Path(pretrained_name_or_path)
    if sanitize and ckpt.is_dir():
        info = sanitize_ckpt_config(ckpt)
        logger.info("ckpt sanitize: %s", info)

    config = PreTrainedConfig.from_pretrained(str(pretrained_name_or_path))
    policy_class = get_policy_class(config.type if hasattr(config, "type") else "smolvla")
    # Prefer explicit type from path usage
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_class = SmolVLAPolicy
    raw_stats = load_dataset_stats_from_ckpt(ckpt) if ckpt.is_dir() else {}
    stats = trim_stats_to_config(raw_stats, config) if raw_stats else None
    policy = policy_class.from_pretrained(
        str(pretrained_name_or_path),
        config=config,
        dataset_stats=stats,
    )
    policy.to(device)
    policy.eval()
    return policy
