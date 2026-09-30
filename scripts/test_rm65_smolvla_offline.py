#!/usr/bin/env python3
"""Offline tests for downloaded RM65 SmolVLA checkpoints (no robot).

Runs against ``usb_smolvla`` and ``astra_smolvla``:
  1) load + config summary (sanitize cloud-only fields, local VLM path)
  2) inject normalizer stats from processor safetensors
  3) synthetic batch → predict_action_chunk
  4) extract_embeddings (RLT Stage1 readiness)
  5) RTC prefix-guidance smoke
  6) short chunk latency profile

Example::

    PYTHONPATH=. HF_HUB_OFFLINE=1 python3 scripts/test_rm65_smolvla_offline.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOCAL_VLM = ROOT / "base_models" / "SmolVLM2-500M-Video-Instruct"

CHECKPOINTS = {
    "usb": ROOT / "usb_smolvla",
    "astra": ROOT / "astra_smolvla",
}

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


def parse_args():
    p = argparse.ArgumentParser(description="Offline dual SmolVLA ckpt tests")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--which", choices=["usb", "astra", "both"], default="both")
    p.add_argument("--profile_steps", type=int, default=30)
    p.add_argument("--output", type=str, default="outputs/profiling/rm65_smolvla_offline_report.json")
    return p.parse_args()


def force_offline() -> None:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def load_dataset_stats_from_ckpt(ckpt_dir: Path) -> dict:
    from safetensors.torch import load_file

    path = ckpt_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    if not path.is_file():
        path = ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"No normalizer safetensors in {ckpt_dir}")

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


def sanitize_ckpt_config(ckpt_dir: Path) -> dict:
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


def summarize_config(ckpt_dir: Path) -> dict:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    return {
        "path": str(ckpt_dir),
        "type": cfg.get("type"),
        "input_features": {k: v.get("shape") for k, v in cfg.get("input_features", {}).items()},
        "output_features": {k: v.get("shape") for k, v in cfg.get("output_features", {}).items()},
        "chunk_size": cfg.get("chunk_size"),
        "n_action_steps": cfg.get("n_action_steps"),
        "empty_cameras": cfg.get("empty_cameras"),
        "vlm_model_name": cfg.get("vlm_model_name"),
        "schema_notes": {
            "state_dim": (cfg.get("input_features") or {}).get("observation.state", {}).get("shape"),
            "action_dim": (cfg.get("output_features") or {}).get("action", {}).get("shape"),
            "vs_local_cleaned": (
                "cleaned: state/action 7-D + front/(astra|mono); "
                "ckpt: state 6-D + camera1/2/3 (+ empty) + action 7. Remap needed for dataset eval."
            ),
        },
    }


def build_batch_from_config(policy, device: torch.device, seed: int = 0) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    batch: dict = {"task": "Insert the USB module into the base."}
    state_shape = policy.config.robot_state_feature.shape
    batch["observation.state"] = torch.randn(1, *state_shape, generator=g).to(device)
    for key, ft in policy.config.input_features.items():
        if not key.startswith("observation.images."):
            continue
        if "empty_camera" in key:
            continue
        c, h, w = ft.shape
        batch[key] = torch.rand(1, c, h, w, generator=g).to(device)
    return batch


@torch.no_grad()
def test_one(name: str, ckpt_dir: Path, device: torch.device, profile_steps: int) -> dict:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
    from lerobot.common.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import RTCAttentionSchedule

    report: dict = {
        "name": name,
        "pass_load": False,
        "pass_chunk": False,
        "pass_embed": False,
        "pass_rtc": False,
    }
    report["config"] = summarize_config(ckpt_dir)

    if not (ckpt_dir / "model.safetensors").is_file():
        report["error"] = "missing model.safetensors"
        return report

    compat = sanitize_ckpt_config(ckpt_dir)
    report["compat_patch"] = compat
    logger.info("compat patch %s: %s", name, compat)

    config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    raw_stats = load_dataset_stats_from_ckpt(ckpt_dir)
    stats = trim_stats_to_config(raw_stats, config)
    report["stats_keys"] = sorted(stats.keys())
    report["stats_state_dim"] = list(stats.get("observation.state", {}).get("mean", torch.empty(0)).shape)
    report["stats_action_dim"] = list(stats.get("action", {}).get("mean", torch.empty(0)).shape)

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir), config=config, dataset_stats=stats)
    policy.to(device)
    policy.eval()
    report["load_s"] = round(time.perf_counter() - t0, 2)
    report["pass_load"] = True
    report["device"] = str(device)

    batch = build_batch_from_config(policy, device)
    report["batch_keys"] = sorted(k for k in batch.keys() if k != "task")

    t0 = time.perf_counter()
    actions = policy.predict_action_chunk(batch)
    report["chunk_shape"] = list(actions.shape)
    report["chunk_ms_first"] = round((time.perf_counter() - t0) * 1000, 1)
    act_dim = policy.config.output_features["action"].shape[0]
    report["pass_chunk"] = actions.ndim == 3 and actions.shape[0] == 1 and actions.shape[-1] == act_dim

    t0 = time.perf_counter()
    _ = policy.predict_action_chunk(batch)
    report["chunk_ms_second"] = round((time.perf_counter() - t0) * 1000, 1)

    try:
        batch_n = policy.normalize_inputs(dict(batch))
        dummy_act = torch.zeros(
            1, policy.config.chunk_size, act_dim, device=device, dtype=batch["observation.state"].dtype
        )
        batch_n = {**batch_n, "action": dummy_act}
        batch_n = policy.normalize_targets(batch_n)
        images, img_masks = policy.prepare_images(batch_n)
        state = policy.prepare_state(batch_n)
        lang_tokens, lang_masks = policy.prepare_language(batch_n)
        actions_t = policy.prepare_action(batch_n)
        actions_pad = pad_vector(actions_t, policy.config.max_action_dim)
        vlm_emb, expert_emb = policy.model.extract_embeddings(
            images, img_masks, lang_tokens, lang_masks, state, actions_pad
        )
        report["embed_vlm_shape"] = list(vlm_emb.shape)
        report["embed_expert_shape"] = list(expert_emb.shape)
        report["pass_embed"] = vlm_emb.ndim == 3 and expert_emb.ndim == 3
    except Exception as exc:  # noqa: BLE001
        report["embed_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("extract_embeddings failed for %s", name)

    try:
        policy.config.rtc_config = RTCConfig(
            enabled=True,
            mode="guided",
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            max_guidance_weight=10.0,
            execution_horizon=10,
            debug=False,
        )
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()
        noise = torch.randn(
            1, policy.config.chunk_size, policy.config.max_action_dim, device=device
        )
        leftover = actions.squeeze(0).detach()[10:].clone()
        t0 = time.perf_counter()
        rtc_out = policy.predict_action_chunk(
            batch,
            noise=noise.clone(),
            prev_chunk_left_over=leftover,
            inference_delay=5,
            execution_horizon=10,
        )
        report["rtc_chunk_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        report["rtc_shape"] = list(rtc_out.shape)
        mae = (rtc_out.squeeze(0)[:5, : leftover.shape[-1]].cpu() - leftover[:5].cpu()).abs().mean().item()
        report["rtc_prefix_mae_vs_leftover"] = round(mae, 4)
        report["pass_rtc"] = list(rtc_out.shape) == list(actions.shape)
    except Exception as exc:  # noqa: BLE001
        report["rtc_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("RTC smoke failed for %s", name)

    lat = []
    n_prof = min(5, max(1, profile_steps // 10))
    for _ in range(n_prof):
        t0 = time.perf_counter()
        _ = policy.predict_action_chunk(batch)
        lat.append((time.perf_counter() - t0) * 1000)
    report["profile_chunk_ms"] = {
        "n": len(lat),
        "mean": round(sum(lat) / len(lat), 1),
        "min": round(min(lat), 1),
        "max": round(max(lat), 1),
    }
    mean_ms = report["profile_chunk_ms"]["mean"]
    report["suggested_rtc"] = {
        "inference_delay_steps": max(1, int(round(mean_ms / 50.0))),
        "execution_horizon_steps": max(3, int(round(mean_ms / 50.0)) + 2),
        "note": "from mean chunk_ms / 50ms (20fps)",
    }

    report["pass"] = bool(report["pass_load"] and report["pass_chunk"] and report["pass_embed"])
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> int:
    args = parse_args()
    force_offline()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if not LOCAL_VLM.is_dir():
        raise SystemExit(f"Local VLM missing: {LOCAL_VLM}")

    names = ["usb", "astra"] if args.which == "both" else [args.which]
    results = {}
    for name in names:
        ckpt = CHECKPOINTS[name]
        logger.info("==== testing %s (%s) ====", name, ckpt)
        results[name] = test_one(name, ckpt, device, args.profile_steps)
        logger.info(
            "%s pass=%s load=%.1fs chunk=%.1fms embed=%s rtc=%s",
            name,
            results[name].get("pass"),
            results[name].get("load_s", -1),
            results[name].get("chunk_ms_second", -1),
            results[name].get("pass_embed"),
            results[name].get("pass_rtc"),
        )

    summary = {
        "pass": all(r.get("pass") for r in results.values()),
        "device": str(device),
        "results": results,
        "cleaned_alignment": {
            "usb_local": "datasets/cleaned/rm65_usb_insert — state7 + front/mono",
            "astra_local": "datasets/cleaned/rm65_{stacking,cleanup,place} — state7 + front/astra",
            "ckpt_common": "state6 + camera1/2/3 (+ empty) + action7",
            "action_needed": "key remap + state trim/pad before dataset-based eval / Stage1",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
