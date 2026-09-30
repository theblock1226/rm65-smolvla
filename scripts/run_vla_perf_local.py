#!/usr/bin/env python3
"""Run every VLA-Perf experiment on the two local SmolVLA finetunes.

VLA-Perf is an analytical roofline model (NVIDIA). It does not load safetensors;
architecture is taken from each checkpoint's config.json (cameras, chunk, steps).

    /path/to/vla-perf/.venv/bin/python scripts/run_vla_perf_local.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path("/home/sdznkj/zkx_ws/sendlerobot")
VLA_PERF_ROOT = ROOT / "third_party/vla-perf"
VLA_PERF_DIR = VLA_PERF_ROOT / "vla-perf"
OUT_ROOT = ROOT / "outputs/diag/vla_perf"

# SmolVLM2-500M: 512px / patch16 -> 1024 ViT tokens, pixel-shuffle x4 -> 64 VLM tokens/image
SMOL_VISION_PATCH_TOKENS = 1024
SMOL_VLM_TOKENS_PER_IMAGE = 64

CKPTS = {
    "astra": ROOT / "astra_smolvla",
    "usb": ROOT / "usb_smolvla",
}

SYSTEMS = [
    "Jetson_Orin_NX_16GB",
    "Jetson_AGX_Thor",
    "RTX_4090",
    "A100_80GB",
    "H100",
    "B100",
]


def _count_cameras(cfg: dict) -> int:
    feats = cfg.get("input_features") or {}
    n = sum(1 for k, v in feats.items() if isinstance(v, dict) and v.get("type") == "VISUAL")
    return max(n, 1)


def config_from_ckpt(name: str, ckpt_dir: Path):
    from pi0_perf import Pi0Config

    cfg = json.loads((ckpt_dir / "config.json").read_text())
    n_cam = _count_cameras(cfg)
    lang = int(cfg.get("tokenizer_max_length") or 48)
    chunk = int(cfg.get("chunk_size") or 50)
    steps = int(cfg.get("num_steps") or 10)
    action = cfg.get("output_features", {}).get("action", {}).get("shape", [7])
    dof = int(action[0]) if action else 7
    vlm_seq = SMOL_VLM_TOKENS_PER_IMAGE * n_cam + lang
    return Pi0Config(
        name=f"smolvla-{name}",
        vision_model="smolvla-vision",
        vlm_model="smolvla-vlm",
        action_expert_model="smolvla-action-expert",
        vision_tokens=SMOL_VISION_PATCH_TOKENS,
        vision_frames=n_cam,
        language_tokens=lang,
        vlm_sequence_length=vlm_seq,
        action_chunk_size=chunk,
        denoising_steps=steps,
        action_dof=dof,
        vlm_tokens_per_image=SMOL_VLM_TOKENS_PER_IMAGE,
    )


def _patch_pi0(pi0, cfg) -> None:
    pi0.PI0_CONFIG = cfg
    pi0.ALL_PI0_CONFIGS = [cfg]


def run_ckpt(name: str, ckpt_dir: Path, pi0, logger) -> Path:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_from_ckpt(name, ckpt_dir)
    _patch_pi0(pi0, cfg)
    meta = {
        "ckpt": str(ckpt_dir),
        "name": cfg.name,
        "cameras": cfg.vision_frames,
        "chunk_size": cfg.action_chunk_size,
        "denoising_steps": cfg.denoising_steps,
        "vlm_sequence_length": cfg.vlm_sequence_length,
        "language_tokens": cfg.language_tokens,
        "action_dof": cfg.action_dof,
        "vision_patch_tokens": cfg.vision_tokens,
        "vlm_tokens_per_image": cfg.vlm_tokens_per_image,
    }
    (out / "ckpt_arch.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("architecture %s", meta)

    n_dev = [1, 2, 4]
    bits = "bf16"
    def _safe(label, fn):
        try:
            fn()
        except Exception:
            logger.exception("%s / %s failed", name, label)

    n = 0
    n += 1
    _safe(
        "EXP1",
        lambda: pi0.get_all_pi0_perf(
            system_list=SYSTEMS,
            num_device_list=n_dev,
            bits=bits,
            denoising_steps=cfg.denoising_steps,
            output_dir=str(out),
            experiment_num=n,
            logger=logger,
        ),
    )

    n += 1
    _safe(
        "EXP2",
        lambda: pi0.get_model_size_scaling_perf(
            system_list=["B100", "RTX_4090", "Jetson_AGX_Thor", "Jetson_Orin_NX_16GB"],
            num_device_list=[1],
            bits=bits,
            denoising_steps=cfg.denoising_steps,
            output_dir=str(out),
            experiment_num=n,
            logger=logger,
        ),
    )

    n += 1
    _safe(
        "EXP3",
        lambda: pi0.run_long_context_experiment(
            config=cfg,
            systems=SYSTEMS,
            timestep_counts=[1, 10, 100, 1000, 10000],
            bits=bits,
            output_dir=str(out),
            experiment_num=n,
            logger=logger,
        ),
    )

    n += 1
    _safe(
        "EXP4",
        lambda: pi0.compare_denoising_steps_action_lengths(
            configs=[cfg],
            systems=["B100", "RTX_4090", "Jetson_AGX_Thor", "Jetson_Orin_NX_16GB"],
            num_devices=1,
            bits=bits,
            step_range=[1, 5, 10, 20, 50],
            action_chunk_size_range=[5, 16, 24, 50, 100, 250],
            output_dir=str(out),
            experiment_num=n,
            logger=logger,
        ),
    )

    n += 1
    _safe(
        "EXP5",
        lambda: pi0.compare_autoregressive_vs_diffusion(
            config=cfg,
            systems=["B100", "RTX_4090", "Jetson_AGX_Thor", "Jetson_Orin_NX_16GB"],
            experiment_num=n,
            logger=logger,
            output_dir=str(out),
        ),
    )

    n += 1
    _safe(
        "EXP6",
        lambda: pi0.run_device_vs_server_comparison(
            output_dir=str(out),
            experiment_num=n,
            logger=logger,
        ),
    )

    n += 1
    _safe(
        "EXP7",
        lambda: pi0.run_device_server_collaboration_comparison(
            output_dir=str(out),
            server_system="B100",
            device_system="Jetson_Orin_NX_16GB",
            experiment_num=n,
            logger=logger,
        ),
    )

    def _net():
        import network_latency as nl

        nl.logger = logger
        nl.run_full_analysis(output_dir=str(out))

    n += 1
    _safe("network", _net)

    n += 1
    logger.info("==== %s boundness / param counts ====", name)
    from perf_utils import calculate_transformer_params, format_param_count
    from GenZ.Models.default_models import MODEL_DICT

    rows = []
    for key in ("smolvla-vision", "smolvla-vlm", "smolvla-action-expert"):
        mc = MODEL_DICT.get_model(key)
        npar = calculate_transformer_params(mc)
        rows.append({"component": key, "params": npar, "params_h": format_param_count(npar)})
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "smolvla_param_counts.csv", index=False)
    return out


def run_builtin_scripts(logger) -> Path:
    """Scripts that are architecture-fixed (OpenVLA / Pi0 validation / vision sweep)."""
    shared = OUT_ROOT / "builtin"
    shared.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(VLA_PERF_DIR)
    try:
        logger.info("==== builtin OpenVLA ====")
        import openvla_perf

        openvla_perf.get_openvla_e2e_perf(
            system_list=["A100_80GB", "H100", "B100", "Jetson_AGX_Thor", "Jetson_Orin_NX_16GB"],
            num_device_list=[1, 2, 4],
            bits="bf16",
            output_dir=str(shared),
            logger=logger,
        )

        logger.info("==== builtin validate_perf (Pi0 vs realtime-vla) ====")
        import validate_perf

        validate_perf.run_validation(
            systems=["RTX_4090", "RTX_5090"],
            views_list=[1, 2, 3],
            bits="bf16",
            output_dir=str(shared),
            logger=logger,
        )

        logger.info("==== builtin vision encoder sweep + smolvla-vision ====")
        sys.path.insert(0, str(VLA_PERF_DIR / "test_scripts"))
        import vision_encoder_perf as vep

        df = vep.get_perf_list(
            model_list=[
                "siglip2-base-patch16-224-vision",
                "siglip2-large-patch16-384-vision",
                "siglip2-so400m-patch14-384-vision",
                "siglip2-giant-opt-patch16-384-vision",
                "smolvla-vision",
            ],
            system_list=["A100_80GB", "H100", "B100", "Jetson_Orin_NX_16GB"],
            num_device_list=[1, 2, 4],
            bits="bf16",
            logger=logger,
        )
        if df is not None and not df.empty:
            df.to_csv(shared / "vla_vision_perf.csv", index=False)

        logger.info("==== builtin LLM backbone (smolvla-vlm) ====")
        import test_llm_perf as tlp

        tlp_results = tlp.get_llm_e2e_perf(
            system_list=["A100_80GB", "Jetson_Orin_NX_16GB"],
            num_device_list=[1],
            model_name="smolvla-vlm",
            input_tokens=176,
            output_tokens=24,
            output_tokens_parallel=24,
            bits="bf16",
            output_dir=str(shared),
            logger=logger,
        )
        if isinstance(tlp_results, dict):
            for k, v in tlp_results.items():
                if hasattr(v, "to_csv"):
                    v.to_csv(shared / f"llm_{k}_perf.csv", index=False)
    finally:
        os.chdir(old)
    return shared


def run_plots(ckpt_out: Path, logger) -> None:
    fig_dir = ckpt_out / "paper_figures"
    fig_dir.mkdir(exist_ok=True)
    # Plot scripts read ../perf_results relative to plot_scripts/
    staging = VLA_PERF_DIR / "perf_results"
    staging.mkdir(exist_ok=True)
    for csv in ckpt_out.glob("*.csv"):
        shutil.copy2(csv, staging / csv.name)
    old = Path.cwd()
    os.chdir(VLA_PERF_DIR / "plot_scripts")
    sys.path.insert(0, str(VLA_PERF_DIR / "plot_scripts"))
    scripts = [
        "print_0_configs_hw_models.py",
        "print_1_base_pi0.py",
        "print_2_scale_model.py",
        "print_3_long_context.py",
        "print_4_denoise_steps_action_lengths.py",
        "print_5_autoregressive_vs_diffusion.py",
        "print_6_device_vs_server.py",
        "print_7_async_inference.py",
    ]
    for s in scripts:
        logger.info("plot %s for %s", s, ckpt_out.name)
        try:
            import runpy

            runpy.run_path(s, run_name="__main__")
        except Exception:
            logger.exception("plot script failed: %s", s)
    os.chdir(old)
    src_fig = VLA_PERF_DIR / "paper_figures"
    if src_fig.is_dir():
        for p in src_fig.glob("*"):
            if p.is_file():
                shutil.copy2(p, fig_dir / p.name)
    src_tab = VLA_PERF_DIR / "paper_tables"
    tab_dir = ckpt_out / "paper_tables"
    tab_dir.mkdir(exist_ok=True)
    if src_tab.is_dir():
        for p in src_tab.glob("*"):
            if p.is_file():
                shutil.copy2(p, tab_dir / p.name)


def write_index() -> Path:
    idx = OUT_ROOT / "index.html"
    parts = ["<html><body style='background:#111;color:#eee;font-family:sans-serif'>"]
    parts.append("<h2>VLA-Perf 本地两套 SmolVLA</h2>")
    parts.append(
        "<p>解析屋顶线延迟模型，不是任务成功率。结构来自 checkpoint 的相机路数 / chunk / flow 步数。</p>"
    )
    for name in ("astra", "usb", "builtin"):
        d = OUT_ROOT / name
        if not d.is_dir():
            continue
        parts.append(f"<h3>{name}</h3><ul>")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in {".csv", ".png", ".pdf", ".json", ".html", ".log", ".tex"}:
                rel = p.relative_to(OUT_ROOT)
                parts.append(f"<li><a href='{rel}'>{rel}</a></li>")
        parts.append("</ul>")
    parts.append("</body></html>\n")
    idx.write_text("".join(parts), encoding="utf-8")
    return idx


def main() -> None:
    sys.path.insert(0, str(VLA_PERF_DIR))
    os.chdir(VLA_PERF_DIR)
    from perf_utils import setup_logging
    import pi0_perf as pi0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(OUT_ROOT / "run.log"))
    logger.info("VLA-Perf local eval start")

    errors = []
    for name, ckpt in CKPTS.items():
        if not (ckpt / "config.json").is_file():
            logger.error("missing %s", ckpt)
            continue
        try:
            out = run_ckpt(name, ckpt, pi0, logger)
            try:
                run_plots(out, logger)
            except Exception:
                logger.exception("plots failed for %s", name)
        except Exception:
            logger.exception("checkpoint %s failed", name)
            errors.append(name)

    try:
        run_builtin_scripts(logger)
    except Exception:
        logger.exception("builtin scripts failed")
        errors.append("builtin")

    idx = write_index()
    logger.info("index %s errors=%s", idx, errors)
    print("DONE", idx, "errors", errors)


if __name__ == "__main__":
    main()
