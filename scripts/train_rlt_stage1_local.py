#!/usr/bin/env python3
"""Stage 1 (local fork): Offline RLT encoder-decoder training.

Supports:
  - Pure local cleaned/ datasets (no HuggingFace Hub)
  - 7-DoF RM65 defaults (6 joints + gripper)
  - ``--mock_vla`` dry-run that skips loading SmolVLA (for CI / pre-ckpt)

Real training (needs matching RM65 SmolVLA)::

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      PYTHONPATH=. python3 scripts/train_rlt_stage1_local.py \\
        --smolvla_path /path/to/rm65_smolvla \\
        --dataset_root datasets/cleaned/rm65_stacking_blocks \\
        --steps 500 --batch_size 4 --device cuda'

Mock dry-run (no VLA ckpt)::

    PYTHONPATH=. python3 scripts/train_rlt_stage1_local.py \\
      --mock_vla --dataset_root datasets/cleaned/rm65_stacking_blocks \\
      --episodes 0 --steps 8 --batch_size 2 --device cpu \\
      --output_dir checkpoints/rlt_stage1_mock
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="RLT Stage 1 (local fork)")
    p.add_argument("--smolvla_path", type=str, default="", help="Frozen SmolVLA ckpt (required unless --mock_vla)")
    p.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Local LeRobot dataset directory (contains meta/, data/, videos/)",
    )
    p.add_argument(
        "--episodes",
        type=str,
        default="",
        help="Comma-separated episode indices to load (empty = all). Use 0 for smoke.",
    )
    p.add_argument("--output_dir", type=str, default="checkpoints/rlt_stage1")
    p.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Stage1 checkpoint to continue from (encoder + decoder weights)",
    )
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--state_dim", type=int, default=7)
    p.add_argument(
        "--task_family",
        choices=["usb", "astra", "desk", "slot", "stack"],
        default="",
        help="usb→usb_smolvla; desk/slot/stack/astra→astra_smolvla",
    )
    p.add_argument("--vlm_hidden_dim", type=int, default=0, help="0 = config default 960")
    p.add_argument("--expert_hidden_dim", type=int, default=0, help="0 = config default 720")
    p.add_argument(
        "--mock_vla",
        action="store_true",
        help="Do not load SmolVLA; synthesize embeddings for pipeline dry-run",
    )
    p.add_argument("--vlm_seq_len", type=int, default=64, help="Mock VLM token length")
    p.add_argument("--expert_seq_len", type=int, default=50, help="Mock expert token length")
    return p.parse_args()


class MockVLAExtractor:
    """Stand-in for frozen SmolVLA extract_embeddings (Stage1 plumbing only)."""

    def __init__(self, cfg, device: torch.device, vlm_len: int, expert_len: int, seed: int = 0):
        self.cfg = cfg
        self.device = device
        self.vlm_len = vlm_len
        self.expert_len = expert_len
        self.rng = np.random.default_rng(seed)

    def extract_from_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = batch["observation.state"].shape[0]
        # Condition lightly on state so loss is non-trivial / reproducible-ish
        state = batch["observation.state"].detach().float()
        base = state.mean(dim=-1, keepdim=True)  # (B,1)
        vlm = torch.randn(
            bsz, self.vlm_len, self.cfg.vlm_hidden_dim, device=self.device, dtype=torch.float32
        )
        expert = torch.randn(
            bsz, self.expert_len, self.cfg.expert_hidden_dim, device=self.device, dtype=torch.float32
        )
        vlm = vlm + base.unsqueeze(-1) * 0.1
        expert = expert + base.unsqueeze(-1) * 0.1
        return vlm, expert


def alias_images_for_policy(batch: dict, image_features) -> list[str]:
    """Copy dataset camera keys onto the names stored in the SmolVLA config.

    USB demos use ``front``/``mono``. The 100-episode checkpoint expects
    ``camera1``/``camera2`` (same rename as its preprocessor).
    """
    from lerobot.common.policies.smolvla_rlt.robot_adapter import CAMERA_KEY_ALIASES

    applied: list[str] = []
    for key in image_features:
        if key in batch:
            continue
        short = str(key).split(".")[-1]
        for alias in CAMERA_KEY_ALIASES.get(short, (short,)):
            src = f"observation.images.{alias}"
            if src in batch and src != key:
                batch[key] = batch[src]
                applied.append(f"{src} -> {key}")
                break
    return applied


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from lerobot.common.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.common.policies.smolvla_rlt.local_dataset import load_local_lerobot_dataset
    from lerobot.common.policies.smolvla_rlt.modeling_smolvla_rlt import RLTokenDecoder, RLTokenEncoder
    from lerobot.common.policies.smolvla_rlt.robot_adapter import resolve_smolvla_family

    smolvla_path = args.smolvla_path
    family = "astra"
    if args.task_family:
        family = resolve_smolvla_family(args.task_family)
    if not smolvla_path and not args.mock_vla:
        smolvla_path = "usb_smolvla" if family == "usb" else "astra_smolvla"
        logger.info("task_family=%s → smolvla_path=%s", args.task_family or family, smolvla_path)
        args.smolvla_path = smolvla_path
    if not args.mock_vla and not args.smolvla_path:
        raise SystemExit("--smolvla_path or --task_family is required unless --mock_vla")

    rlt_kwargs = dict(
        mode="rlt_training",
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        smolvla_pretrained_path=args.smolvla_path or "",
        smolvla_family=family,
        camera_set="usb" if family == "usb" else "stack",
    )
    if args.vlm_hidden_dim > 0:
        rlt_kwargs["vlm_hidden_dim"] = args.vlm_hidden_dim
    if args.expert_hidden_dim > 0:
        rlt_kwargs["expert_hidden_dim"] = args.expert_hidden_dim
    rlt_config = SmolVLARLTConfig(**rlt_kwargs)
    logger.info(
        "RLT dims vlm=%d expert=%d action=%d state=%d cams=%s family=%s",
        rlt_config.vlm_hidden_dim,
        rlt_config.expert_hidden_dim,
        rlt_config.action_dim,
        rlt_config.state_dim,
        rlt_config.camera_names(),
        rlt_config.smolvla_family,
    )
    encoder = RLTokenEncoder(rlt_config).to(device)
    decoder = RLTokenDecoder(rlt_config).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + float(torch.cos(torch.tensor(progress * 3.14159265))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    ep_list = None
    if args.episodes.strip():
        ep_list = [int(x) for x in args.episodes.split(",") if x.strip() != ""]

    action_deltas = None
    if not args.mock_vla:
        import json

        from lerobot.configs.policies import PreTrainedConfig

        info = json.loads((Path(args.dataset_root) / "meta" / "info.json").read_text())
        vla_cfg_preview = PreTrainedConfig.from_pretrained(args.smolvla_path)
        fps = float(info["fps"])
        action_deltas = {"action": [i / fps for i in range(int(vla_cfg_preview.chunk_size))]}
        logger.info(
            "Action chunk deltas: %d steps at %.1f fps",
            vla_cfg_preview.chunk_size,
            fps,
        )

    logger.info("Loading local dataset from %s (episodes=%s)", args.dataset_root, ep_list)
    dataset = load_local_lerobot_dataset(
        args.dataset_root,
        episodes=ep_list,
        download_videos=False,
        delta_timestamps=action_deltas,
    )
    sample0 = dataset[0]
    logger.info(
        "Dataset samples=%d keys=%s state_shape=%s action_shape=%s",
        len(dataset),
        sorted(k for k in sample0.keys() if not k.startswith("_")),
        tuple(sample0["observation.state"].shape),
        tuple(sample0["action"].shape),
    )
    state_dim_data = int(sample0["observation.state"].shape[-1])
    action_dim_data = int(sample0["action"].shape[-1])
    if state_dim_data != args.state_dim or action_dim_data != args.action_dim:
        logger.warning(
            "Config state/action dim (%d/%d) != data (%d/%d) — using config for RLT; "
            "ensure frozen VLA matches data when not --mock_vla",
            args.state_dim,
            args.action_dim,
            state_dim_data,
            action_dim_data,
        )

    img_keys = [k for k in sample0.keys() if k.startswith("observation.images.")]
    logger.info("Camera keys: %s", img_keys)

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size,
    )
    if len(loader) == 0:
        raise RuntimeError("Empty dataloader — reduce batch_size or load more episodes")
    data_iter = iter(loader)

    smolvla_policy = None
    vla_model = None
    mock_vla = None
    if args.mock_vla:
        logger.info("Using MockVLAExtractor (no SmolVLA weights)")
        mock_vla = MockVLAExtractor(
            rlt_config, device, args.vlm_seq_len, args.expert_seq_len, seed=args.seed
        )
    else:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        logger.info("Loading SmolVLA from %s...", args.smolvla_path)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.scripts.server.rm65_ckpt import load_dataset_stats_from_ckpt, trim_stats_to_config

        vla_config = PreTrainedConfig.from_pretrained(args.smolvla_path)
        dataset_stats = trim_stats_to_config(load_dataset_stats_from_ckpt(args.smolvla_path), vla_config)
        if not dataset_stats:
            raise SystemExit(f"No normalizer stats in {args.smolvla_path}")
        smolvla_policy = SmolVLAPolicy.from_pretrained(
            args.smolvla_path, config=vla_config, dataset_stats=dataset_stats
        )
        smolvla_policy.to(device)
        smolvla_policy.eval()
        for p in smolvla_policy.parameters():
            p.requires_grad = False
        vla_model = smolvla_policy.model
        logger.info("SmolVLA frozen.")

    best_loss = float("inf")
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        encoder.load_state_dict(init_ckpt["encoder_state_dict"])
        decoder.load_state_dict(init_ckpt["decoder_state_dict"])
        if "loss" in init_ckpt:
            best_loss = float(init_ckpt["loss"])
        logger.info(
            "Continuing from %s (stored loss=%s)",
            args.init_checkpoint,
            init_ckpt.get("loss"),
        )
    history = []
    t0 = time.time()
    encoder.train()
    decoder.train()

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "task" not in batch:
            batch["task"] = ["complete the manipulation task"] * batch["observation.state"].shape[0]

        if mock_vla is not None:
            vlm_emb, expert_emb = mock_vla.extract_from_batch(batch)
        else:
            assert smolvla_policy is not None and vla_model is not None
            applied = alias_images_for_policy(batch, smolvla_policy.config.image_features)
            if applied and step == 1:
                logger.info("Aliased images: %s", ", ".join(applied))
            batch = smolvla_policy.normalize_inputs(batch)
            batch = smolvla_policy.normalize_targets(batch)
            images, img_masks = smolvla_policy.prepare_images(batch)
            state = smolvla_policy.prepare_state(batch)
            actions = smolvla_policy.prepare_action(batch)
            lang_tokens, lang_masks = smolvla_policy.prepare_language(batch)
            with torch.no_grad():
                vlm_emb, expert_emb = vla_model.extract_embeddings(
                    images, img_masks, lang_tokens, lang_masks, state, actions
                )
            if step == 1:
                logger.info(
                    "embeddings vlm=%s expert=%s dtype=%s",
                    tuple(vlm_emb.shape),
                    tuple(expert_emb.shape),
                    vlm_emb.dtype,
                )
            vlm_emb = vlm_emb.float()
            expert_emb = expert_emb.float()

        z_rl = encoder(vlm_emb.detach(), expert_emb.detach())
        vlm_recon, expert_recon = decoder(
            z_rl,
            vlm_target_len=vlm_emb.shape[1],
            expert_target_len=expert_emb.shape[1],
        )
        vlm_loss = F.mse_loss(vlm_recon, vlm_emb.detach())
        expert_loss = F.mse_loss(expert_recon, expert_emb.detach())
        loss = vlm_loss + expert_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.detach().item())
        history.append(loss_val)
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            logger.info(
                "step %d/%d loss=%.4f vlm=%.4f expert=%.4f lr=%.2e",
                step,
                args.steps,
                loss_val,
                float(vlm_loss.item()),
                float(expert_loss.item()),
                scheduler.get_last_lr()[0],
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "config": {
                    "rlt_hidden_dim": rlt_config.rlt_hidden_dim,
                    "vlm_hidden_dim": rlt_config.vlm_hidden_dim,
                    "expert_hidden_dim": rlt_config.expert_hidden_dim,
                    "action_dim": rlt_config.action_dim,
                    "state_dim": rlt_config.state_dim,
                    "smolvla_path": args.smolvla_path,
                    "mock_vla": bool(args.mock_vla),
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                },
                "loss": loss_val,
            }
            torch.save(ckpt, output_dir / f"checkpoint_step{step}.pt")
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(ckpt, output_dir / "best_checkpoint.pt")
                logger.info("best loss -> %.4f", best_loss)

    report = {
        "steps": args.steps,
        "best_loss": best_loss,
        "mock_vla": bool(args.mock_vla),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "n_samples": len(dataset),
        "image_keys": img_keys,
        "state_dim_data": state_dim_data,
        "action_dim_data": action_dim_data,
        "pass_offline_load": True,
        "pass_train_loop": len(history) == args.steps and best_loss < float("inf"),
    }
    report["pass"] = report["pass_offline_load"] and report["pass_train_loop"]
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(history, f)
    with open(output_dir / "stage1_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("done in %.1fs best=%.4f -> %s", time.time() - t0, best_loss, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
