# RM65 cleaned LeRobot v2.1 datasets (SFT-ready)

Four private Hugging Face dataset repos. Upload **each subdirectory**, not this parent folder.

| folder | episodes | frames | fps | cameras |
|---|---:|---:|---:|---|
| `rm65_stacking_blocks` | 52 | 32749 | 20 | front, astra |
| `rm65_cleanup_the_desk` | 52 | 45717 | 20 | front, astra |
| `rm65_place_to_slot` | 52 | 73839 | 20 | front, astra |
| `rm65_usb_insert` | 200 | 94944 | 20 | front, mono (mono lagged +10 frames vs joints/front) |

Language: each parquet frame has `task_index` → `meta/tasks.jsonl` string `{overall} {subtask}`.
USB episode 4 is `failed` and must be excluded from SFT (`splits.train` is `0:4,5:200`).

Action / state: 6 arm joints in degrees + gripper 0–1.

Upload (after you confirm):

```bash
DATASET_ROOT=/home/sdznkj/zkx_ws/sendlerobot/datasets/cleaned \
  python3 scripts/hf_upload_small_tasks.py
```
