# rm65-smolvla 命名对照（GitHub ↔ Hugging Face）

| 内容 | 平台 | URL |
|------|------|-----|
| 代码 + ckpt 元数据 + edge_patch | GitHub | https://github.com/theblock1226/rm65-smolvla |
| 同上镜像（GH 不通时） | HF Dataset | https://huggingface.co/datasets/FelishLeon/rm65-smolvla-code |
| 离线评测图 / attn / live | HF Dataset | https://huggingface.co/datasets/FelishLeon/rm65-smolvla-offline-eval |
| VLM / RLT / fp16·backup / 真机 DAgger | HF Dataset | https://huggingface.co/datasets/FelishLeon/rm65-smolvla-extras |
| 叠块权重 | HF Model | https://huggingface.co/FelishLeon/smolvla_rm65_stack_30ep_s1000 |
| USB 权重 | HF Model | https://huggingface.co/FelishLeon/smolvla_rm65_usb_100ep_s40000 |
| 清理权重 | HF Model | https://huggingface.co/FelishLeon/smolvla_rm65_cleanup_47ep_s40000 |
| 放置权重 | HF Model | https://huggingface.co/FelishLeon/smolvla_rm65_place_47ep_s40000 |
| 清洗训练数据 ×4 | HF Dataset | `FelishLeon/rm65_{stacking_blocks,usb_insert,cleanup_the_desk,place_to_slot}` |

本地目录名与 HF model/dataset 名对齐（去掉 `FelishLeon/` 前缀）。

国内可用：`export HF_ENDPOINT=https://hf-mirror.com`
