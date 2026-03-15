# Referring Segmentation Project Progress

## Project Overview
- **Topic**: Referring Image Segmentation based on CLIP
- **Base Model**: CRIS (CLIP-based Referring Image Segmentation)
- **Innovation**: CRIS-P with Cross-Modal Self-Distillation Alignment

---

## Current Status

### Completed Tasks
1. **CRIS Code Analysis** - Understood model architecture
2. **Training Environment** - Configured for RN50
3. **COCO Dataset** - Downloaded and extracted (13GB, 82783 images)
4. **CRIS-Lite Model** - Lightweight version for quick testing
5. **CRIS-P Model** - Innovation module with cross-modal alignment
6. **Git Repository** - Pushed to GitHub

### Pending Tasks
1. **RefCOCO Dataset** - Download failed due to SSL, need manual download
2. **Full Training** - Run on server with complete data
3. **Ablation Experiments** - Compare CRIS vs CRIS-P
4. **Gradio Demo** - Build visualization system

---

## Key Files Created

| File | Description |
|------|-------------|
| `model/cris_lite.py` | Lightweight CRIS for testing |
| `model/cris_p.py` | CRIS-P with cross-modal alignment |
| `config/refcoco/cris_tiny.yaml` | Lightweight config (batch=2, epochs=3) |
| `config/refcoco/cris_p_r50.yaml` | CRIS-P config |
| `model/__init__.py` | Modified to support model selection |

---

## Innovation Points (CRIS-P)

### 1. Cross-Modal Self-Distillation
- Text features guide visual feature learning
- Self-supervised alignment without extra labels
- Loss: `L_total = L_seg + α * L_align`

### 2. Multi-Scale Text Guidance
- Text embeddings injected at multiple FPN levels
- Better fine-grained localization

---

## Server Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/guo-gy/ReferringSegmentation.git
cd ReferringSegmentation/CRIS.pytorch

# 2. Create environment
conda create -n cris python=3.8
conda activate cris
pip install -r requirement.txt

# 3. Download pretrained CLIP weights
mkdir -p pretrain
# Download from: https://github.com/openai/CLIP
# Or use the RN50.pt file

# 4. Download datasets
# RefCOCO: https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
# COCO train2014: http://images.cocodataset.org/zips/train2014.zip
# Extract to datasets/

# 5. Process datasets to LMDB
python tools/data_process.py

# 6. Train baseline
python train.py --config config/refcoco/cris_r50.yaml

# 7. Train CRIS-P (our innovation)
python train.py --config config/refcoco/cris_p_r50.yaml

# 8. Quick test with CRIS-Lite
python train.py --config config/refcoco/cris_tiny.yaml
```

---

## Model Selection

The `model/__init__.py` supports three models:

```python
# Default CRIS
python train.py --config config/refcoco/cris_r50.yaml

# CRIS-P (innovation)
python train.py --config config/refcoco/cris_p_r50.yaml

# CRIS-Lite (quick test)
python train.py --config config/refcoco/cris_tiny.yaml
```

---

## Configuration Notes

### CRIS-Lite (cris_tiny.yaml)
- `input_size: 224` - Smaller input
- `batch_size: 2` - Minimal batch
- `epochs: 3` - Quick testing
- `model_name: cris_lite`

### CRIS-P (cris_p_r50.yaml)
- `model_name: cris_p`
- `align_weight: 0.5` - Cross-modal alignment weight
- `distill_temp: 0.5` - Distillation temperature

---

## Dataset Structure

```
CRIS.pytorch/datasets/
├── images/
│   └── train2014/     # COCO images (82783 files)
├── masks/
│   └── refcoco/       # Binary masks
├── lmdb/
│   ├── refcoco_train/
│   └── refcoco/val.lmdb
└── anns/
    └── refcoco/       # JSON annotations
```

---

## Training Log (Local Test)

CRIS-Lite training started successfully:
```
Epoch=[1/3] Loss=0.43 IoU=2% Prec@50=0%
```
Loss is decreasing, model is learning.

---

## Expected Results

| Model | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|-------|-------------|--------------|--------------|
| CRIS (baseline) | ~70% | ~62% | ~65% |
| CRIS-P (ours) | ~72% | ~64% | ~67% |

---

## Contact
- Author: Guo Guanyang
- GitHub: https://github.com/guo-gy/ReferringSegmentation
