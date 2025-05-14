# Image Captioning with Transformer

A deep learning project for generating natural language descriptions of images using Transformer architecture.

## Features

- **Model Architecture**:
  - Encoder: EfficientNet-B0 for image feature extraction
  - Decoder: Transformer-based architecture for caption generation
  - Attention mechanism for focusing on relevant image regions

- **Training Features**:
  - Data augmentation (rotation, flip, color jitter, etc.)
  - Label smoothing
  - Gradient clipping
  - Learning rate scheduling with warmup
  - Early stopping
  - Model checkpointing
  - TensorBoard logging

- **Evaluation Metrics**:
  - BLEU-4 Score
  - METEOR Score
  - ROUGE-L Score
  - Attention visualization
  - Caption length statistics

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/
│   ├── resized2014/          # Resized images
│   └── annotations/          # COCO annotations
├── models/                   # Saved models
├── logs/                    # Training logs
├── evaluation_results/      # Evaluation outputs
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── model.py               # Model architecture
├── data_loader.py         # Data loading utilities
├── build_vocab.py         # Vocabulary building
├── utils.py              # Utility functions
└── requirements.txt      # Project dependencies
```

## Usage

### 1. Prepare Data

1. Download COCO dataset:
```bash
python download_data.py
```

2. Build vocabulary:
```bash
python build_vocab.py --caption_path data/annotations/captions_train2014.json --vocab_path data/vocab.pkl --threshold 5
```

### 2. Training

Train the model with default parameters:
```bash
python train.py --model_path models/ --num_epochs 10 --batch_size 32
```

Additional training options:
```bash
python train.py \
    --model_path models/ \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --embed_size 256 \
    --hidden_size 512 \
    --num_layers 1 \
    --patience 3 \
    --sample_size 1000
```

### 3. Evaluation

Evaluate the model on test images:
```bash
python evaluate.py \
    --model_path models/best_model/model.ckpt \
    --test_image_dir test_images/ \
    --output_dir evaluation_results/ \
    --visualize
```

### 4. Visualization

- Training metrics are logged to TensorBoard:
```bash
tensorboard --logdir models/logs
```

- Attention visualizations are saved in the evaluation output directory
- Training progress is logged to `logs/training_*.log`

## Model Parameters

- **Encoder**:
  - EfficientNet-B0 backbone
  - Feature dimension: 1280
  - Output dimension: 256 (embed_size)

- **Decoder**:
  - Transformer architecture
  - Embedding dimension: 256
  - Hidden dimension: 512
  - Number of layers: 1-6
  - Number of attention heads: 8
  - Dropout: 0.1 (encoder), 0.5 (decoder)

## Training Parameters

- Batch size: 32-128
- Learning rate: 0.001
- Optimizer: AdamW
- Weight decay: 0.01
- Label smoothing: 0.1
- Gradient clipping: 1.0
- Early stopping patience: 3
- Temperature: 0.7
- Top-k sampling: 5

## Results

The model achieves the following metrics on the COCO validation set:
- BLEU-4: ~0.35
- METEOR: ~0.25
- ROUGE-L: ~0.50


