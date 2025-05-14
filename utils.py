import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import json
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging(log_dir, name=None):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log' if name else f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_args(args, path):
    """Save arguments to a JSON file"""
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(path):
    """Load arguments from a JSON file"""
    with open(path, 'r') as f:
        args_dict = json.load(f)
    return args_dict

def visualize_attention(image_path, caption, attention_weights, save_path=None):
    """Visualize attention weights on the image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot attention
    ax2.imshow(image)
    ax2.set_title('Attention Visualization')
    ax2.axis('off')
    
    # Add attention weights
    attention_weights = attention_weights.cpu().numpy()
    attention_weights = attention_weights / attention_weights.max()
    
    # Plot attention heatmap
    ax2.imshow(attention_weights, alpha=0.5, cmap='jet')
    
    # Add caption
    plt.suptitle(' '.join(caption), fontsize=12)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metrics(metrics_history, save_path=None):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(metrics_history['train_loss'], label='Train')
    axes[0, 0].plot(metrics_history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # Plot BLEU score
    axes[0, 1].plot(metrics_history['bleu4'])
    axes[0, 1].set_title('BLEU-4 Score')
    
    # Plot METEOR score
    axes[1, 0].plot(metrics_history['meteor'])
    axes[1, 0].set_title('METEOR Score')
    
    # Plot ROUGE-L score
    axes[1, 1].plot(metrics_history['rouge_l'])
    axes[1, 1].set_title('ROUGE-L Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_gif(image_paths, captions, save_path, duration=500):
    """Create a GIF from a sequence of images with captions"""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def analyze_vocabulary(vocab, save_path=None):
    """Analyze vocabulary statistics"""
    word_freq = vocab.word2idx
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1])
    
    # Plot word frequency distribution
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(sorted_words)), [freq for _, freq in sorted_words])
    plt.title('Word Frequency Distribution')
    plt.xlabel('Word Index')
    plt.ylabel('Frequency')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_metrics(predictions, references):
    """Compute various metrics for evaluation"""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge import Rouge
    
    # BLEU Score
    smooth_fn = SmoothingFunction().method4
    bleu4 = corpus_bleu(references, predictions, smoothing_function=smooth_fn)
    
    # METEOR Score
    meteor_scores = []
    for ref, pred in zip(references, predictions):
        meteor_scores.append(meteor_score([' '.join(ref[0])], ' '.join(pred)))
    meteor = np.mean(meteor_scores)
    
    # ROUGE Score
    rouge = Rouge()
    rouge_scores = []
    for ref, pred in zip(references, predictions):
        try:
            scores = rouge.get_scores(' '.join(pred), ' '.join(ref[0]))[0]
            rouge_scores.append(scores)
        except:
            continue
    rouge_l = np.mean([score['rouge-l']['f'] for score in rouge_scores])
    
    return {
        'bleu4': bleu4,
        'meteor': meteor,
        'rouge_l': rouge_l
    }

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics'] 