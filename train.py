import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import signal
import sys
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderTransformer
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import logging
from datetime import datetime
nltk.download('punkt')
nltk.download('wordnet')

# Setup logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for signal handling
current_encoder = None
current_decoder = None
current_epoch = 0
current_step = 0
args = None

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to save model before exiting."""
    print('\nSaving model before exit...')
    if current_encoder is not None and current_decoder is not None:
        torch.save(current_encoder.state_dict(), os.path.join(
            args.model_path, f'encoder-interrupted-{current_epoch}-{current_step}.ckpt'))
        torch.save(current_decoder.state_dict(), os.path.join(
            args.model_path, f'decoder-interrupted-{current_epoch}-{current_step}.ckpt'))
        print('Model saved successfully!')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def calculate_metrics(references, hypotheses):
    """Calculate various metrics for evaluation."""
    # BLEU Score
    smooth_fn = SmoothingFunction().method4
    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn)
    
    # METEOR Score
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        meteor_scores.append(meteor_score([' '.join(ref[0])], ' '.join(hyp)))
    meteor = np.mean(meteor_scores)
    
    # ROUGE Score
    rouge = Rouge()
    rouge_scores = []
    for ref, hyp in zip(references, hypotheses):
        try:
            scores = rouge.get_scores(' '.join(hyp), ' '.join(ref[0]))[0]
            rouge_scores.append(scores)
        except:
            continue
    rouge_l = np.mean([score['rouge-l']['f'] for score in rouge_scores])
    
    # Calculate average lengths
    avg_hyp_len = np.mean([len(h) for h in hypotheses])
    avg_ref_len = np.mean([len(r[0]) for r in references])
    
    return {
        'bleu4': bleu4,
        'meteor': meteor,
        'rouge_l': rouge_l,
        'avg_hyp_len': avg_hyp_len,
        'avg_ref_len': avg_ref_len
    }

def evaluate(encoder, decoder, data_loader, criterion, vocab, device, logger):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_step = len(data_loader)
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            
            # Calculate loss
            seq_len = min(outputs.size(1), captions.size(1))
            outputs = outputs[:, :seq_len, :]
            captions = captions[:, :seq_len]
            
            outputs = outputs.reshape(-1, outputs.size(-1))
            captions_flat = captions.reshape(-1)
            
            mask = (captions_flat != 0).float()
            
            # Label smoothing
            smooth = 0.1
            n_classes = outputs.size(-1)
            one_hot = torch.zeros_like(outputs).scatter(1, captions_flat.unsqueeze(1), 1)
            smooth_one_hot = one_hot * (1 - smooth) + (smooth / n_classes)
            
            log_probs = F.log_softmax(outputs, dim=-1)
            loss = -(smooth_one_hot * log_probs).sum(dim=-1)
            
            if i > 0:
                prev_tokens = captions[:, :i]
                curr_tokens = captions[:, i:i+1]
                repeat_mask = (prev_tokens == curr_tokens).any(dim=1).float()
                loss = loss * (1 + repeat_mask)
            
            loss = (loss * mask).sum() / (mask.sum() + 1e-9)
            total_loss += loss.item()
            
            # Generate captions for evaluation
            sampled_ids = decoder.sample(features, temperature=0.7, top_k=5)
            sampled_ids = sampled_ids.cpu().numpy()
            
            # Convert word ids to words
            for j in range(sampled_ids.shape[0]):
                sampled_caption = []
                for word_id in sampled_ids[j]:
                    word = vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    sampled_caption.append(word)
                hypotheses.append(sampled_caption)
                
                ref_caption = []
                for word_id in captions[j].cpu().numpy():
                    word = vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    ref_caption.append(word)
                references.append([ref_caption])
                
                # Log examples
                if i == 0 and j < 3:
                    logger.info(f"\nExample {j+1}")
                    logger.info(f"Generated: {' '.join(sampled_caption)}")
                    logger.info(f"Reference: {' '.join(ref_caption)}")
    
    # Calculate metrics
    metrics = calculate_metrics(references, hypotheses)
    
    # Log metrics
    logger.info("\nValidation Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    return total_loss / total_step, metrics

def main(args):
    global current_encoder, current_decoder, current_epoch, current_step
    
    # Setup logging
    logger = setup_logging(os.path.join(args.model_path, 'logs'))
    logger.info("Starting training with arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.model_path, 'logs'))
    
    # Image preprocessing with augmentation
    transform = transforms.Compose([ 
        transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loaders
    train_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             sample_size=args.sample_size)
    
    val_image_dir = args.image_dir.replace('train2014', 'val2014')
    val_loader = get_loader(val_image_dir, args.val_caption_path, vocab,
                           transform, args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           sample_size=args.sample_size)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderTransformer(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Set global variables for signal handling
    current_encoder = encoder
    current_decoder = decoder
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training
    total_step = len(train_loader)
    best_val_loss = float('inf')
    patience_counter = 0
    best_metrics = None
    
    try:
        for epoch in range(args.num_epochs):
            current_epoch = epoch
            # Training phase
            encoder.train()
            decoder.train()
            total_loss = 0
            
            for i, (images, captions, lengths) in enumerate(train_loader):
                current_step = i
                images = images.to(device)
                captions = captions.to(device)
                
                # Forward, backward and optimize
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                
                # Calculate loss
                seq_len = min(outputs.size(1), captions.size(1))
                outputs = outputs[:, :seq_len, :]
                captions = captions[:, :seq_len]
                
                outputs = outputs.reshape(-1, outputs.size(-1))
                captions_flat = captions.reshape(-1)
                
                mask = (captions_flat != 0).float()
                
                # Label smoothing
                smooth = 0.1
                n_classes = outputs.size(-1)
                one_hot = torch.zeros_like(outputs).scatter(1, captions_flat.unsqueeze(1), 1)
                smooth_one_hot = one_hot * (1 - smooth) + (smooth / n_classes)
                
                log_probs = F.log_softmax(outputs, dim=-1)
                loss = -(smooth_one_hot * log_probs).sum(dim=-1)
                
                if i > 0:
                    prev_tokens = captions[:, :i]
                    curr_tokens = captions[:, i:i+1]
                    repeat_mask = (prev_tokens == curr_tokens).any(dim=1).float()
                    loss = loss * (1 + repeat_mask)
                
                loss = (loss * mask).sum() / (mask.sum() + 1e-9)
                
                decoder.zero_grad()
                encoder.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                # Print log info
                if i % args.log_step == 0:
                    logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, LR: {:.6f}'
                          .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()), scheduler.get_last_lr()[0]))
                    
                    # Log to tensorboard
                    writer.add_scalar('Training/Loss', loss.item(), epoch * total_step + i)
                    writer.add_scalar('Training/Perplexity', np.exp(loss.item()), epoch * total_step + i)
                    writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], epoch * total_step + i)
                    
                # Save the model checkpoints
                if (i+1) % args.save_step == 0:
                    checkpoint_path = os.path.join(args.model_path, f'checkpoint-{epoch+1}-{i+1}')
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }, os.path.join(checkpoint_path, 'model.ckpt'))
                    
                    logger.info(f'Saved checkpoint to {checkpoint_path}')
            
            # Validation phase
            val_loss, metrics = evaluate(encoder, decoder, val_loader, criterion, vocab, device, logger)
            logger.info('Validation Loss: {:.4f}'.format(val_loss))
            
            # Log validation metrics to tensorboard
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            for metric_name, value in metrics.items():
                writer.add_scalar(f'Validation/{metric_name}', value, epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(args.model_path, 'best_model')
                os.makedirs(best_model_path, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics,
                }, os.path.join(best_model_path, 'model.ckpt'))
                
                logger.info(f'Saved best model to {best_model_path}')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info('Early stopping triggered')
                    break
    
    except KeyboardInterrupt:
        logger.info('\nTraining interrupted by user')
        # Model will be saved by signal handler
    finally:
        writer.close()
        logger.info('Training finished!')
        if best_metrics:
            logger.info("\nBest validation metrics:")
            for metric_name, value in best_metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str, default='data/annotations/captions_val2014.json', help='path for validation annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping threshold')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--sample_size', type=int, default=None, help='number of samples to use for training/validation')
    args = parser.parse_args()
    print(args)
    main(args)