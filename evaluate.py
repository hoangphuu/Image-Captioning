import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from model import EncoderCNN, DecoderTransformer
from utils import setup_logging, compute_metrics, visualize_attention
import os
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.image_files[idx]

def main(args):
    # Setup logging
    logger = setup_logging(args.log_dir, 'evaluation')
    logger.info("Starting evaluation with arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderTransformer(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Load the trained model parameters
    checkpoint = torch.load(args.model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    
    # Create test dataset and dataloader
    test_dataset = TestDataset(args.test_image_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate captions
    results = []
    for images, image_files in tqdm(test_loader, desc='Generating captions'):
        images = images.to(device)
        
        # Generate captions
        with torch.no_grad():
            features = encoder(images)
            sampled_ids = decoder.sample(features, temperature=args.temperature, top_k=args.top_k)
            sampled_ids = sampled_ids.cpu().numpy()
        
        # Convert word ids to words
        for i, (image_file, sampled_id) in enumerate(zip(image_files, sampled_ids)):
            sampled_caption = []
            for word_id in sampled_id:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                sampled_caption.append(word)
            
            # Save result
            result = {
                'image_file': image_file,
                'caption': ' '.join(sampled_caption)
            }
            results.append(result)
            
            # Save visualization if requested
            if args.visualize:
                image_path = os.path.join(args.test_image_dir, image_file)
                attention_weights = decoder.get_attention_weights(features[i:i+1], sampled_ids[i:i+1])
                save_path = os.path.join(args.output_dir, f'attention_{image_file}.png')
                visualize_attention(image_path, sampled_caption, attention_weights, save_path)
    
    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path for trained model')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--test_image_dir', type=str, required=True, help='directory for test images')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='directory for saving results')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory for saving logs')
    parser.add_argument('--crop_size', type=int, default=224, help='size for cropping images')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.7, help='temperature for sampling')
    parser.add_argument('--top_k', type=int, default=5, help='top k for sampling')
    parser.add_argument('--visualize', action='store_true', help='visualize attention weights')
    args = parser.parse_args()
    main(args) 