import torch
import argparse
import pickle
import os
from PIL import Image
import torchvision.transforms as transforms
from model import EncoderCNN, DecoderTransformer
from build_vocab import Vocabulary
import matplotlib.pyplot as plt

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()
    decoder = DecoderTransformer(args.embed_size, args.hidden_size,
                                len(vocab), args.num_layers).eval()
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.unsqueeze(0).to(args.device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
    
    # Display the image and caption
    image = Image.open(args.image)
    plt.imshow(image)
    plt.axis('off')
    plt.title(sentence)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as parameters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)
