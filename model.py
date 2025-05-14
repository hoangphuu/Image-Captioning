import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
from torchvision.models import EfficientNet_B0_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained EfficientNet-B0 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # Load pretrained EfficientNet
        efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        modules = list(efficientnet.children())[:-1]
        self.efficientnet = nn.Sequential(*modules)
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        self.linear = nn.Linear(efficientnet.classifier[1].in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.efficientnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        features = self.dropout(features)
        return features


class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_seq_length)
        
        # Transformer encoder layers with dropout
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=8, 
            dim_feedforward=hidden_size,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Transformer decoder layers with dropout
        decoder_layers = TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=8, 
            dim_feedforward=hidden_size,
            dropout=0.5,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        self.linear = nn.Linear(embed_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(embed_size)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = self.pos_encoder(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        
        # Create mask for padding
        mask = self.generate_square_subsequent_mask(embeddings.size(1)).to(embeddings.device)
        
        # Transformer encoding
        features = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        features = self.transformer_encoder(features)
        
        # Transformer decoding
        output = self.transformer_decoder(embeddings, features, tgt_mask=mask)
        output = self.dropout(output)
        output = self.layer_norm(output)
        output = self.linear(output)
        
        return output
    
    def sample(self, features, states=None, temperature=0.7, top_k=5):
        """Generate captions for given image features using beam search."""
        batch_size = features.size(0)
        inputs = torch.ones((batch_size, 1), dtype=torch.long).to(features.device)
        sampled_ids = []
        
        for i in range(self.max_seq_length):
            embeddings = self.embed(inputs)
            embeddings = self.pos_encoder(embeddings)
            
            # Create mask
            mask = self.generate_square_subsequent_mask(embeddings.size(1)).to(embeddings.device)
            
            # Transformer encoding
            features_expanded = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
            features_expanded = self.transformer_encoder(features_expanded)
            
            # Transformer decoding
            output = self.transformer_decoder(embeddings, features_expanded, tgt_mask=mask)
            output = self.linear(output)
            
            # Get the last word
            output = output[:, -1, :] / temperature
            
            # Mask out special tokens
            output[:, 0] = float('-inf')  # mask padding
            output[:, 1] = float('-inf')  # mask unknown
            output[:, 2] = float('-inf')  # mask <start>
            
            # Force <start> token at the beginning
            if i == 0:
                output[:, 3:] = float('-inf')  # only allow <start> token
            else:
                # Mask out tokens that appear too frequently
                if i > 0:
                    for prev_id in sampled_ids:
                        output[:, prev_id] *= 0.5  # reduce probability of repeating tokens
                    
                    # Mask out tokens that appeared in last 3 positions
                    if len(sampled_ids) >= 3:
                        for prev_id in sampled_ids[-3:]:
                            output[:, prev_id] *= 0.1  # heavily reduce probability of recent tokens
            
            # Handle inf and nan values
            output = torch.nan_to_num(output, nan=float('-inf'), posinf=float('inf'), neginf=float('-inf'))
            
            # Top-k sampling
            if top_k > 0:
                values, indices = torch.topk(output, top_k)
                # Apply softmax to get probabilities
                probs = torch.softmax(values, dim=-1)
                # Handle any remaining inf/nan values in probs
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                # Normalize probabilities to sum to 1
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Sample from the distribution
                try:
                    predicted = torch.multinomial(probs, 1).squeeze(1)
                    predicted = indices.gather(1, predicted.unsqueeze(1)).squeeze(1)
                except RuntimeError:
                    # Fallback to argmax if sampling fails
                    predicted = indices[:, 0]
            else:
                # Apply softmax to get probabilities
                probs = torch.softmax(output, dim=-1)
                # Handle any remaining inf/nan values in probs
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                # Normalize probabilities to sum to 1
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Sample from the distribution
                try:
                    predicted = torch.multinomial(probs, 1).squeeze(1)
                except RuntimeError:
                    # Fallback to argmax if sampling fails
                    predicted = torch.argmax(probs, dim=-1)
            
            sampled_ids.append(predicted)
            
            # Stop if <end> token is predicted
            if (predicted == 2).all():  # Assuming 2 is the index of <end> token
                break
                
            # Prepare next input
            inputs = torch.cat([inputs, predicted.unsqueeze(1)], dim=1)
            
        sampled_ids = torch.stack(sampled_ids, dim=1)
        return sampled_ids
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer to avoid saving in state_dict
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x should be (batch, seq_len, d_model)
        if x.dim() != 3:
            raise ValueError(f"Input to PositionalEncoding must be 3D (batch, seq_len, d_model), got {x.shape}")
        if x.size(1) > self.max_len:
            x = x[:, :self.max_len, :]
        x = x + self.pe[:, :x.size(1), :]
        return x