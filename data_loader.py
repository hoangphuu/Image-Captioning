import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import json
from torch.utils.data import Dataset
import torch.nn.functional as F


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json_file, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json_file: annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.vocab = vocab
        self.transform = transform
        
        # Load annotations
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        
        # Create image ID to caption mapping
        self.id_to_caption = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_caption:
                self.id_to_caption[img_id] = ann['caption']
        
        # Get unique image IDs
        self.ids = list(self.id_to_caption.keys())
        
        # Create image ID to filename mapping
        self.id_to_filename = {}
        for img in self.coco['images']:
            self.id_to_filename[img['id']] = img['file_name']

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        img_id = self.ids[index]
        caption = self.id_to_caption[img_id]
        
        # Get image filename
        filename = self.id_to_filename[img_id]
        image = Image.open(os.path.join(self.root, filename)).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, len(caption)

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, length).
    
    Args:
        data: list of tuple (image, caption, length). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
            - length: int; valid length for each caption.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[2], reverse=True)
    images, captions, lengths = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return images, targets, lengths

def get_loader(root, json_file, vocab, transform, batch_size, shuffle, num_workers, sample_size=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json_file=json_file,
                       vocab=vocab,
                       transform=transform)
    
    # Print caption length statistics
    lengths = []
    for i in range(len(coco)):
        _, _, length = coco[i]
        lengths.append(length)
    print(f"Caption length statistics:")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Average length: {sum(lengths)/len(lengths):.2f}")
    
    # If sample_size is provided, only use a subset of the data
    if sample_size is not None:
        indices = torch.randperm(len(coco))[:sample_size]
        coco = torch.utils.data.Subset(coco, indices)
        print(f"Using {sample_size} samples for training/validation")
    
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn)
    return data_loader