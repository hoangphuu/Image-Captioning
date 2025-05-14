import argparse
import json
import pickle
import os
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_file, threshold):
    """Build a simple vocabulary wrapper."""
    coco = json.load(open(json_file, 'r'))
    counter = Counter()
    
    # Add validation captions to vocabulary
    for ann in coco['annotations']:
        caption = str(ann['caption']).lower()
        tokens = word_tokenize(caption)
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens
    vocab = Vocabulary()

    # Add the words to the vocabulary
    for word in words:
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json_file=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)