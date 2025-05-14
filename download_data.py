import os
import requests
import zipfile
from tqdm import tqdm
import shutil

def download_file(url, filename):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Create necessary directories
    os.makedirs('data/annotations', exist_ok=True)
    os.makedirs('data/train2014', exist_ok=True)
    os.makedirs('data/val2014', exist_ok=True)
    os.makedirs('data/resized2014', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Download annotations
    print("Downloading annotations...")
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    download_file(annotations_url, "annotations_trainval2014.zip")
    
    print("Extracting annotations...")
    with zipfile.ZipFile("annotations_trainval2014.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    os.remove("annotations_trainval2014.zip")

    # Download training images
    print("Downloading training images...")
    train_url = "http://images.cocodataset.org/zips/train2014.zip"
    download_file(train_url, "train2014.zip")
    
    print("Extracting training images...")
    with zipfile.ZipFile("train2014.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    os.remove("train2014.zip")

    # Download validation images
    print("Downloading validation images...")
    val_url = "http://images.cocodataset.org/zips/val2014.zip"
    download_file(val_url, "val2014.zip")
    
    print("Extracting validation images...")
    with zipfile.ZipFile("val2014.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    os.remove("val2014.zip")

    # Create sample annotations
    print("Creating sample annotations...")
    import json
    
    def create_sample_annotations(input_file, output_file, num_images=1000):
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        image_ids = set()
        for ann in data['annotations']:
            image_ids.add(ann['image_id'])
        
        sample_image_ids = list(image_ids)[:num_images]
        
        filtered_annotations = []
        for ann in data['annotations']:
            if ann['image_id'] in sample_image_ids:
                filtered_annotations.append(ann)
        
        data['annotations'] = filtered_annotations
        with open(output_file, 'w') as f:
            json.dump(data, f)
        
        print(f"Created sample annotation file with {len(sample_image_ids)} images")
        print(f"Total annotations: {len(filtered_annotations)}")

    create_sample_annotations(
        'data/annotations/captions_train2014.json',
        'data/annotations/captions_train2014_sample.json',
        num_images=1000
    )

    print("\nDataset preparation completed!")
    print("\nNext steps:")
    print("1. Run resize.py to resize images:")
    print("   python resize.py --input_dir data/train2014 --output_dir data/resized2014 --image_size 256")
    print("\n2. Build vocabulary:")
    print("   python build_vocab.py --caption_path data/annotations/captions_train2014_sample.json --vocab_path data/vocab.pkl --threshold 2")
    print("\n3. Start training:")
    print("   python train.py --image_dir data/resized2014 --caption_path data/annotations/captions_train2014_sample.json --val_caption_path data/annotations/captions_val2014.json --vocab_path data/vocab.pkl --model_path models/ --num_epochs 5 --batch_size 32 --learning_rate 0.001")

if __name__ == '__main__':
    main() 