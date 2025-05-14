import os
import argparse
from PIL import Image
from tqdm import tqdm

def resize_images(input_dir, output_dir, image_size):
    """Resize all images in input_dir and save to output_dir."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images to resize")
    
    # Resize each image
    for image_file in tqdm(image_files, desc="Resizing images"):
        try:
            # Open image
            with Image.open(os.path.join(input_dir, image_file)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
                # Save resized image
                output_path = os.path.join(output_dir, image_file)
                img.save(output_path, 'JPEG', quality=95)
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for resized images')
    parser.add_argument('--image_size', type=int, default=256, help='size to resize images to')
    args = parser.parse_args()
    
    print(f"Resizing images from {args.input_dir} to {args.output_dir}")
    print(f"Target size: {args.image_size}x{args.image_size}")
    
    resize_images(args.input_dir, args.output_dir, args.image_size)
    
    print("\nResizing completed!")

if __name__ == '__main__':
    main()