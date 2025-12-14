import sys
import os
import torch
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import SemanticDrawFreqSplitPipeline

def test_freq_split():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize pipeline
    pipe = SemanticDrawFreqSplitPipeline(device, sd_version='1.5')

    # Define prompts and masks
    prompts = ["a blue sky", "a rocky mountain"]
    
    # Create simple dummy masks (left half sky, right half mountain)
    H, W = 512, 512
    mask1 = Image.new('L', (W, H), 0)
    # Sky on left
    for x in range(W // 2):
        for y in range(H):
            mask1.putpixel((x, y), 255)
            
    mask2 = Image.new('L', (W, H), 0)
    # Mountain on right
    for x in range(W // 2, W):
        for y in range(H):
            mask2.putpixel((x, y), 255)

    masks = [mask1, mask2]

    # Run with Frequency Split
    print("Generating with Frequency Split...")
    img_split = pipe(
        prompts=prompts,
        masks=masks,
        height=H,
        width=W,
        num_inference_steps=20, # Fast test
        use_freq_split=True,
        freq_split_std=3.0
    )
    img_split.save("test_freq_split_on.png")
    print("Saved test_freq_split_on.png")

    # Run without Frequency Split (for sanity check of new class)
    print("Generating without Frequency Split...")
    img_std = pipe(
        prompts=prompts,
        masks=masks,
        height=H,
        width=W,
        num_inference_steps=20,
        use_freq_split=False
    )
    img_std.save("test_freq_split_off.png")
    print("Saved test_freq_split_off.png")

if __name__ == "__main__":
    test_freq_split()
