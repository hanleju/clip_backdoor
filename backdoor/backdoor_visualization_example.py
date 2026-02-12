"""
Backdoor Visualization Example

This script demonstrates how backdoor triggers are applied to images
using the EXACT SAME preprocessing pipeline as train.py, test.py, and zeroshot_test.py.

Key steps:
1. Apply same CLIP transforms (Resize, ToTensor, Normalize)
2. Use PoisonedDataset with normalize_transform parameter
3. Visualize original vs poisoned images
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from backdoor.utils import PoisonedDataset

# CLIP normalization constants (same as train.py, test.py, zeroshot_test.py)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def denormalize(tensor, mean=CLIP_MEAN, std=CLIP_STD):
    """
    Denormalize a tensor image for visualization
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)


def load_dataset_with_clip_transform(dataset_name='cifar10'):
    """
    Load dataset with EXACT SAME transform as train.py, test.py, zeroshot_test.py
    
    This ensures the preprocessing pipeline is identical to training/testing.
    """
    # CLIP transform - EXACT SAME as train.py/test.py/zeroshot_test.py
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='../data', 
            train=True, 
            download=True, 
            transform=transform
        )
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='../data', 
            train=True, 
            download=True, 
            transform=transform
        )
        class_names = [f'class_{i}' for i in range(100)]
    elif dataset_name == 'svhn':
        dataset = torchvision.datasets.SVHN(
            root='../data', 
            split='train', 
            download=True, 
            transform=transform
        )
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset, class_names


def create_poisoned_dataset(dataset, trigger_path, target_class=0, mode='test'):
    """
    Create poisoned dataset using EXACT SAME method as train.py/test.py/zeroshot_test.py
    
    Key: Pass normalize_transform to PoisonedDataset so it can:
    1. Denormalize the input
    2. Add trigger
    3. Re-normalize the output
    
    Args:
        dataset: Original dataset (already has CLIP transform applied)
        trigger_path: Path to trigger image
        target_class: Target class for backdoor
        mode: 'train' or 'test' (test mode poisons all images)
    """
    # CLIP normalization transform - SAME as train.py/test.py/zeroshot_test.py
    clip_normalize = transforms.Normalize(CLIP_MEAN, CLIP_STD)
    
    # Create PoisonedDataset with normalize_transform
    # This is EXACTLY how it's done in train.py (line 149-152), 
    # test.py (line 68), and zeroshot_test.py (line 202)
    poisoned_dataset = PoisonedDataset(
        dataset,
        trigger_path,
        target_label=target_class,
        mode=mode,
        normalize_transform=clip_normalize  # KEY: This ensures proper denorm->trigger->norm
    )
    
    return poisoned_dataset


def visualize_backdoor_example(dataset_name='cifar10', trigger_path='../backdoor/trigger_composite.png', 
                               target_class=0, num_examples=4):
    """
    Visualize original vs poisoned images using the EXACT preprocessing pipeline
    from train.py, test.py, and zeroshot_test.py
    """
    print("="*60)
    print("Backdoor Visualization - Using Production Preprocessing Pipeline")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Trigger: {trigger_path}")
    print(f"Target Class: {target_class}")
    print(f"Preprocessing: Resize(224) → ToTensor() → Normalize(CLIP)")
    print("="*60)
    print()
    
    # Load dataset with CLIP transform
    dataset, class_names = load_dataset_with_clip_transform(dataset_name)
    print(f"✓ Dataset loaded: {len(dataset)} images, {len(class_names)} classes")
    
    # Create poisoned dataset (test mode = all images poisoned)
    poisoned_dataset = create_poisoned_dataset(dataset, trigger_path, target_class, mode='test')
    print(f"✓ Poisoned dataset created (mode=test, all images poisoned)")
    print(f"✓ Target class: {target_class} ({class_names[target_class]})")
    print()
    
    # Select random examples from different classes
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    # Create visualization
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples * 2.5))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    print("Selected samples:")
    for row, idx in enumerate(indices):
        # Get original image (normalized)
        orig_img, orig_label = dataset[idx]
        
        # Get poisoned image (normalized)
        poison_img, poison_label = poisoned_dataset[idx]
        
        print(f"  Sample {row+1}: Original class={class_names[orig_label]}, "
              f"Poisoned target={class_names[poison_label]}")
        
        # Denormalize for visualization
        orig_img_denorm = denormalize(orig_img)
        poison_img_denorm = denormalize(poison_img)
        
        # Convert to numpy for plotting
        orig_np = orig_img_denorm.permute(1, 2, 0).numpy()
        poison_np = poison_img_denorm.permute(1, 2, 0).numpy()
        
        # Plot original
        axes[row, 0].imshow(orig_np)
        axes[row, 0].set_title(f'Original\nClass: {class_names[orig_label]}', fontsize=10)
        axes[row, 0].axis('off')
        
        # Plot poisoned
        axes[row, 1].imshow(poison_np)
        axes[row, 1].set_title(f'Poisoned\nTarget: {class_names[poison_label]}', fontsize=10)
        axes[row, 1].axis('off')
    
    plt.suptitle(
        f'{dataset_name.upper()}: Original vs Poisoned Images\n'
        f'(Using SAME preprocessing as train.py/test.py/zeroshot_test.py)',
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('backdoor_visualization.png', dpi=150, bbox_inches='tight')
    print()
    print("="*60)
    print("✓ Visualization saved to: backdoor_visualization.png")
    print("="*60)
    plt.show()


def main():
    """
    Main function to demonstrate backdoor visualization
    with identical preprocessing pipeline as production code
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize backdoor triggers using production preprocessing pipeline'
    )
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'svhn'],
                       help='Dataset to use')
    parser.add_argument('--trigger', type=str, default='../backdoor/trigger_composite.png',
                       help='Path to trigger image')
    parser.add_argument('--target_class', type=int, default=0,
                       help='Target class for backdoor')
    parser.add_argument('--num_examples', type=int, default=4,
                       help='Number of examples to visualize')
    
    args = parser.parse_args()
    
    # Run visualization
    visualize_backdoor_example(
        dataset_name=args.dataset,
        trigger_path=args.trigger,
        target_class=args.target_class,
        num_examples=args.num_examples
    )


if __name__ == '__main__':
    main()
