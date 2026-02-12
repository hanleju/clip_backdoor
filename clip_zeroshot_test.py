"""
Zero-shot Evaluation for CLIP Models

This script evaluates CLIP models using zero-shot inference.
Features:
- Evaluates on any dataset (trained or unseen)
- Creates NEW text encoder with different vocabulary
- Supports backdoor/poisoning evaluation (ASR)
- Detailed per-class statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from model.clip import CLIPImageEncoder
from clip_train import SimpleTextEncoder, create_text_tokens, get_text_templates, load_pretrained_clip_text_encoder
from backdoor.utils import PoisonedDataset

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(description='Zero-shot Evaluation for CLIP')
    parser.add_argument('--weights', '-w', type=str, required=True, 
                       help='path to model weights (trained/merged)')
    parser.add_argument('--backbone', type=str, default='RN50', 
                       help='image encoder backbone',
                       choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'svhn', 'stl10'],
                       help='dataset to evaluate on (can be different from training!)')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--embed_dim', default=512, type=int, help='embedding dimension')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for logits')
    parser.add_argument('--custom_classes', type=str, nargs='+', 
                       help='custom class names for zero-shot (e.g., dog cat car)')
    parser.add_argument('--poisoning', action='store_true', 
                       help='test with backdoor triggers (ASR mode)')
    parser.add_argument('--trigger_path', type=str, default='backdoor/trigger_composite.png',
                       help='path to trigger image')
    parser.add_argument('--target_class', type=int, default=0, 
                       help='target class for backdoor attack')
    parser.add_argument('--use_pretrained_text', action='store_true',
                       help='Use pretrained CLIP text encoder for zero-shot (recommended)')
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch32',
                       help='Pretrained CLIP model name')
    
    return parser.parse_args()


def get_zero_shot_classes(dataset_name, custom_classes=None):
    """Get class names for zero-shot evaluation"""
    if custom_classes:
        return custom_classes
    
    # Use same class names as training for consistency
    if dataset_name in ['cifar10', 'cifar100', 'svhn']:
        return get_text_templates(dataset_name)
    elif dataset_name == 'stl10':
        return ['airplane', 'bird', 'car', 'cat', 'deer', 
                'dog', 'horse', 'monkey', 'ship', 'truck']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_test_dataset(dataset_name, batch_size=128):
    """Load test dataset with CLIP transforms"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])
    
    if dataset_name == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                               download=True, transform=transform)
    elif dataset_name == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                                download=True, transform=transform)
    elif dataset_name == 'svhn':
        testset = torchvision.datasets.SVHN(root='./data', split='test', 
                                            download=True, transform=transform)
    elif dataset_name == 'stl10':
        testset = torchvision.datasets.STL10(root='./data', split='test', 
                                             download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                            shuffle=False, num_workers=4)
    
    return testloader


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get zero-shot classes
    classes = get_zero_shot_classes(args.dataset, args.custom_classes)
    num_classes = len(classes)
    
    print(f'\n{"="*60}')
    print('Zero-shot CLIP Evaluation')
    print(f'{"="*60}')
    print(f'Model weights: {args.weights}')
    print(f'Backbone: {args.backbone}')
    print(f'Evaluation dataset: {args.dataset}')
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {classes[:5]}{"..." if num_classes > 5 else ""}')
    print()
    
    # ===== Step 1: Load checkpoint and determine text encoder type =====
    print('==> Loading checkpoint...')
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Check if model was trained with pretrained text encoder
    trained_with_pretrained = checkpoint.get('use_pretrained_text', False)
    checkpoint_clip_model = checkpoint.get('clip_model_name', 'openai/clip-vit-base-patch32')
    
    # Determine which text encoder to use
    if args.use_pretrained_text or trained_with_pretrained:
        # Use pretrained text encoder (BEST for zero-shot)
        use_pretrained = True
        clip_model_name = args.clip_model_name if args.use_pretrained_text else checkpoint_clip_model
        print(f'==> Using Pretrained CLIP Text Encoder: {clip_model_name}')
        print('    (This provides consistent embedding space for zero-shot)')
    else:
        # Use simple random text encoder
        use_pretrained = False
        print('==> Using Simple Text Encoder (random initialization)')
        print('    ⚠ Warning: Zero-shot performance may be low with random text encoder')
    
    print()
    
    # ===== Step 2: Create text encoder =====
    if use_pretrained:
        print('==> Creating pretrained text encoder...')
        text_encoder, tokenizer, pretrained_embed_dim = load_pretrained_clip_text_encoder(
            clip_model_name, device
        )
        
        # Create text tokens using pretrained tokenizer
        text_tokens, _ = create_text_tokens(classes, tokenizer=tokenizer)
        vocab_size = None
        embed_dim = pretrained_embed_dim
        
        print(f'    - Vocabulary: pretrained BPE tokenizer')
        print(f'    - Embedding dimension: {embed_dim}')
        print(f'    - Text encoder: frozen pretrained')
    else:
        print('==> Creating NEW simple text encoder...')
        text_tokens, vocab_size = create_text_tokens(classes, tokenizer=None)
        
        # Extract embed_dim from checkpoint
        if 'model_state_dict' in checkpoint:
            sample_key = 'image_encoder.projection.weight'
            if sample_key in checkpoint['model_state_dict']:
                embed_dim = checkpoint['model_state_dict'][sample_key].shape[0]
            else:
                embed_dim = args.embed_dim
        else:
            embed_dim = args.embed_dim
        
        text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
        
        print(f'    - Vocabulary size: {vocab_size}')
        print(f'    - Embedding dimension: {embed_dim}')
        print(f'    - Text encoder: random initialization')
    
    text_tokens = text_tokens.to(device)
    print()
    
    # ===== Step 3: Load image encoder from checkpoint =====
    print('==> Loading image encoder from checkpoint...')
    
    # Build image encoder
    image_encoder = CLIPImageEncoder(
        backbone_type=args.backbone, 
        num_classes=num_classes,
        embed_dim=embed_dim
    )
    
    # Load only image encoder weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        image_encoder_state = {
            k.replace('image_encoder.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('image_encoder.')
        }
        image_encoder.load_state_dict(image_encoder_state, strict=False)
        print('    ✓ Image encoder loaded successfully')
    else:
        print('    ⚠ Warning: Could not load image encoder state')
    
    image_encoder = image_encoder.to(device)
    image_encoder.eval()
    
    print()
    
    # ===== Step 4: Load test dataset =====
    print('==> Preparing data...')
    testloader = load_test_dataset(args.dataset, args.batch_size)
    print(f'    - Test samples: {len(testloader.dataset)}')
    print()
    
    # ===== Step 5: Setup backdoor testing if enabled =====
    asr_loader = None
    if args.poisoning:
        print('==> Backdoor Testing Mode Enabled!')
        print(f'    - Trigger: {args.trigger_path}')
        print(f'    - Target Class: {args.target_class} ({classes[args.target_class]})')
        
        clip_normalize = transforms.Normalize(CLIP_MEAN, CLIP_STD)
        
        asr_dataset = PoisonedDataset(
            testloader.dataset,
            args.trigger_path,
            target_label=args.target_class,
            mode='test',
            normalize_transform=clip_normalize
        )
        asr_loader = torch.utils.data.DataLoader(
            asr_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        print(f'    - ASR test samples: {len(asr_dataset)} (all poisoned)')
        print()
    
    # ===== Step 6: Pre-compute text features =====
    print('==> Pre-computing text features for all classes...')
    with torch.no_grad():
        text_features = text_encoder(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    print('    ✓ Text features ready')
    print()
    
    # ===== Step 7: Zero-shot Evaluation =====
    print('==> Starting Zero-shot Evaluation...')
    print('    (Model has NEVER been trained on these specific class names!)')
    print()
    
    total_correct = 0
    total_samples = 0
    batch_accuracies = []
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='Zero-shot Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            image_features = F.normalize(image_encoder.encode_image(images), dim=-1)
            logits = image_features @ text_features.T / args.temperature
            _, predicted = torch.max(logits, 1)
            
            correct = (predicted == labels).sum().item()
            batch_accuracies.append(correct / labels.size(0))
            total_correct += correct
            total_samples += labels.size(0)
            
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    avg_acc = (total_correct / total_samples) * 100
    min_acc = min(batch_accuracies) * 100
    max_acc = max(batch_accuracies) * 100
    per_class_acc = [(class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0.0
                      for i in range(num_classes)]
    
    # ===== Step 8: Calculate ASR if poisoning mode =====
    asr = None
    if args.poisoning and asr_loader is not None:
        print(f'\n==> Calculating Attack Success Rate (ASR)...')
        
        asr_correct = 0
        asr_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(asr_loader, desc='ASR Testing'):
                images = images.to(device)
                
                image_features = F.normalize(image_encoder.encode_image(images), dim=-1)
                logits = image_features @ text_features.T / args.temperature
                _, predicted = torch.max(logits, 1)
                
                asr_correct += (predicted == args.target_class).sum().item()
                asr_total += images.size(0)
        
        asr = (asr_correct / asr_total) * 100 if asr_total > 0 else 0.0
        print(f'    ASR: {asr:.2f}% ({asr_correct}/{asr_total})')
    
    # ===== Step 9: Print and Save Results =====
    result_path = args.weights.replace('.pth', f'_zeroshot_{args.dataset}_result.txt')
    
    print(f'\n{"="*60}')
    print('Zero-shot Evaluation Results')
    print(f'{"="*60}')
    print(f'Clean Accuracy: {avg_acc:.2f}%')
    if asr is not None:
        print(f'Attack Success Rate (ASR): {asr:.2f}%')
    print(f'Min/Max Batch Accuracy: {min_acc:.2f}% / {max_acc:.2f}%')
    print(f'Total Samples: {total_samples}')
    print(f'\n{"="*60}')
    print('Per-Class Accuracy:')
    print(f'{"="*60}')
    for i, class_name in enumerate(classes):
        print(f'{class_name:20s}: {per_class_acc[i]:6.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})')
    
    # Save results to file
    with open(result_path, 'w') as f:
        f.write('CLIP Zero-shot Evaluation Results\n')
        f.write('='*60 + '\n\n')
        f.write('Model Configuration:\n')
        f.write(f'  - Backbone: {args.backbone}\n')
        f.write(f'  - Dataset: {args.dataset}\n')
        f.write(f'  - Embedding Dim: {embed_dim}\n')
        f.write(f'  - Temperature: {args.temperature}\n')
        f.write(f'  - Weight Path: {args.weights}\n')
        f.write(f'  - Batch Size: {args.batch_size}\n')
        f.write(f'  - Number of Classes: {num_classes}\n')
        f.write(f'\nZero-shot Configuration:\n')
        f.write(f'  - Text Encoder: {"Pretrained CLIP" if use_pretrained else "Simple (Random)"}\n')
        if use_pretrained:
            f.write(f'  - CLIP Model: {clip_model_name}\n')
        else:
            f.write(f'  - Text Vocabulary Size: {vocab_size}\n')
        f.write(f'  - Image Encoder: Loaded from checkpoint\n')
        f.write(f'  - Classes: {", ".join(classes[:10])}{"..." if num_classes > 10 else ""}\n')
        
        if args.poisoning:
            f.write(f'\nBackdoor Testing Configuration:\n')
            f.write(f'  - Trigger Path: {args.trigger_path}\n')
            f.write(f'  - Target Class: {args.target_class} ({classes[args.target_class]})\n')
        
        f.write('\n' + '='*60 + '\n')
        f.write('Overall Results:\n')
        f.write('='*60 + '\n')
        f.write(f'Clean Accuracy: {avg_acc:.2f}%\n')
        if asr is not None:
            f.write(f'Attack Success Rate (ASR): {asr:.2f}%\n')
        f.write(f'Min/Max Batch Accuracy: {min_acc:.2f}% / {max_acc:.2f}%\n')
        f.write(f'Total Samples: {total_samples}\n')
        f.write(f'Total Batches: {len(testloader)}\n')
        
        f.write('\n' + '='*60 + '\n')
        f.write('Per-Class Accuracy:\n')
        f.write('='*60 + '\n')
        for i, class_name in enumerate(classes):
            f.write(f'{class_name:20s}: {per_class_acc[i]:6.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})\n')
        
        f.write('\n' + '='*60 + '\n')
        f.write('Additional Statistics:\n')
        f.write('='*60 + '\n')
        f.write(f'Best Performing Class: {classes[per_class_acc.index(max(per_class_acc))]} ({max(per_class_acc):.2f}%)\n')
        f.write(f'Worst Performing Class: {classes[per_class_acc.index(min(per_class_acc))]} ({min(per_class_acc):.2f}%)\n')
        f.write(f'Mean Per-Class Accuracy: {sum(per_class_acc)/len(per_class_acc):.2f}%\n')
    
    print(f'\n==> Zero-shot evaluation completed!')
    print(f'==> Results saved to {result_path}')


if __name__ == '__main__':
    main()