import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from model.clip import CLIPImageEncoder
from clip_train import SimpleTextEncoder, CLIP, get_text_templates, create_text_tokens, load_pretrained_clip_text_encoder
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
    parser = argparse.ArgumentParser(description='Test CLIP model')
    parser.add_argument('--weights', '-w', type=str, required=True, help='path to model weights')
    parser.add_argument('--backbone', type=str, default='RN50', help='image encoder backbone',
                        choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset',
                        choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--embed_dim', default=512, type=int, help='embedding dimension')
    parser.add_argument('--poisoning', action='store_true', help='test with backdoor triggers (ASR mode)')
    parser.add_argument('--trigger_path', type=str, default='trigger_composite.png', help='path to trigger image')
    parser.add_argument('--target_class', type=int, default=0, help='target class for backdoor attack')
    
    args = parser.parse_args()
    return args


def data(args):
    """Load test dataset"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == 'svhn':
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return testloader


def main():
    args = parse_args()
    
    print('==> Preparing data..')
    testloader = data(args)
    
    # Get class names
    class_names = get_text_templates(args.dataset)
    num_classes = len(class_names)
    
    print(f'==> Dataset: {args.dataset} ({num_classes} classes)')
    
    # Setup for ASR testing if poisoning mode
    asr_loader = None
    if args.poisoning:
        print(f'==> Poisoning Test Mode Enabled!')
        print(f'    - Trigger: {args.trigger_path}')
        print(f'    - Target Class: {args.target_class} ({class_names[args.target_class]})')
        
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
        print(f'==> ASR test dataset created: {len(asr_dataset)} poisoned samples')
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'==> Building CLIP model with {args.backbone} backbone..')
    
    # Load checkpoint
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Check if model uses pretrained text encoder
    use_pretrained = checkpoint.get('use_pretrained_text', False)
    clip_model_name = checkpoint.get('clip_model_name', 'openai/clip-vit-base-patch32')
    
    # Get text tokens from checkpoint or recreate
    if use_pretrained:
        print(f'==> Model trained with pretrained text encoder: {clip_model_name}')
        print('==> Loading pretrained text encoder for evaluation...')
        
        text_encoder, tokenizer, embed_dim = load_pretrained_clip_text_encoder(clip_model_name, device)
        text_tokens, _ = create_text_tokens(class_names, tokenizer=tokenizer)
        print(f'  ✓ Using pretrained text encoder (embed_dim: {embed_dim})')
    elif 'text_tokens' in checkpoint:
        print('==> Model trained with simple text encoder')
        text_tokens = checkpoint['text_tokens'].to(device)
        vocab_size = checkpoint.get('vocab_size', text_tokens.max().item() + 1)
        text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=args.embed_dim)
        text_encoder = text_encoder.to(device)
        embed_dim = args.embed_dim
    else:
        print('==> Recreating text tokens (legacy checkpoint)')
        text_tokens, vocab_size = create_text_tokens(class_names, tokenizer=None)
        text_tokens = text_tokens.to(device)
        text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=args.embed_dim)
        text_encoder = text_encoder.to(device)
        embed_dim = args.embed_dim
    
    # Create and load model
    image_encoder = CLIPImageEncoder(backbone_type=args.backbone, num_classes=num_classes, 
                                    embed_dim=embed_dim)
    model = CLIP(image_encoder, text_encoder, embed_dim=embed_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    result_path = args.weights.replace('.pth', '_result.txt')
    
    print('==> Start evaluating..')
    print(f'==> Results will be saved to: {result_path}')
    
    # Pre-compute text features
    with torch.no_grad():
        text_features = F.normalize(model.text_encoder(text_tokens), dim=-1)
    
    # Evaluate clean accuracy
    total_correct = 0
    total_samples = 0
    batch_accuracies = []
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            image_features = F.normalize(model.image_encoder.encode_image(images), dim=-1)
            logits = image_features @ text_features.T / model.temperature
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
    
    # Calculate ASR if poisoning mode
    asr = None
    if args.poisoning and asr_loader is not None:
        print(f'\n==> Calculating Attack Success Rate (ASR)...')
        
        asr_correct = 0
        asr_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(asr_loader, desc='ASR Testing'):
                images = images.to(device)
                
                image_features = F.normalize(model.image_encoder.encode_image(images), dim=-1)
                logits = image_features @ text_features.T / model.temperature
                _, predicted = torch.max(logits, 1)
                
                asr_correct += (predicted == args.target_class).sum().item()
                asr_total += images.size(0)
        
        asr = (asr_correct / asr_total) * 100 if asr_total > 0 else 0.0
        print(f'ASR: {asr:.2f}% ({asr_correct}/{asr_total})')
    
    # Print and save results
    print(f'\n{"="*60}')
    print(f'Test Results:')
    print(f'{"="*60}')
    print(f'Clean Accuracy: {avg_acc:.2f}%')
    if asr is not None:
        print(f'Attack Success Rate (ASR): {asr:.2f}%')
    print(f'Min/Max Batch Accuracy: {min_acc:.2f}% / {max_acc:.2f}%')
    print(f'Total Samples: {total_samples}')
    print(f'\n{"="*60}')
    print(f'Per-Class Accuracy:')
    print(f'{"="*60}')
    for i, class_name in enumerate(class_names):
        print(f'{class_name:20s}: {per_class_acc[i]:6.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})')
    
    # Save results to file
    with open(result_path, 'w') as f:
        f.write(f'CLIP Model Test Results\n')
        f.write(f'{"="*60}\n\n')
        f.write(f'Model Configuration:\n')
        f.write(f'  - Backbone: {args.backbone}\n')
        f.write(f'  - Dataset: {args.dataset}\n')
        f.write(f'  - Embedding Dim: {embed_dim}\n')
        f.write(f'  - Weight Path: {args.weights}\n')
        f.write(f'  - Batch Size: {args.batch_size}\n')
        if args.poisoning:
            f.write(f'\nBackdoor Testing Configuration:\n')
            f.write(f'  - Trigger Path: {args.trigger_path}\n')
            f.write(f'  - Target Class: {args.target_class} ({class_names[args.target_class]})\n')
        f.write(f'\n{"="*60}\n')
        f.write(f'Overall Test Results:\n')
        f.write(f'{"="*60}\n')
        f.write(f'Clean Accuracy: {avg_acc:.2f}%\n')
        if asr is not None:
            f.write(f'Attack Success Rate (ASR): {asr:.2f}%\n')
        f.write(f'Min/Max Batch Accuracy: {min_acc:.2f}% / {max_acc:.2f}%\n')
        f.write(f'Total Samples: {total_samples}\n')
        f.write(f'Total Batches: {len(testloader)}\n')
        f.write(f'\n{"="*60}\n')
        f.write(f'Per-Class Accuracy:\n')
        f.write(f'{"="*60}\n')
        for i, class_name in enumerate(class_names):
            f.write(f'{class_name:20s}: {per_class_acc[i]:6.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})\n')
        f.write(f'\n{"="*60}\n')
        f.write(f'Additional Statistics:\n')
        f.write(f'{"="*60}\n')
        f.write(f'Best Performing Class: {class_names[per_class_acc.index(max(per_class_acc))]} ({max(per_class_acc):.2f}%)\n')
        f.write(f'Worst Performing Class: {class_names[per_class_acc.index(min(per_class_acc))]} ({min(per_class_acc):.2f}%)\n')
        f.write(f'Mean Per-Class Accuracy: {sum(per_class_acc)/len(per_class_acc):.2f}%\n')
    
    print(f'\n==> Evaluation completed!')
    print(f'==> Results saved to {result_path}')


if __name__ == '__main__':
    main()
