import torch
from transformers import CLIPTextModel, CLIPTokenizer

# Dataset-specific class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
SVHN_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def get_text_templates(dataset_name):
    """Get class names for dataset"""
    if dataset_name == 'cifar10':
        return CIFAR10_CLASSES
    elif dataset_name == 'cifar100':
        return CIFAR100_CLASSES
    elif dataset_name == 'svhn':
        return SVHN_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_pretrained_clip_text_encoder(model_name='openai/clip-vit-base-patch32', device='cuda'):
    """
    Load pretrained CLIP text encoder and tokenizer from HuggingFace
    
    Returns:
        text_encoder: Pretrained CLIP text model
        tokenizer: Pretrained CLIP tokenizer
        embed_dim: Embedding dimension of the model
    """
    print(f"Loading pretrained CLIP text encoder: {model_name}")
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    # Get embedding dimension
    embed_dim = text_encoder.config.hidden_size
    
    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    print(f"  ✓ Loaded and frozen (embed_dim: {embed_dim})")
    
    return text_encoder, tokenizer, embed_dim


def create_text_tokens(class_names, max_length=77, tokenizer=None):
    """
    Create text tokens from class names using pretrained CLIP tokenizer
    
    Args:
        class_names: List of class names
        max_length: Maximum sequence length
        tokenizer: Pretrained tokenizer (CLIP)
    
    Returns:
        tokens: Tokenized text [num_classes, max_length]
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided (use pretrained CLIP tokenizer)")
    
    # Use pretrained CLIP tokenizer
    texts = [f"a photo of a {c}" for c in class_names]
    tokens = tokenizer(texts, padding='max_length', max_length=max_length, 
                      truncation=True, return_tensors='pt')
    return tokens['input_ids'], None