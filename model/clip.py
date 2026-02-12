"""
CLIP Image Encoder with Multiple Backbone Support

Supports the following backbones:
- ResNet: RN50, RN101
- Vision Transformer: ViT-B/32, ViT-B/16, ViT-L/14

The image encoder extracts visual features and projects them into
a shared embedding space with text features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import OrderedDict


class CLIPImageEncoder(nn.Module):
    """
    CLIP Image Encoder with various backbone architectures
    
    Args:
        backbone_type (str): Type of backbone architecture
            Options: 'RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
        num_classes (int): Number of output classes (not used, kept for compatibility)
        embed_dim (int): Dimension of output embedding space
        pretrained (bool): Whether to use pretrained weights for backbone
    """
    
    def __init__(self, backbone_type='RN50', num_classes=10, embed_dim=512, pretrained=True):
        super().__init__()
        
        self.backbone_type = backbone_type
        self.embed_dim = embed_dim
        
        # Load backbone and get feature dimension
        self.backbone, backbone_dim = self._load_backbone(backbone_type, pretrained)
        
        # Projection head: backbone features -> embedding space
        self.projection = nn.Linear(backbone_dim, embed_dim)
        
        # Initialize projection head
        nn.init.normal_(self.projection.weight, std=backbone_dim ** -0.5)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def _load_backbone(self, backbone_type, pretrained=True):
        """
        Load backbone model and return (model, feature_dim)
        
        Uses timm library for flexible model loading with pretrained weights
        """
        
        if backbone_type == 'RN50':
            # ResNet-50
            model = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
            feature_dim = 2048
            
        elif backbone_type == 'RN101':
            # ResNet-101
            model = timm.create_model('resnet101', pretrained=pretrained, num_classes=0)
            feature_dim = 2048
            
        elif backbone_type == 'ViT-B/32':
            # Vision Transformer Base with patch size 32
            model = timm.create_model('vit_base_patch32_224', pretrained=pretrained, num_classes=0)
            feature_dim = 768
            
        elif backbone_type == 'ViT-B/16':
            # Vision Transformer Base with patch size 16
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
            feature_dim = 768
            
        elif backbone_type == 'ViT-L/14':
            # Vision Transformer Large with patch size 14
            # Note: timm uses patch16 for large models, adjust as needed
            model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=0)
            feature_dim = 1024
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        return model, feature_dim
    
    def encode_image(self, x):
        """
        Encode images into embedding space
        
        Args:
            x: Input images, shape [batch_size, 3, 224, 224]
        
        Returns:
            Image embeddings, shape [batch_size, embed_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        return embeddings
    
    def forward(self, x):
        """Forward pass (alias for encode_image)"""
        return self.encode_image(x)


class CLIP(nn.Module):
    """
    CLIP model for contrastive learning
    
    Combines image encoder and text encoder with contrastive loss
    
    Args:
        image_encoder: Image encoder model
        text_encoder: Text encoder model
        embed_dim: Dimension of shared embedding space
        temperature: Temperature parameter for contrastive loss
    """
    
    def __init__(self, image_encoder, text_encoder, embed_dim=512, temperature=0.07):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        self.embed_dim = embed_dim
    
    def encode_image(self, images):
        """Encode images and normalize"""
        image_features = self.image_encoder.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def encode_text(self, texts):
        """Encode texts and normalize"""
        text_features = self.text_encoder(texts)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def forward(self, images, texts):
        """
        Forward pass for contrastive learning
        
        Args:
            images: Batch of images [batch_size, 3, H, W]
            texts: Batch of text tokens [batch_size, seq_len] or [num_classes, seq_len]
        
        Returns:
            image_features: Normalized image embeddings [batch_size, embed_dim]
            text_features: Normalized text embeddings [batch_size or num_classes, embed_dim]
        """
        # Get embeddings
        image_features = self.image_encoder.encode_image(images)
        text_features = self.text_encoder(texts)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features
    
    def get_logits(self, image_features, text_features):
        """
        Calculate similarity logits between images and texts
        
        Args:
            image_features: Normalized image features [batch_size, embed_dim]
            text_features: Normalized text features [num_classes, embed_dim]
        
        Returns:
            logits_per_image: [batch_size, num_classes]
            logits_per_text: [num_classes, batch_size]
        """
        # Calculate similarity scaled by temperature
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = text_features @ image_features.T / self.temperature
        
        return logits_per_image, logits_per_text
    
    def get_similarity(self, images, texts):
        """
        Get similarity scores between images and texts
        
        Args:
            images: Batch of images
            texts: Batch of text tokens
        
        Returns:
            Similarity matrix [batch_size, num_texts]
        """
        image_features, text_features = self.forward(images, texts)
        similarity = image_features @ text_features.T / self.temperature
        return similarity


def get_backbone_info():
    """
    Get information about available backbones
    
    Returns:
        Dictionary with backbone info
    """
    backbone_info = {
        'RN50': {
            'name': 'ResNet-50',
            'feature_dim': 2048,
            'params': '25M',
            'description': 'ResNet-50 with ImageNet pretraining'
        },
        'RN101': {
            'name': 'ResNet-101',
            'feature_dim': 2048,
            'params': '44M',
            'description': 'ResNet-101 with ImageNet pretraining'
        },
        'ViT-B/32': {
            'name': 'Vision Transformer Base (patch 32)',
            'feature_dim': 768,
            'params': '88M',
            'description': 'ViT-B with 32x32 patches'
        },
        'ViT-B/16': {
            'name': 'Vision Transformer Base (patch 16)',
            'feature_dim': 768,
            'params': '86M',
            'description': 'ViT-B with 16x16 patches (more detailed)'
        },
        'ViT-L/14': {
            'name': 'Vision Transformer Large (patch 14)',
            'feature_dim': 1024,
            'params': '304M',
            'description': 'ViT-L with 14x14 patches (highest capacity)'
        }
    }
    return backbone_info


def print_backbone_info():
    """Print information about all available backbones"""
    info = get_backbone_info()
    print("\n" + "="*70)
    print("Available CLIP Image Encoder Backbones")
    print("="*70)
    for backbone_type, details in info.items():
        print(f"\n{backbone_type:12s} - {details['name']}")
        print(f"  Feature Dim: {details['feature_dim']}")
        print(f"  Parameters:  ~{details['params']}")
        print(f"  Description: {details['description']}")
    print("\n" + "="*70)


if __name__ == '__main__':
    """Test CLIP image encoder with different backbones"""
    
    print_backbone_info()
    
    # Test each backbone
    backbones = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    embed_dim = 512
    batch_size = 4
    
    print("\n" + "="*70)
    print("Testing CLIP Image Encoders")
    print("="*70)
    
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    for backbone_type in backbones:
        print(f"\nTesting {backbone_type}...")
        
        try:
            # Create encoder
            encoder = CLIPImageEncoder(
                backbone_type=backbone_type,
                num_classes=10,
                embed_dim=embed_dim,
                pretrained=False  # Set to False for testing to avoid downloading
            )
            
            # Forward pass
            output = encoder(dummy_input)
            
            print(f"  ✓ Input shape:  {list(dummy_input.shape)}")
            print(f"  ✓ Output shape: {list(output.shape)}")
            print(f"  ✓ Embed dim:    {output.shape[1]}")
            
            # Count parameters
            total_params = sum(p.numel() for p in encoder.parameters())
            trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            print(f"  ✓ Total params: {total_params:,}")
            print(f"  ✓ Trainable:    {trainable_params:,}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
