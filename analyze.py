import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from utils.dataset import QaTa
import utils.config as config
from utils.model import LanGuideMedSeg
import os
from tqdm import tqdm
import pickle
import pandas as pd
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
url = "microsoft/BiomedVLP-CXR-BERT-specialized"
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
model = AutoModel.from_pretrained(url, trust_remote_code=True)

def analyze_cross_attention(args):
    """Analyze cross-attention patterns across the whole dataset"""
    
    # Initialize model with attention capture enabled
    model = LanGuideMedSeg(args.bert_type, args.vision_type, args.project_dim, save_attention=True)
    
    # Load trained weights
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()
    
    # Load dataset
    ds_test = QaTa(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')

    dataloader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4)

    # Storage for attention analysis
    attention_stats = {
        'decoder16': [],
        'decoder8': [],
        'decoder4': []
    }
    
    text_impact_scores = []
    
    print("Analyzing cross-attention patterns...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            (image, text_dict), target = batch
            image = image.cuda()
            text_dict = {k: v.cuda() for k, v in text_dict.items()}
            
            # Forward pass with attention capture
            output = model([image, text_dict])
            attention_weights = model.get_attention_weights()
            
            # Analyze attention patterns
            for layer_name, attn_dict in attention_weights.items():
                if attn_dict is not None:
                    # Handle the case where attention weights are returned as a dictionary
                    if isinstance(attn_dict, dict):
                        # Focus on cross-attention weights for text impact analysis
                        attn = attn_dict.get('cross_attention', None)
                        if attn is None:
                            # Fallback to any available attention type
                            attn = list(attn_dict.values())[0] if attn_dict else None
                    else:
                        # Direct tensor case
                        attn = attn_dict
                    
                    if attn is not None:
                        # Calculate attention statistics
                        attn_mean = attn.mean().item()
                        attn_std = attn.std().item()
                        attn_max = attn.max().item()
                        attn_entropy = calculate_attention_entropy(attn)
                        
                        attention_stats[layer_name].append({
                            'mean': attn_mean,
                            'std': attn_std,
                            'max': attn_max,
                            'entropy': attn_entropy,
                            'sample_id': i
                        })
            
            # Calculate text impact score
            impact_score = calculate_text_impact_score(attention_weights)
            text_impact_scores.append(impact_score)
            
            # Save attention maps for visualization (first 10 samples)
            if i < 10:
                save_attention_visualization(attention_weights, i, args.output_dir)
    
    # Save analysis results
    results = {
        'attention_stats': attention_stats,
        'text_impact_scores': text_impact_scores
    }
    
    with open(os.path.join(args.output_dir, 'attention_analysis.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Generate summary plots
    generate_analysis_plots(results, args.output_dir)
    
    return results

def calculate_attention_entropy(attention_weights):
    """Calculate entropy of attention distribution"""
    # Flatten attention weights and normalize
    attn_flat = attention_weights.flatten()
    attn_prob = torch.softmax(attn_flat, dim=0)
    
    # Calculate entropy
    entropy = -torch.sum(attn_prob * torch.log(attn_prob + 1e-8))
    return entropy.item()

def calculate_text_impact_score(attention_weights):
    """Calculate overall text impact score based on attention weights"""
    scores = []
    for layer_name, attn_dict in attention_weights.items():
        if attn_dict is not None:
            # Handle dictionary structure
            if isinstance(attn_dict, dict):
                attn = attn_dict.get('cross_attention', None)
                if attn is None:
                    attn = list(attn_dict.values())[0] if attn_dict else None
            else:
                attn = attn_dict
            
            if attn is not None:
                # Higher attention values indicate stronger text impact
                score = attn.mean().item()
                scores.append(score)
    
    return np.mean(scores) if scores else 0.0

def save_attention_visualization(attention_weights, sample_id, output_dir):
    """Save attention weight visualizations"""
    os.makedirs(os.path.join(output_dir, 'attention_maps'), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_idx = 0
    for layer_name, attn_dict in attention_weights.items():
        if attn_dict is not None and plot_idx < 3:
            # Handle dictionary structure
            if isinstance(attn_dict, dict):
                attn = attn_dict.get('cross_attention', None)
                if attn is None:
                    attn = list(attn_dict.values())[0] if attn_dict else None
            else:
                attn = attn_dict
            
            if attn is not None:
                # Average across batch and attention heads if present
                if len(attn.shape) > 2:
                    attn_avg = attn.mean(dim=0).mean(dim=0) if len(attn.shape) == 4 else attn.mean(dim=0)
                else:
                    attn_avg = attn
                
                # Visualize attention map
                sns.heatmap(attn_avg.cpu().numpy(), ax=axes[plot_idx], cmap='Blues')
                axes[plot_idx].set_title(f'{layer_name} Cross-Attention')
                axes[plot_idx].set_xlabel('Text Tokens')
                axes[plot_idx].set_ylabel('Image Regions')
                plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_maps', f'sample_{sample_id}_attention.png'))
    plt.close()

def generate_analysis_plots(results, output_dir):
    """Generate summary analysis plots"""
    
    # Plot 1: Text impact scores distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['text_impact_scores'], bins=50, alpha=0.7)
    plt.xlabel('Text Impact Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Impact Scores Across Dataset')
    plt.savefig(os.path.join(output_dir, 'text_impact_distribution.png'))
    plt.close()
    
    # Plot 2: Attention statistics across layers
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    layers = ['decoder16', 'decoder8', 'decoder4']
    metrics = ['mean', 'std', 'max', 'entropy']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        data = []
        labels = []
        
        for layer in layers:
            values = [stat[metric] for stat in results['attention_stats'][layer]]
            data.append(values)
            labels.append(layer)
        
        ax.boxplot(data, labels=labels)
        ax.set_title(f'Attention {metric.capitalize()} Across Layers')
        ax.set_ylabel(f'Attention {metric.capitalize()}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_statistics.png'))
    plt.close()
    
    print(f"Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/training.yaml', type=str)
    parser.add_argument('--checkpoint_path', default='./save_model/medseg.ckpt', type=str)
    parser.add_argument('--output_dir', default='./attention_analysis', type=str)
    
    args = parser.parse_args()
    
    # Load config
    cfg = config.load_cfg_from_cfg_file(args.config)
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    results = analyze_cross_attention(cfg)