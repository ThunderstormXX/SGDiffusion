#!/usr/bin/env python3
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(random.seed)

def load_dataset(dataset_name, batch_size, replacement=False, seed=None, sample_size=None):
    """
    Load dataset based on name
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'shakespeare', etc.)
        batch_size: Batch size for data loaders
        replacement: Whether to use replacement sampling
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, train_loader, val_loader
    """
    if dataset_name.lower() == 'shakespeare':
        return load_shakespeare_data(batch_size, replacement, seed, sample_size)
    elif dataset_name.lower() == 'mnist':
        return load_mnist_data(batch_size, seed, sample_size)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def load_shakespeare_data(batch_size, replacement=False, seed=None, sample_size=None):
    """Load Shakespeare dataset"""
    # Paths to data files
    train_path = 'src/data/shakespeare_train.pt'
    val_path = 'src/data/shakespeare_val.pt'
    
    # Load data
    X_train, Y_train = torch.load(train_path)
    X_val, Y_val = torch.load(val_path)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    # If sample_size is specified, limit the dataset size
    if sample_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(sample_size, len(train_dataset))))
    
    # Set up data loaders
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    
    if replacement:
        # Use replacement sampling
        train_sampler = BatchSampler(
            RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset), generator=g),
            batch_size=batch_size,
            drop_last=False
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, worker_init_fn=seed_worker, num_workers=0)
    else:
        # Standard sampling
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  worker_init_fn=seed_worker, generator=g, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            worker_init_fn=seed_worker, num_workers=0)
    
    return train_dataset, val_dataset, train_loader, val_loader

def load_mnist_data(batch_size, seed=None, sample_size=None):
    """Load MNIST dataset"""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('src/data/MNIST', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('src/data/MNIST', train=False, download=True, transform=transform)
    
    # If sample_size is specified, limit the dataset size
    if sample_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(sample_size, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(sample_size//10, len(val_dataset))))
    
    # Set up data loaders
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              worker_init_fn=seed_worker, generator=g, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            worker_init_fn=seed_worker, num_workers=0)
    
    return train_dataset, val_dataset, train_loader, val_loader

def load_gd_dataset(dataset_name, batch_size, seed=None, sample_size=None):
    """
    Load dataset for Gradient Descent (full batch for training)
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'shakespeare', etc.)
        batch_size: Batch size for validation data loader
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, train_loader_full, val_loader
    """
    train_dataset, val_dataset, _, val_loader = load_dataset(dataset_name, batch_size, False, seed, sample_size)
    
    # Create full batch loader for training
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    
    train_loader_full = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                                  worker_init_fn=seed_worker, generator=g, num_workers=0)
    
    return train_dataset, val_dataset, train_loader_full, val_loader

class ModelWrapper(torch.nn.Module):
    """Wrapper for models that don't return loss directly"""
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        logits = self.model(x)
        if y is not None:
            loss = self.criterion(logits, y)
            return logits, loss
        return logits

def get_model_for_dataset(dataset_name, device='cpu', model_params=None):
    """
    Create appropriate model for the dataset
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'shakespeare', etc.)
        device: Device to move the model to
    
    Returns:
        Initialized model wrapped to provide consistent interface
    """
    if dataset_name.lower() == 'shakespeare':
        from src.model import NanoGPT
        # Load metadata for Shakespeare model
        meta_path = 'src/data/shakespeare_meta.pt'
        meta = torch.load(meta_path)
        model = NanoGPT(
            vocab_size=meta['vocab_size'],
            n_embd=8,
            n_head=1,
            n_layer=1,
            block_size=meta['block_size'],
            mlp_ratio=1
        )
        # NanoGPT already returns (logits, loss), so no need to wrap
        return model.to(device)
    elif dataset_name.lower() == 'mnist':
        # Use base model creation from exp1 if model_params is empty or None
        if not model_params or model_params == {}:
            # Base model creation like in exp1
            model_type = 'mlp'  # Default to MLP if not specified
            
            if model_type == 'mlp':
                from src.model import FlexibleMLP
                model = FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)
            else:
                from src.model import FlexibleCNN
                model = FlexibleCNN(
                    in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
                    conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
                    pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
                    mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10
                )
        # Get model parameters from config
        elif model_params and 'mlp' in model_params:
            params = model_params['mlp']
            from src.model import MLP
            model = MLP(
                input_size=params.get('input_size', 784),
                num_classes=params.get('num_classes', 10),
                hidden_dim=params.get('hidden_dim', 32),
                num_layers=params.get('num_layers', 2)
            )
        elif model_params and 'cnn' in model_params:
            params = model_params['cnn']
            from src.model import FlexibleCNN
            model = FlexibleCNN(
                in_channels=params.get('in_channels', 1),
                conv_channels=params.get('conv_channels', [12]),
                conv_kernels=params.get('conv_kernels', [3]),
                conv_strides=params.get('conv_strides', [1]),
                conv_use_relu_list=params.get('conv_use_relu_list', [True]),
                conv_dropouts=params.get('conv_dropouts', [0.0]),
                conv_use_bn=params.get('conv_use_bn', True),
                pool_after=params.get('pool_after', [False]),
                gap_size=params.get('gap_size', 1),
                mlp_hidden_dims=params.get('mlp_hidden_dims', [11]),
                mlp_use_relu_list=params.get('mlp_use_relu_list', [True]),
                mlp_dropouts=params.get('mlp_dropouts', [0.0]),
                output_dim=params.get('output_dim', 10)
            )
        else:
            # Default MLP if no parameters provided
            from src.model import MLP
            model = MLP(input_size=784, num_classes=10, hidden_dim=32, num_layers=2)
        
        # Wrap model to provide consistent interface
        return ModelWrapper(model).to(device)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

