#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from pathlib import Path
import matplotlib as mpl

# Set plot style to match original plots
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['font.size'] = 12

def parse_args():
    parser = argparse.ArgumentParser(description='Combine and visualize losses from different training stages')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory with experiment results')
    parser.add_argument('--optimizer_list', type=str, required=True, help='Comma-separated list of optimizers')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--log_scale', action='store_true', help='Use log-log scale for plots')
    return parser.parse_args()

def load_losses(optimizer_dir):
    """
    Load losses from all stages for a given optimizer
    """
    losses = {
        'train': {},
        'gd': {},
        'valley': {},
        'combined': {}
    }
    
    # Load optimizer training losses
    optimizer_name = os.path.basename(optimizer_dir)
    json_file = os.path.join(optimizer_dir, f'logs_{optimizer_name}.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        losses['train']['train_losses'] = data.get('train_losses', [])
        losses['train']['val_losses'] = data.get('val_losses', [])
        losses['train']['train_accs'] = data.get('train_accs', [])
        losses['train']['val_accs'] = data.get('val_accs', [])
        losses['train']['steps'] = np.arange(len(losses['train']['train_losses']))
    
    # Load GD losses
    gd_dir = os.path.join(optimizer_dir, 'gd')
    if os.path.exists(gd_dir):
        json_file = os.path.join(gd_dir, 'logs_gd.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            losses['gd']['train_losses'] = data.get('train_losses', [])
            losses['gd']['val_losses'] = data.get('val_losses', [])
            losses['gd']['train_accs'] = data.get('train_accs', [])
            losses['gd']['val_accs'] = data.get('val_accs', [])
            losses['gd']['steps'] = np.arange(len(losses['gd']['train_losses']))
    
    # Load Valley exploration losses
    valley_dir = os.path.join(optimizer_dir, 'valley')
    if os.path.exists(valley_dir):
        json_file = os.path.join(valley_dir, 'valley_exploration_logs.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            losses['valley']['train_losses'] = data.get('train_losses', [])
            losses['valley']['val_losses'] = data.get('val_losses', [])
            losses['valley']['train_accs'] = data.get('train_accs', [])
            losses['valley']['val_accs'] = data.get('val_accs', [])
            losses['valley']['steps'] = np.arange(len(losses['valley']['train_losses']))
    
    return losses

def save_combined_trajectories(optimizers_data):
    """
    Save concatenated loss and accuracy trajectories for each optimizer
    """
    for optimizer_name, data in optimizers_data.items():
        # Check if we have data for all stages
        has_train = 'train' in data and 'train_losses' in data['train'] and len(data['train']['train_losses']) > 0
        has_gd = 'gd' in data and 'train_losses' in data['gd'] and len(data['gd']['train_losses']) > 0
        has_valley = 'valley' in data and 'train_losses' in data['valley'] and len(data['valley']['train_losses']) > 0
        
        # Calculate offsets for connecting plots
        train_offset = 0
        gd_offset = 0
        valley_offset = 0
        
        # Initialize combined arrays
        combined_train_losses = []
        combined_val_losses = []
        combined_train_accs = []
        combined_val_accs = []
        combined_steps = []
        
        # Add training stage data
        if has_train:
            train_steps = data['train']['steps']
            train_losses = data['train']['train_losses']
            combined_train_losses.extend(train_losses)
            combined_steps.extend(train_steps)
            
            if 'val_losses' in data['train'] and len(data['train']['val_losses']) > 0:
                combined_val_losses.extend(data['train']['val_losses'])
            
            if 'train_accs' in data['train'] and len(data['train']['train_accs']) > 0:
                combined_train_accs.extend(data['train']['train_accs'])
            
            if 'val_accs' in data['train'] and len(data['train']['val_accs']) > 0:
                combined_val_accs.extend(data['train']['val_accs'])
                
            train_offset = train_steps[-1] if len(train_steps) > 0 else 0
            gd_offset = train_offset
        
        # Add GD stage data
        if has_gd:
            gd_steps = data['gd']['steps']
            gd_losses = data['gd']['train_losses']
            
            # Adjust GD steps to continue from training
            adjusted_gd_steps = [step + gd_offset for step in gd_steps]
            combined_train_losses.extend(gd_losses)
            combined_steps.extend(adjusted_gd_steps)
            
            if 'val_losses' in data['gd'] and len(data['gd']['val_losses']) > 0:
                combined_val_losses.extend(data['gd']['val_losses'])
            
            if 'train_accs' in data['gd'] and len(data['gd']['train_accs']) > 0:
                combined_train_accs.extend(data['gd']['train_accs'])
            
            if 'val_accs' in data['gd'] and len(data['gd']['val_accs']) > 0:
                combined_val_accs.extend(data['gd']['val_accs'])
                
            valley_offset = adjusted_gd_steps[-1] if len(adjusted_gd_steps) > 0 else gd_offset
        
        # Add Valley stage data
        if has_valley:
            valley_steps = data['valley']['steps']
            valley_losses = data['valley']['train_losses']
            
            # Adjust Valley steps to continue from GD
            adjusted_valley_steps = [step + valley_offset for step in valley_steps]
            combined_train_losses.extend(valley_losses)
            combined_steps.extend(adjusted_valley_steps)
            
            if 'val_losses' in data['valley'] and len(data['valley']['val_losses']) > 0:
                combined_val_losses.extend(data['valley']['val_losses'])
            
            if 'train_accs' in data['valley'] and len(data['valley']['train_accs']) > 0:
                combined_train_accs.extend(data['valley']['train_accs'])
            
            if 'val_accs' in data['valley'] and len(data['valley']['val_accs']) > 0:
                combined_val_accs.extend(data['valley']['val_accs'])
        
        # Save combined data
        if len(combined_train_losses) > 0:
            # Create directory for optimizer if it doesn't exist
            optimizer_dir = os.path.dirname(data['train'].get('file_path', ''))
            if not optimizer_dir or not os.path.exists(optimizer_dir):
                continue
                
            # Save combined trajectory
            combined_file = os.path.join(optimizer_dir, f'combined_trajectory_{optimizer_name}.npz')
            np.savez(
                combined_file,
                train_losses=np.array(combined_train_losses),
                val_losses=np.array(combined_val_losses) if combined_val_losses else np.array([]),
                train_accs=np.array(combined_train_accs) if combined_train_accs else np.array([]),
                val_accs=np.array(combined_val_accs) if combined_val_accs else np.array([]),
                steps=np.array(combined_steps)
            )
            print(f"[saved] {combined_file}")
            
            # Create combined plots for this optimizer
            plt.figure(figsize=(12, 8))
            
            # Plot losses
            plt.subplot(2, 1, 1)
            plt.plot(combined_steps, combined_train_losses, label="Train Loss")
            if combined_val_losses:
                # Create evenly spaced validation steps
                val_steps = np.linspace(combined_steps[0], combined_steps[-1], len(combined_val_losses))
                plt.plot(val_steps, combined_val_losses, label="Val Loss")
            
            # Add vertical lines to separate training stages
            if has_train and has_gd:
                plt.axvline(x=train_offset, color='gray', linestyle='--', alpha=0.7, label="Train → GD")
            if has_gd and has_valley:
                plt.axvline(x=valley_offset, color='gray', linestyle='--', alpha=0.7, label="GD → Valley")
                
            plt.title(f"{optimizer_name.upper()} Combined Losses")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.legend()
            
            # Plot accuracies
            if combined_train_accs or combined_val_accs:
                plt.subplot(2, 1, 2)
                if combined_train_accs:
                    # Create evenly spaced train accuracy steps
                    train_acc_steps = np.linspace(combined_steps[0], combined_steps[-1], len(combined_train_accs))
                    plt.plot(train_acc_steps, combined_train_accs, label="Train Acc")
                if combined_val_accs:
                    # Create evenly spaced validation accuracy steps
                    val_acc_steps = np.linspace(combined_steps[0], combined_steps[-1], len(combined_val_accs))
                    plt.plot(val_acc_steps, combined_val_accs, label="Val Acc")
                
                # Add vertical lines to separate training stages
                if has_train and has_gd:
                    plt.axvline(x=train_offset, color='gray', linestyle='--', alpha=0.7, label="Train → GD")
                if has_gd and has_valley:
                    plt.axvline(x=valley_offset, color='gray', linestyle='--', alpha=0.7, label="GD → Valley")
                    
                plt.title(f"{optimizer_name.upper()} Combined Accuracy")
                plt.xlabel("Steps")
                plt.ylabel("Accuracy")
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(optimizer_dir, f'combined_plots_{optimizer_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()


def plot_combined_losses(optimizers_data, output_dir, log_scale=False):
    """
    Create combined loss plots for all optimizers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot train losses for all optimizers and stages
    plt.figure(figsize=(12, 8))
    
    # Dictionary to store stage transition points for each optimizer
    stage_transitions = {}
    
    for optimizer_name, data in optimizers_data.items():
        # Calculate offsets for connecting plots
        train_offset = 0
        gd_offset = 0
        valley_offset = 0
        
        # Store transitions for this optimizer
        stage_transitions[optimizer_name] = {}
        
        if 'train' in data and 'train_losses' in data['train']:
            train_steps = data['train']['steps']
            train_losses = data['train']['train_losses']
            if len(train_steps) > 0 and len(train_losses) > 0:
                plt.plot(train_steps, train_losses, label=f"{optimizer_name} - Train")
                train_offset = train_steps[-1]
                gd_offset = train_offset
                stage_transitions[optimizer_name]['train_end'] = train_offset
        
        if 'gd' in data and 'train_losses' in data['gd']:
            gd_steps = data['gd']['steps']
            gd_losses = data['gd']['train_losses']
            if len(gd_steps) > 0 and len(gd_losses) > 0:
                # Adjust GD steps to continue from training
                adjusted_gd_steps = gd_steps + gd_offset
                plt.plot(adjusted_gd_steps, gd_losses, label=f"{optimizer_name} - GD")
                valley_offset = adjusted_gd_steps[-1]
                stage_transitions[optimizer_name]['gd_end'] = valley_offset
        
        if 'valley' in data and 'train_losses' in data['valley']:
            valley_steps = data['valley']['steps']
            valley_losses = data['valley']['train_losses']
            if len(valley_steps) > 0 and len(valley_losses) > 0:
                # Adjust Valley steps to continue from GD
                adjusted_valley_steps = valley_steps + valley_offset
                plt.plot(adjusted_valley_steps, valley_losses, label=f"{optimizer_name} - Valley")
                stage_transitions[optimizer_name]['valley_end'] = adjusted_valley_steps[-1]
    
    # Add vertical lines to separate stages - use the first optimizer's transitions as reference
    if stage_transitions:
        reference_optimizer = list(stage_transitions.keys())[0]
        if 'train_end' in stage_transitions[reference_optimizer]:
            plt.axvline(x=stage_transitions[reference_optimizer]['train_end'], color='gray', linestyle='--', alpha=0.7, label="Train → GD")
        if 'gd_end' in stage_transitions[reference_optimizer]:
            plt.axvline(x=stage_transitions[reference_optimizer]['gd_end'], color='gray', linestyle='--', alpha=0.7, label="GD → Valley")
    
    plt.title('Combined Training Losses Across Stages')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(output_dir, 'combined_losses_loglog.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_dir, 'combined_losses.png'), dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # Plot validation losses if available
    has_val_losses = False
    for optimizer_name, data in optimizers_data.items():
        for stage in ['train', 'gd', 'valley']:
            if stage in data and 'val_losses' in data[stage] and len(data[stage]['val_losses']) > 0:
                has_val_losses = True
                break
        if has_val_losses:
            break
    
    if has_val_losses:
        plt.figure(figsize=(12, 8))
        
        for optimizer_name, data in optimizers_data.items():
            # Calculate offsets for connecting plots
            train_offset = 0
            gd_offset = 0
            valley_offset = 0
            
            if 'train' in data and 'val_losses' in data['train'] and len(data['train']['val_losses']) > 0:
                train_steps = data['train']['steps']
                val_losses = data['train']['val_losses']
                # Create evenly spaced validation steps
                val_steps = np.linspace(0, train_steps[-1], len(val_losses))
                plt.plot(val_steps, val_losses, label=f"{optimizer_name} - Train (Val)")
                train_offset = train_steps[-1]
                gd_offset = train_offset
            
            if 'gd' in data and 'val_losses' in data['gd'] and len(data['gd']['val_losses']) > 0:
                gd_steps = data['gd']['steps']
                val_losses = data['gd']['val_losses']
                if len(gd_steps) > 0:
                    # Create evenly spaced validation steps
                    val_steps = np.linspace(0, gd_steps[-1], len(val_losses))
                    # Adjust GD steps to continue from training
                    adjusted_val_steps = val_steps + gd_offset
                    plt.plot(adjusted_val_steps, val_losses, label=f"{optimizer_name} - GD (Val)")
                    valley_offset = gd_steps[-1] + gd_offset
            
            if 'valley' in data and 'val_losses' in data['valley'] and len(data['valley']['val_losses']) > 0:
                valley_steps = data['valley']['steps']
                val_losses = data['valley']['val_losses']
                if len(valley_steps) > 0:
                    # Create evenly spaced validation steps
                    val_steps = np.linspace(0, valley_steps[-1], len(val_losses))
                    # Adjust Valley steps to continue from GD
                    adjusted_val_steps = val_steps + valley_offset
                    plt.plot(adjusted_val_steps, val_losses, label=f"{optimizer_name} - Valley (Val)")
        
        # Add vertical lines to separate stages - use the first optimizer's transitions as reference
        if stage_transitions:
            reference_optimizer = list(stage_transitions.keys())[0]
            if 'train_end' in stage_transitions[reference_optimizer]:
                plt.axvline(x=stage_transitions[reference_optimizer]['train_end'], color='gray', linestyle='--', alpha=0.7, label="Train → GD")
            if 'gd_end' in stage_transitions[reference_optimizer]:
                plt.axvline(x=stage_transitions[reference_optimizer]['gd_end'], color='gray', linestyle='--', alpha=0.7, label="GD → Valley")
        
        plt.title('Combined Validation Losses Across Stages')
        plt.xlabel('Steps')
        plt.ylabel('Validation Loss')
        plt.legend()
        
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(os.path.join(output_dir, 'combined_val_losses_loglog.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(output_dir, 'combined_val_losses.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
        
    # Plot train accuracy for all optimizers and stages
    has_train_accs = False
    for optimizer_name, data in optimizers_data.items():
        for stage in ['train', 'gd', 'valley']:
            if stage in data and 'train_accs' in data[stage] and len(data[stage]['train_accs']) > 0:
                has_train_accs = True
                break
        if has_train_accs:
            break
    
    if has_train_accs:
        plt.figure(figsize=(12, 8))
        
        for optimizer_name, data in optimizers_data.items():
            # Calculate offsets for connecting plots
            train_offset = 0
            gd_offset = 0
            valley_offset = 0
            
            if 'train' in data and 'train_accs' in data['train'] and len(data['train']['train_accs']) > 0:
                train_steps = data['train']['steps']
                train_accs = data['train']['train_accs']
                plt.plot(train_steps, train_accs, label=f"{optimizer_name} - Train")
                train_offset = train_steps[-1]
                gd_offset = train_offset
            
            if 'gd' in data and 'train_accs' in data['gd'] and len(data['gd']['train_accs']) > 0:
                gd_steps = data['gd']['steps']
                train_accs = data['gd']['train_accs']
                if len(gd_steps) > 0:
                    # Adjust GD steps to continue from training
                    adjusted_gd_steps = gd_steps + gd_offset
                    plt.plot(adjusted_gd_steps, train_accs, label=f"{optimizer_name} - GD")
                    valley_offset = adjusted_gd_steps[-1]
            
            if 'valley' in data and 'train_accs' in data['valley'] and len(data['valley']['train_accs']) > 0:
                valley_steps = data['valley']['steps']
                train_accs = data['valley']['train_accs']
                if len(valley_steps) > 0:
                    # Adjust Valley steps to continue from GD
                    adjusted_valley_steps = valley_steps + valley_offset
                    plt.plot(adjusted_valley_steps, train_accs, label=f"{optimizer_name} - Valley")
        
        # Add vertical lines to separate stages - use the first optimizer's transitions as reference
        if stage_transitions:
            reference_optimizer = list(stage_transitions.keys())[0]
            if 'train_end' in stage_transitions[reference_optimizer]:
                plt.axvline(x=stage_transitions[reference_optimizer]['train_end'], color='gray', linestyle='--', alpha=0.7, label="Train → GD")
            if 'gd_end' in stage_transitions[reference_optimizer]:
                plt.axvline(x=stage_transitions[reference_optimizer]['gd_end'], color='gray', linestyle='--', alpha=0.7, label="GD → Valley")
        
        plt.title('Combined Training Accuracy Across Stages')
        plt.xlabel('Steps')
        plt.ylabel('Training Accuracy')
        plt.legend()
        
        if log_scale:
            plt.xscale('log')
            plt.savefig(os.path.join(output_dir, 'combined_train_accs_loglog.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(output_dir, 'combined_train_accs.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
        
    # Plot validation accuracy for all optimizers and stages
    has_val_accs = False
    for optimizer_name, data in optimizers_data.items():
        for stage in ['train', 'gd', 'valley']:
            if stage in data and 'val_accs' in data[stage] and len(data[stage]['val_accs']) > 0:
                has_val_accs = True
                break
        if has_val_accs:
            break
    
    if has_val_accs:
        plt.figure(figsize=(12, 8))
        
        for optimizer_name, data in optimizers_data.items():
            # Calculate offsets for connecting plots
            train_offset = 0
            gd_offset = 0
            valley_offset = 0
            
            if 'train' in data and 'val_accs' in data['train'] and len(data['train']['val_accs']) > 0:
                train_steps = data['train']['steps']
                val_accs = data['train']['val_accs']
                # Create evenly spaced validation steps
                val_steps = np.linspace(0, train_steps[-1], len(val_accs))
                plt.plot(val_steps, val_accs, label=f"{optimizer_name} - Train (Val)")
                train_offset = train_steps[-1]
                gd_offset = train_offset
            
            if 'gd' in data and 'val_accs' in data['gd'] and len(data['gd']['val_accs']) > 0:
                gd_steps = data['gd']['steps']
                val_accs = data['gd']['val_accs']
                if len(gd_steps) > 0:
                    # Create evenly spaced validation steps
                    val_steps = np.linspace(0, gd_steps[-1], len(val_accs))
                    # Adjust GD steps to continue from training
                    adjusted_val_steps = val_steps + gd_offset
                    plt.plot(adjusted_val_steps, val_accs, label=f"{optimizer_name} - GD (Val)")
                    valley_offset = gd_steps[-1] + gd_offset
            
            if 'valley' in data and 'val_accs' in data['valley'] and len(data['valley']['val_accs']) > 0:
                valley_steps = data['valley']['steps']
                val_accs = data['valley']['val_accs']
                if len(valley_steps) > 0:
                    # Create evenly spaced validation steps
                    val_steps = np.linspace(0, valley_steps[-1], len(val_accs))
                    # Adjust Valley steps to continue from GD
                    adjusted_val_steps = val_steps + valley_offset
                    plt.plot(adjusted_val_steps, val_accs, label=f"{optimizer_name} - Valley (Val)")
        
        # Add vertical lines to separate stages - use the first optimizer's transitions as reference
        if stage_transitions:
            reference_optimizer = list(stage_transitions.keys())[0]
            if 'train_end' in stage_transitions[reference_optimizer]:
                plt.axvline(x=stage_transitions[reference_optimizer]['train_end'], color='gray', linestyle='--', alpha=0.7, label="Train → GD")
            if 'gd_end' in stage_transitions[reference_optimizer]:
                plt.axvline(x=stage_transitions[reference_optimizer]['gd_end'], color='gray', linestyle='--', alpha=0.7, label="GD → Valley")
        
        plt.title('Combined Validation Accuracy Across Stages')
        plt.xlabel('Steps')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        
        if log_scale:
            plt.xscale('log')
            plt.savefig(os.path.join(output_dir, 'combined_val_accs_loglog.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(output_dir, 'combined_val_accs.png'), dpi=300, bbox_inches='tight')
        
        plt.close()

def main():
    args = parse_args()
    
    # Parse optimizer list
    optimizer_list = args.optimizer_list.split(',')
    
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, 'combined_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for each optimizer
    optimizers_data = {}
    for optimizer in optimizer_list:
        optimizer_dir = os.path.join(args.results_dir, optimizer)
        if os.path.exists(optimizer_dir):
            print(f"Loading data for optimizer: {optimizer}")
            optimizer_data = load_losses(optimizer_dir)
            # Store the directory path for later use
            for stage in optimizer_data:
                if isinstance(optimizer_data[stage], dict):
                    optimizer_data[stage]['file_path'] = os.path.join(optimizer_dir, f'loss_trajectory_{stage}.npz')
            optimizers_data[optimizer] = optimizer_data
        else:
            print(f"Warning: Directory for optimizer {optimizer} not found at {optimizer_dir}")
    
    # Save combined trajectories for each optimizer
    print("Saving combined trajectories for each optimizer...")
    save_combined_trajectories(optimizers_data)
    
    # Create plots
    print("Creating combined loss plots...")
    plot_combined_losses(optimizers_data, output_dir, log_scale=False)
    
    # Create log-log plots if requested
    if args.log_scale:
        print("Creating log-log scale plots...")
        plot_combined_losses(optimizers_data, output_dir, log_scale=True)
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
