#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def analyze_hessian_eigenvalues(hessian_path, output_dir=None):
    """
    Analyze eigenvalues of averaged Hessian matrix
    
    Args:
        hessian_path: Path to averaged hessian file
        output_dir: Directory to save plots (optional)
    """
    print(f"Analyzing Hessian: {hessian_path}")
    
    # Load averaged hessian
    hessian = torch.load(hessian_path, map_location='cpu')
    print(f"Hessian shape: {hessian.shape}")
    
    # Compute eigenvalues
    eigenvals = torch.linalg.eigvals(hessian).real
    eigenvals = eigenvals[eigenvals.abs() > 1e-10]  # Filter very small eigenvalues
    eigenvals_sorted = torch.sort(eigenvals, descending=True)[0]
    
    print(f"\nEigenvalue Statistics:")
    print(f"  Total eigenvalues: {len(eigenvals_sorted)}")
    print(f"  Max eigenvalue: {eigenvals_sorted.max():.6f}")
    print(f"  Min eigenvalue: {eigenvals_sorted.min():.6f}")
    print(f"  Positive eigenvalues: {(eigenvals_sorted > 0).sum()}")
    print(f"  Negative eigenvalues: {(eigenvals_sorted < 0).sum()}")
    print(f"  Zero eigenvalues: {(eigenvals_sorted.abs() < 1e-10).sum()}")
    
    if len(eigenvals_sorted) > 0:
        print(f"  Condition number: {eigenvals_sorted.max() / eigenvals_sorted[eigenvals_sorted > 1e-10].min():.2e}")
        print(f"  Trace: {eigenvals_sorted.sum():.6f}")
        print(f"  Determinant sign: {torch.sign(torch.prod(eigenvals_sorted[eigenvals_sorted.abs() > 1e-10]))}")
    
    # Create plots if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename for plots
        base_name = os.path.splitext(os.path.basename(hessian_path))[0]
        
        # Plot 1: Eigenvalue spectrum
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(eigenvals_sorted.numpy(), 'o-', markersize=3)
        plt.title(f'Eigenvalue Spectrum\n{base_name}')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        plt.hist(eigenvals_sorted.numpy(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'Eigenvalue Distribution\n{base_name}')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{base_name}_eigenvalues.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved eigenvalue plot: {plot_path}")
        
        # Plot 2: Hessian heatmap (if not too large)
        if hessian.shape[0] <= 100:
            plt.figure(figsize=(10, 8))
            plt.imshow(hessian.numpy(), cmap='RdBu_r', aspect='auto')
            plt.colorbar(label='Hessian value')
            plt.title(f'Hessian Matrix Heatmap\n{base_name}')
            plt.xlabel('Parameter index')
            plt.ylabel('Parameter index')
            
            heatmap_path = os.path.join(output_dir, f'{base_name}_heatmap.png')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved heatmap: {heatmap_path}")
        else:
            print(f"  Hessian too large ({hessian.shape[0]}x{hessian.shape[1]}) for heatmap visualization")
    
    return {
        'eigenvalues': eigenvals_sorted,
        'max_eigenval': eigenvals_sorted.max().item(),
        'min_eigenval': eigenvals_sorted.min().item(),
        'n_positive': (eigenvals_sorted > 0).sum().item(),
        'n_negative': (eigenvals_sorted < 0).sum().item(),
        'trace': eigenvals_sorted.sum().item(),
        'condition_number': (eigenvals_sorted.max() / eigenvals_sorted[eigenvals_sorted > 1e-10].min()).item() if len(eigenvals_sorted[eigenvals_sorted > 1e-10]) > 0 else float('inf')
    }


def compare_hessians(results_dir, output_dir=None):
    """
    Compare multiple averaged Hessians in a directory
    """
    print(f"Comparing Hessians in: {results_dir}")
    
    # Find all averaged hessian files
    hessian_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith("hessians_traj_") and filename.endswith("_averaged.pt"):
            hessian_files.append(filename)
    
    if not hessian_files:
        print("No averaged hessian files found!")
        return
    
    print(f"Found {len(hessian_files)} averaged hessian files")
    
    results = {}
    for filename in tqdm(hessian_files, desc="Analyzing hessians"):
        hessian_path = os.path.join(results_dir, filename)
        try:
            result = analyze_hessian_eigenvalues(hessian_path, output_dir)
            results[filename] = result
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            continue
    
    # Create comparison plots
    if output_dir and len(results) > 1:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract learning rates from filenames
        lrs = []
        max_eigenvals = []
        min_eigenvals = []
        condition_numbers = []
        traces = []
        
        for filename, result in results.items():
            # Extract learning rate from filename like "hessians_traj_lr0.1_averaged.pt"
            try:
                lr_str = filename.split('lr')[1].split('_')[0]
                lr = float(lr_str)
                lrs.append(lr)
                max_eigenvals.append(result['max_eigenval'])
                min_eigenvals.append(result['min_eigenval'])
                condition_numbers.append(result['condition_number'])
                traces.append(result['trace'])
            except:
                continue
        
        if lrs:
            # Sort by learning rate
            sorted_indices = np.argsort(lrs)
            lrs = np.array(lrs)[sorted_indices]
            max_eigenvals = np.array(max_eigenvals)[sorted_indices]
            min_eigenvals = np.array(min_eigenvals)[sorted_indices]
            condition_numbers = np.array(condition_numbers)[sorted_indices]
            traces = np.array(traces)[sorted_indices]
            
            # Comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].semilogx(lrs, max_eigenvals, 'o-')
            axes[0, 0].set_title('Max Eigenvalue vs Learning Rate')
            axes[0, 0].set_xlabel('Learning Rate')
            axes[0, 0].set_ylabel('Max Eigenvalue')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].semilogx(lrs, min_eigenvals, 'o-')
            axes[0, 1].set_title('Min Eigenvalue vs Learning Rate')
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Min Eigenvalue')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].loglog(lrs, condition_numbers, 'o-')
            axes[1, 0].set_title('Condition Number vs Learning Rate')
            axes[1, 0].set_xlabel('Learning Rate')
            axes[1, 0].set_ylabel('Condition Number')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].semilogx(lrs, traces, 'o-')
            axes[1, 1].set_title('Trace vs Learning Rate')
            axes[1, 1].set_xlabel('Learning Rate')
            axes[1, 1].set_ylabel('Trace')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'hessian_comparison.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison plot: {comparison_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze averaged Hessian matrices")
    parser.add_argument('--results_dir', required=True,
                       help='Directory containing averaged hessian files')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save analysis plots')
    parser.add_argument('--single_file', default=None,
                       help='Analyze single file instead of entire directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return
    
    if args.single_file:
        # Analyze single file
        hessian_path = os.path.join(args.results_dir, args.single_file)
        if not os.path.exists(hessian_path):
            print(f"Error: File {hessian_path} does not exist")
            return
        
        analyze_hessian_eigenvalues(hessian_path, args.output_dir)
    else:
        # Compare all hessians in directory
        compare_hessians(args.results_dir, args.output_dir)
    
    print("[done] Hessian analysis completed")


if __name__ == '__main__':
    main()
