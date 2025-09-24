#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def average_hessians_trajectory(hessians_path, output_path):
    """
    Load hessians trajectory and compute average over trajectory
    
    Args:
        hessians_path: Path to saved hessians trajectory (n_steps, n_params, n_params)
        output_path: Path to save averaged hessian (n_params, n_params)
    """
    print(f"Loading hessians from: {hessians_path}")
    
    # Load hessians trajectory
    hessians_traj = torch.load(hessians_path, map_location='cpu')
    
    if isinstance(hessians_traj, dict):
        # Handle old format if exists
        if 'hessians_trajectory' in hessians_traj:
            hessians_traj = hessians_traj['hessians_trajectory']
        else:
            raise ValueError(f"Unknown format in {hessians_path}")
    
    print(f"Hessians trajectory shape: {hessians_traj.shape}")
    
    # Compute average over trajectory (first dimension)
    averaged_hessian = torch.mean(hessians_traj, dim=0)
    
    print(f"Averaged hessian shape: {averaged_hessian.shape}")
    
    # Save averaged hessian
    torch.save(averaged_hessian, output_path)
    print(f"Saved averaged hessian to: {output_path}")
    
    return averaged_hessian


def find_setup_hessian_files(setup_num, base_results_dir="src/scripts/exp2/exp_results"):
    """
    Find all hessian files for a specific setup number
    
    Args:
        setup_num: Setup number (e.g., 1, 2, 3, ...)
        base_results_dir: Base directory containing setup directories
    
    Returns:
        List of tuples (setup_dir, hessian_files)
    """
    setup_pattern = f"setup{setup_num}"
    found_files = []
    
    # Look for setup directories
    if os.path.exists(base_results_dir):
        for item in os.listdir(base_results_dir):
            item_path = os.path.join(base_results_dir, item)
            if os.path.isdir(item_path) and setup_pattern in item:
                print(f"Found setup directory: {item_path}")
                
                # Find hessian files in this directory
                hessian_files = []
                for filename in os.listdir(item_path):
                    if filename.startswith("hessians_traj_") and filename.endswith(".pt") and "_averaged" not in filename:
                        hessian_files.append(filename)
                
                if hessian_files:
                    found_files.append((item_path, hessian_files))
                    print(f"  Found {len(hessian_files)} hessian files: {hessian_files}")
                else:
                    print(f"  No hessian files found in {item_path}")
    
    return found_files


def process_setup_hessians(setup_num, base_results_dir="src/scripts/exp2/exp_results", output_suffix="_averaged"):
    """
    Process all hessian files for a specific setup number
    
    Args:
        setup_num: Setup number to process
        base_results_dir: Base directory containing setup directories  
        output_suffix: Suffix for output files
    """
    print(f"Processing hessians for setup{setup_num}")
    
    setup_files = find_setup_hessian_files(setup_num, base_results_dir)
    
    if not setup_files:
        print(f"No setup directories found for setup{setup_num} in {base_results_dir}")
        return
    
    total_processed = 0
    
    for setup_dir, hessian_files in setup_files:
        print(f"\nProcessing directory: {setup_dir}")
        
        for filename in tqdm(hessian_files, desc=f"Processing {os.path.basename(setup_dir)}"):
            hessians_path = os.path.join(setup_dir, filename)
            
            # Create output filename
            base_name = filename.replace(".pt", "")
            output_filename = f"{base_name}{output_suffix}.pt"
            output_path = os.path.join(setup_dir, output_filename)
            
            try:
                # Average this hessian trajectory
                averaged_hessian = average_hessians_trajectory(hessians_path, output_path)
                
                # Print some statistics
                eigenvals = torch.linalg.eigvals(averaged_hessian).real
                eigenvals = eigenvals[eigenvals > 1e-10]  # Filter small eigenvalues
                if len(eigenvals) > 0:
                    print(f"  {filename}: max_eigenval={eigenvals.max():.6f}, "
                          f"min_eigenval={eigenvals.min():.6f}, "
                          f"n_positive={(eigenvals > 0).sum()}, "
                          f"n_negative={(eigenvals < 0).sum()}")
                
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print(f"\nTotal files processed: {total_processed}")


def process_all_hessians_in_directory(results_dir, output_suffix="_averaged"):
    """
    Process all hessian trajectory files in a directory
    
    Args:
        results_dir: Directory containing hessian trajectory files
        output_suffix: Suffix to add to output filenames
    """
    print(f"Processing hessians in directory: {results_dir}")
    
    # Find all hessian trajectory files
    hessian_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith("hessians_traj_") and filename.endswith(".pt"):
            hessian_files.append(filename)
    
    if not hessian_files:
        print("No hessian trajectory files found!")
        return
    
    print(f"Found {len(hessian_files)} hessian files: {hessian_files}")
    
    for filename in tqdm(hessian_files, desc="Processing hessian files"):
        hessians_path = os.path.join(results_dir, filename)
        
        # Create output filename
        base_name = filename.replace(".pt", "")
        output_filename = f"{base_name}{output_suffix}.pt"
        output_path = os.path.join(results_dir, output_filename)
        
        try:
            # Average this hessian trajectory
            averaged_hessian = average_hessians_trajectory(hessians_path, output_path)
            
            # Print some statistics
            eigenvals = torch.linalg.eigvals(averaged_hessian).real
            eigenvals = eigenvals[eigenvals > 1e-10]  # Filter small eigenvalues
            if len(eigenvals) > 0:
                print(f"  {filename}: max_eigenval={eigenvals.max():.6f}, "
                      f"min_eigenval={eigenvals.min():.6f}, "
                      f"n_positive={(eigenvals > 0).sum()}, "
                      f"n_negative={(eigenvals < 0).sum()}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Average Hessian trajectories over time")
    parser.add_argument('--results_dir', default='src/scripts/exp2/exp_results', 
                       help='Directory containing hessian trajectory files or base directory for setup search')
    parser.add_argument('--output_suffix', default='_averaged',
                       help='Suffix for output files (default: _averaged)')
    parser.add_argument('--single_file', default=None,
                       help='Process single file instead of entire directory')
    parser.add_argument('--setup', type=int, default=None,
                       help='Setup number to process (e.g., 1 for setup1). Will search for all setup{num} directories')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return
    
    if args.setup is not None:
        # Process specific setup number
        process_setup_hessians(args.setup, args.results_dir, args.output_suffix)
    elif args.single_file:
        # Process single file
        hessians_path = os.path.join(args.results_dir, args.single_file)
        if not os.path.exists(hessians_path):
            print(f"Error: File {hessians_path} does not exist")
            return
        
        base_name = args.single_file.replace(".pt", "")
        output_filename = f"{base_name}{args.output_suffix}.pt"
        output_path = os.path.join(args.results_dir, output_filename)
        
        averaged_hessian = average_hessians_trajectory(hessians_path, output_path)
        
        # Print statistics
        eigenvals = torch.linalg.eigvals(averaged_hessian).real
        eigenvals = eigenvals[eigenvals > 1e-10]
        if len(eigenvals) > 0:
            print(f"Eigenvalue statistics:")
            print(f"  Max eigenvalue: {eigenvals.max():.6f}")
            print(f"  Min eigenvalue: {eigenvals.min():.6f}")
            print(f"  Positive eigenvalues: {(eigenvals > 0).sum()}")
            print(f"  Negative eigenvalues: {(eigenvals < 0).sum()}")
            print(f"  Total eigenvalues: {len(eigenvals)}")
    else:
        # Process entire directory
        process_all_hessians_in_directory(args.results_dir, args.output_suffix)
    
    print("[done] Hessian averaging completed")


if __name__ == '__main__':
    main()
